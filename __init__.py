"""iteration-budget — make the model aware of Hermes' per-turn iteration budget.

Hermes caps one turn at N tool-calling iterations (``agent.max_turns``,
``HERMES_MAX_ITERATIONS`` or ``--max-turns``; uncapped when none is set) and hard-stops
the turn when the cap is reached — historically a silent truncation mid-tool-call with no
partial answer. Core's own ``agent.budget_warning_ratio`` (off by default) emits a single
warning close to the cap.

This plugin surfaces the whole budget from the first iteration so the model paces itself
and delivers a real answer before the cap bites:

  pre_llm_call    once per turn: the cap and its cost model (rides the user message)
  llm_request     every API call: running count + convergence nudge, appended to the
                  newest tool-result tail — the same carrier core uses for its own budget
                  warnings. Never touches the system prompt or model output.
  post_tool_call  bookkeeping; refunds execute_code-only iterations exactly as core does
  pre_tool_call   optional enforcement (OFF by default): block new tool calls at the cap
                  so the model must answer instead of being truncated
  subagent_start / pre_llm_call(parent_session_id)  mark delegation children so their cap
                  comes from ``delegation.max_iterations``
  on_session_reset  drop per-session state
  /budget         slash command: cap, used, remaining, thresholds, enforcement state

Public plugin API only — no core monkey-patching. Every hook is fail-open: an error here
never breaks a turn. All shared state is guarded by ``_LOCK`` so concurrent API calls or
delegation children (incl. free-threaded builds) cannot corrupt the turn records.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("hermes.plugins.iteration_budget")

PLUGIN_ID = "iteration-budget"
MARK = "[iteration-budget]"
# Fingerprint appended to every injected line. Idempotency and "did I already annotate
# this?" are decided by THIS tag, not by MARK — so a tool result that merely *mentions*
# MARK (the model ran /budget, or grepped this plugin) is never mistaken for our own
# injection. Plain ASCII on purpose: source stays free of invisible unicode, which
# plugin-catalog security scanners flag.
_INJECT_TAG = "[ib:wire]"
SETTINGS_PATH = "plugins.entries.iteration-budget.settings"

#: Hermes' own defaults, mirrored so the plugin reports the cap the agent actually gets.
_HERMES_DELEGATION_DEFAULT = 250

DEFAULTS: dict[str, Any] = {
    # Cap override for main agents. None -> HERMES_MAX_ITERATIONS -> agent.max_turns
    # (uncapped when neither is set). "unlimited" disables the plugin's awareness.
    "max_iterations": None,
    # Cap override for delegate_task children. None -> delegation.max_iterations.
    "child_max_iterations": None,
    # Phase thresholds (fraction of the cap consumed).
    "warn_ratio": 0.70,
    "urgent_ratio": 0.85,
    "final_ratio": 0.94,
    # Emit a plain count line every N iterations while still in the roomy zone (0 = off).
    "heartbeat_every": 10,
    # One framing line per turn via pre_llm_call.
    "turn_framing": True,
    # Inject the running status into the outbound request payload.
    "inject": True,
    # Wire shapes the injection is known to be safe for. codex_responses uses a different
    # transcript shape ("input") and is opt-in via inject_codex_input.
    "inject_wire_modes": ("chat_completions", "anthropic_messages"),
    "inject_codex_input": False,
    # Enforcement: refuse new tool calls once the cap is reached (default off — the model
    # is expected to self-regulate; core hard-stops at the cap anyway).
    "enforce": False,
    "enforce_margin": 1,       # iterations kept spare for the final answer
    "max_enforce_blocks": 3,   # stop nagging after N blocks; let core finish the turn
    "log": True,               # log milestones (agent.log / -v output)
}

_ctx: Any = None  # PluginContext, captured in register()


# --------------------------------------------------------------------- config helpers

def _cfg(key: str) -> Any:
    """One knob from ``plugins.entries.iteration-budget.settings.<key>`` (plugin-relative)."""
    if _ctx is None:
        return DEFAULTS[key]
    try:
        value = _ctx.get_config(key, None)
    except Exception:
        logger.debug("iteration-budget: rejected settings path %r", key)
        return DEFAULTS[key]
    return DEFAULTS[key] if value is None else value


def _ratio(key: str) -> float:
    try:
        value = float(_cfg(key))
    except (TypeError, ValueError):
        value = float(DEFAULTS[key])
    return min(1.0, max(0.05, value))


def _phase_at(cap: int, key: str) -> int:
    """Smallest ``used`` that trips this phase, matching the ``used/cap >= ratio`` test.

    ``int(cap * ratio)`` truncated *below* the real trigger (e.g. final_ratio 0.94 on a
    cap of 20 printed 18 while the phase only fires at used=19). ``ceil`` with a tiny
    epsilon lands on the same integer the live comparison uses.
    """
    return max(1, math.ceil(cap * _ratio(key) - 1e-9))


def _int_cfg(key: str) -> int:
    try:
        value = int(_cfg(key))
    except (TypeError, ValueError):
        value = int(DEFAULTS[key])
    return max(0, value)


def _bool_cfg(key: str) -> bool:
    value = _cfg(key)
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "off", "no", "none", ""}
    return bool(value)


def _wire_modes() -> tuple[str, ...]:
    value = _cfg("inject_wire_modes")
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(part) for part in value)
    return tuple(DEFAULTS["inject_wire_modes"])


def _coerce_cap(raw: Any) -> Optional[int]:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    text = str(raw).strip().lower()
    if not text or text in {"unlimited", "none", "null", "off", "no"}:
        return None
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if number > 0 else None


def _config_lookup(dotted: str) -> Any:
    """Read a core (non-plugin) config value such as ``agent.max_turns``."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        cfg = load_config_readonly() or {}
        return cfg_get(cfg, *dotted.split("."))
    except Exception:
        return None


def _resolve_cap(session_id: str) -> Optional[int]:
    """Return the finite iteration cap for this agent, or None when Hermes is uncapped."""
    if session_id in _CHILD_SESSIONS:
        explicit: Any = _cfg("child_max_iterations")
        if explicit is None:
            explicit = _config_lookup("delegation.max_iterations")
        if explicit is None:
            explicit = _HERMES_DELEGATION_DEFAULT
    else:
        explicit = _cfg("max_iterations")
        if explicit is None:
            explicit = (os.environ.get("HERMES_MAX_ITERATIONS") or "").strip() or None
        if explicit is None:
            explicit = _config_lookup("agent.max_turns")
    return _coerce_cap(explicit)


# ------------------------------------------------------------------------- turn state

@dataclass
class _Turn:
    turn_id: str = ""
    last_count: int = 0                 # highest api_call_count seen this turn
    refunds: int = 0                    # iterations core refunded (execute_code-only rounds)
    cap: Optional[int] = None
    tools: "dict[str, set[str]]" = field(default_factory=dict)   # api_request_id -> tool names
    last_phase: str = ""
    blocks: int = 0


# ``_LOCK`` guards every mutation of the module-global state below. It is an RLock so the
# handlers (which hold it for their whole body) can call the helpers freely. Individual
# dict/set ops are only atomic under the GIL; the compound sequences here (turn creation +
# eviction, refund settling) are not, and must not race on a free-threaded interpreter.
_LOCK = threading.RLock()
_TURNS: "OrderedDict[str, _Turn]" = OrderedDict()          # session_id -> _Turn
_MAX_TURNS = 64
_CHILD_SESSIONS: set[str] = set()
_LATEST: "dict[str, tuple[int, Optional[int]]]" = {}       # session_id -> (used, cap)
_UNCAPPED_SEEN: set[str] = set()


def _forget(session_id: str) -> None:
    """Drop every trace of a session. Caller holds _LOCK."""
    _TURNS.pop(session_id, None)
    _LATEST.pop(session_id, None)
    _CHILD_SESSIONS.discard(session_id)
    _UNCAPPED_SEEN.discard(session_id)


def _turn_for(session_id: str, turn_id: str, count: int) -> _Turn:
    """Return the live turn, starting a new one when the iteration count restarts.

    Caller holds _LOCK.
    """
    key = session_id or "_"
    turn = _TURNS.get(key)
    tid = turn_id or ""
    new_turn = (
        turn is None
        or (bool(tid) and bool(turn.turn_id) and tid != turn.turn_id)
        or count < turn.last_count
    )
    if new_turn:
        turn = _Turn(turn_id=tid)
        _TURNS[key] = turn
        _TURNS.move_to_end(key)
        # Evict the oldest sessions wholesale (not just their _TURNS entry) so the sibling
        # maps stay bounded alongside _TURNS instead of leaking one entry per session seen.
        while len(_TURNS) > _MAX_TURNS:
            old_key, _old = _TURNS.popitem(last=False)
            _LATEST.pop(old_key, None)
            _CHILD_SESSIONS.discard(old_key)
            _UNCAPPED_SEEN.discard(old_key)
    if tid and not turn.turn_id:
        turn.turn_id = tid
    turn.last_count = max(turn.last_count, count)
    return turn


def _settle_refunds(turn: _Turn, current_request_id: str) -> None:
    """Count finished iterations that core refunds (only tool used was execute_code).

    Caller holds _LOCK.
    """
    for request_id in list(turn.tools):
        if request_id == current_request_id:
            continue
        if turn.tools.get(request_id) == {"execute_code"}:
            turn.refunds += 1
        turn.tools.pop(request_id, None)


def _used(turn: _Turn) -> int:
    return max(0, turn.last_count - turn.refunds)


# ------------------------------------------------------------------------ the messages

def _status(used: int, cap: Optional[int]) -> Optional[tuple[str, str]]:
    """Phase + one-line instruction for the model, or None when there is nothing to say."""
    if not cap or used <= 0:
        return None
    remaining = cap - used
    ratio = used / cap
    if remaining <= 0:
        return ("exhausted",
                f"{MARK} iteration budget SPENT ({used}/{cap}) — no iterations left, Hermes "
                "will not run another tool round this turn. Write the final answer now: what "
                "is done, what is verified, what is still open, with exact paths/IDs.")
    if ratio >= _ratio("final_ratio"):
        return ("final",
                f"{MARK} {used}/{cap} iterations used, {remaining} left. WRAP UP NOW: answer "
                "from what you already have; take no further tool calls unless one is "
                "essential to the deliverable.")
    if ratio >= _ratio("urgent_ratio"):
        return ("urgent",
                f"{MARK} {used}/{cap} used, {remaining} left. Close the step in flight and "
                "start writing the answer; open no new lines of work.")
    if ratio >= _ratio("warn_ratio"):
        return ("warn",
                f"{MARK} {used}/{cap} used, {remaining} left. Start converging: finish "
                "in-flight work before any new exploration.")
    heartbeat = _int_cfg("heartbeat_every")
    # Floor scales with the cap (not a flat 5) so small delegation caps don't get a burst
    # of "near the end" heartbeats before the ratio phases even start. For small caps this
    # tail is effectively silent — warn/urgent/final already cover the final stretch.
    if heartbeat and (used % heartbeat == 0 or remaining <= max(1, cap // 10)):
        return ("heartbeat",
                f"{MARK} {used}/{cap} iterations used, {remaining} left. Each tool round "
                "costs 1; a pure answer costs 0 — keep headroom to finish the turn.")
    return None


def _append_to_text(text: str, line: str) -> str:
    return f"{text}\n\n{line}{_INJECT_TAG}"


def _already(line_owner: str) -> bool:
    return _INJECT_TAG in (line_owner or "")[-400:]


def _annotate_message(message: Any, line: str) -> Optional[bool]:
    """Annotate one message. Returns True/False, or None when the message is off-limits."""
    if not isinstance(message, dict):
        return False
    role = message.get("role")
    if role in {"assistant", "system"}:
        return None  # never rewrite the model's own text or the stable system-prompt message
    content = message.get("content")
    if isinstance(content, str):
        if _already(content):
            return True
        message["content"] = _append_to_text(content, line)
        return True
    if isinstance(content, list):
        for block in reversed(content):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                if _already(block["text"]):
                    return True
                block["text"] = _append_to_text(block["text"], line)
                return True
            if kind in {"tool_result", "tool_call_response"}:
                inner = block.get("content")
                if inner is None:
                    inner = block.get("output")
                if isinstance(inner, str):
                    if _already(inner):
                        return True
                    block["content"] = _append_to_text(inner, line)
                    return True
                if isinstance(inner, list):
                    for part in reversed(inner):
                        if (isinstance(part, dict) and part.get("type") == "text"
                                and isinstance(part.get("text"), str)):
                            if _already(part["text"]):
                                return True
                            part["text"] = _append_to_text(part["text"], line)
                            return True
        if role == "user":
            # Fresh block carries the fingerprint too, so a re-sent payload is not
            # annotated twice.
            content.append({"type": "text", "text": f"{line}{_INJECT_TAG}"})
            return True
        return False
    return False


def _annotate_request(request: Any, api_mode: str, line: str) -> bool:
    """Append the status line to the safest carrier in the outbound payload."""
    if not isinstance(request, dict):
        return False
    messages = request.get("messages")
    if isinstance(messages, list) and messages:
        if api_mode and api_mode not in _wire_modes():
            return False
        for message in reversed(messages):
            verdict = _annotate_message(message, line)
            if verdict is None:
                continue  # skip model/system rows, try the row before them
            return bool(verdict)
        return False
    # codex_responses carries the transcript as ``input`` (opt-in: shape differs).
    transcript = request.get("input")
    if isinstance(transcript, list) and transcript and _bool_cfg("inject_codex_input"):
        for item in reversed(transcript):
            if not isinstance(item, dict):
                continue
            for field_name in ("output", "content"):
                value = item.get(field_name)
                if isinstance(value, str):
                    if _already(value):
                        return True
                    item[field_name] = _append_to_text(value, line)
                    return True
            if isinstance(item.get("content"), list):
                if _annotate_message(item, line):
                    return True
        return False
    return False


# --------------------------------------------------------------------------- the hooks

def _on_pre_llm_call(**kwargs: Any) -> Optional[dict[str, Any]]:
    try:
        with _LOCK:
            session_id = str(kwargs.get("session_id") or "")
            if kwargs.get("parent_session_id"):
                _CHILD_SESSIONS.add(session_id)
            cap = _resolve_cap(session_id)
            _LATEST[session_id] = (_LATEST.get(session_id, (0, None))[0], cap)
            if cap is None:
                if session_id not in _UNCAPPED_SEEN:
                    _UNCAPPED_SEEN.add(session_id)
                    logger.debug(
                        "[iteration-budget] no finite cap for session %s — awareness disabled",
                        session_id,
                    )
                return None
            if not _bool_cfg("turn_framing"):
                return None
            text = (
                f"{MARK} This turn is capped at {cap} tool-call iterations: each round where you "
                "call a tool costs 1, a reply with no tool call costs 0 and ends the turn. Plan "
                "the steps up front and leave headroom (>=5) to write the answer. A running "
                f"count ({MARK} used/cap) is appended to your tool results as you go."
            )
            return {"context": text}
    except Exception:
        logger.debug("[iteration-budget] pre_llm_call failed", exc_info=True)
        return None


def _on_pre_api_request(**kwargs: Any) -> None:
    """Observer only: seed/refresh the turn record before the middleware chain runs next turn."""
    try:
        count = kwargs.get("api_call_count")
        if count is None:
            return
        count = int(count)
        with _LOCK:
            session_id = str(kwargs.get("session_id") or "")
            turn = _turn_for(session_id, str(kwargs.get("turn_id") or ""), count)
            _settle_refunds(turn, str(kwargs.get("api_request_id") or ""))
            turn.cap = _resolve_cap(session_id)
            _LATEST[session_id] = (_used(turn), turn.cap)
    except Exception:
        logger.debug("[iteration-budget] pre_api_request failed", exc_info=True)
        return None


def _on_llm_request(request: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
    """llm_request middleware: hand the model its remaining budget on every call."""
    try:
        raw_count = kwargs.get("api_call_count")
        if raw_count is None:
            return None
        count = int(raw_count)
        with _LOCK:
            session_id = str(kwargs.get("session_id") or "")
            request_id = str(kwargs.get("api_request_id") or "")
            turn = _turn_for(session_id, str(kwargs.get("turn_id") or ""), count)
            _settle_refunds(turn, request_id)
            if turn.cap is None:
                turn.cap = _resolve_cap(session_id)
            used = _used(turn)
            _LATEST[session_id] = (used, turn.cap)
            status = _status(used, turn.cap)
            if status is None:
                return None
            phase, line = status
            if _bool_cfg("log") and phase != turn.last_phase:
                turn.last_phase = phase
                logger.info(
                    "[iteration-budget] %s: %d/%d used, %d left (session=%s)",
                    phase, used, turn.cap or 0, max(0, (turn.cap or 0) - used), session_id,
                )
            if not _bool_cfg("inject"):
                return None
            if not _annotate_request(request, str(kwargs.get("api_mode") or ""), line):
                return None
            return {"request": request, "source": PLUGIN_ID, "reason": f"iteration budget {phase}"}
    except Exception:
        logger.debug("[iteration-budget] llm_request middleware failed", exc_info=True)
        return None


def _on_post_tool_call(**kwargs: Any) -> None:
    """Bookkeeping: remember which tools ran for this API request (execute_code refund)."""
    try:
        tool_name = str(kwargs.get("tool_name") or "")
        if not tool_name:
            return
        count = kwargs.get("api_call_count")
        if count is None:
            return
        with _LOCK:
            session_id = str(kwargs.get("session_id") or "")
            turn = _turn_for(session_id, str(kwargs.get("turn_id") or ""), int(count))
            key = str(kwargs.get("api_request_id") or "") or f"iter{int(count)}"
            turn.tools.setdefault(key, set()).add(tool_name)
            _LATEST[session_id] = (
                _used(turn),
                turn.cap if turn.cap is not None else _resolve_cap(session_id),
            )
    except Exception:
        logger.debug("[iteration-budget] post_tool_call failed", exc_info=True)
        return None


def _on_pre_tool_call(**kwargs: Any) -> Optional[dict[str, Any]]:
    """Optional enforcement: refuse new tool calls once the budget is spent."""
    try:
        if not _bool_cfg("enforce"):
            return None
        with _LOCK:
            session_id = str(kwargs.get("session_id") or "")
            count = kwargs.get("api_call_count")
            if count is None:
                used, cap = _LATEST.get(session_id, (0, None))
            else:
                turn = _turn_for(session_id, str(kwargs.get("turn_id") or ""), int(count))
                used, cap = _used(turn), (turn.cap if turn.cap is not None else _resolve_cap(session_id))
            if not cap:
                return None
            if used < cap - _int_cfg("enforce_margin"):
                return None
            max_blocks = _int_cfg("max_enforce_blocks")
            turn = _TURNS.get(session_id or "_")
            if turn is not None:
                if turn.blocks >= max_blocks:
                    return None
                turn.blocks += 1
            logger.info(
                "[iteration-budget] enforcement block of tool '%s' at %d/%d",
                kwargs.get("tool_name"), used, cap,
            )
            return {
                "action": "block",
                "message": (
                    f"{MARK} iteration budget {used}/{cap} reached — no new tool calls. Write the "
                    "final answer now from what you already have: what is done, what is verified, "
                    "what is still open."
                ),
            }
    except Exception:
        logger.debug("[iteration-budget] pre_tool_call failed", exc_info=True)
        return None


def _on_subagent_start(**kwargs: Any) -> None:
    try:
        child = str(kwargs.get("child_session_id") or kwargs.get("session_id") or "")
        if child:
            with _LOCK:
                _CHILD_SESSIONS.add(child)
    except Exception:
        logger.debug("[iteration-budget] subagent_start failed", exc_info=True)


def _on_session_reset(**kwargs: Any) -> None:
    try:
        session_id = str(kwargs.get("session_id") or kwargs.get("old_session_id") or "")
        if not session_id:
            return
        with _LOCK:
            _forget(session_id)
    except Exception:
        logger.debug("[iteration-budget] on_session_reset failed", exc_info=True)


# ------------------------------------------------------------------------ /budget command

def _cmd_budget(raw_args: str = "", **kwargs: Any) -> str:
    session_id = str(kwargs.get("session_id") or "")
    with _LOCK:
        try:
            used, cap = _LATEST.get(session_id, (0, None))
        except Exception:
            used, cap = 0, None
        if cap is None:
            cap = _resolve_cap(session_id)
        child = session_id in _CHILD_SESSIONS
    lines = [f"{MARK} iteration budget ({'delegation child' if child else 'main agent'})"]
    if cap is None:
        lines.append(
            "no finite cap: agent.max_turns / HERMES_MAX_ITERATIONS unset — Hermes runs uncapped"
        )
    else:
        lines.append(f"cap: {cap}")
        lines.append(f"used this turn: {used}")
        lines.append(f"remaining: {max(0, cap - used)}")
        lines.append(
            "phases: warn >=%d, urgent >=%d, final >=%d, exhausted at %d"
            % (
                _phase_at(cap, "warn_ratio"),
                _phase_at(cap, "urgent_ratio"),
                _phase_at(cap, "final_ratio"),
                cap,
            )
        )
    lines.append(f"inject: {_bool_cfg('inject')} | turn_framing: {_bool_cfg('turn_framing')} "
                 f"| enforce: {_bool_cfg('enforce')} (margin {_int_cfg('enforce_margin')})")
    lines.append(f"override cap: {SETTINGS_PATH}.max_iterations / .child_max_iterations")
    if (raw_args or "").strip().lower() in {"json", "--json"}:
        import json
        return json.dumps({
            "session_id": session_id, "child": child, "cap": cap, "used": used,
            "remaining": (max(0, cap - used) if cap else None),
            "warn_ratio": _ratio("warn_ratio"), "urgent_ratio": _ratio("urgent_ratio"),
            "final_ratio": _ratio("final_ratio"), "inject": _bool_cfg("inject"),
            "enforce": _bool_cfg("enforce"),
        })
    return "\n".join(lines)


# ---------------------------------------------------------------------------- register

def register(ctx: Any) -> None:
    global _ctx
    _ctx = ctx
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("subagent_start", _on_subagent_start)
    ctx.register_hook("on_session_reset", _on_session_reset)
    ctx.register_middleware("llm_request", _on_llm_request)
    ctx.register_command(
        "budget",
        handler=_cmd_budget,
        description="Show this session's iteration budget (cap, used, remaining)",
        args_hint="[json]",
    )
    logger.info(
        "[iteration-budget] loaded: inject=%s turn_framing=%s enforce=%s ratios=%s/%s/%s",
        _bool_cfg("inject"), _bool_cfg("turn_framing"), _bool_cfg("enforce"),
        _ratio("warn_ratio"), _ratio("urgent_ratio"), _ratio("final_ratio"),
    )
