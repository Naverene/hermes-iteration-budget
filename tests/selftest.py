#!/usr/bin/env python3
"""Self-test for the iteration-budget plugin — no API calls, no core internals.

Drives the plugin's hooks and middleware directly with fabricated Hermes hook kwargs and
request payloads (chat_completions + anthropic_messages + codex_responses shapes) and
asserts:

  * a turn-framing line is produced per turn, quoting the real cap
  * the running budget line lands in the newest tool-result tail, never in the system
    prompt or an assistant message
  * heartbeat / warn / urgent / final / exhausted fire at the configured ratios
  * execute_code-only iterations are refunded exactly as core refunds them (raw
    api_call_count and budget.used diverge the same way core reports them)
  * retries (same turn_id, same api_call_count) do not advance the count
  * a new turn_id restarts the record
  * injection is idempotent (no double append on a re-sent payload)
  * uncapped sessions produce nothing
  * enforce=true blocks tool calls at the cap and stops nagging after max_enforce_blocks
  * delegation children take delegation.max_iterations, not agent.max_turns
  * codex_responses ("input" transcripts) is opt-in
  * inject=false keeps framing + accounting but stops payload edits
  * /budget and on_session_reset behave
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
CONFIG = {"agent.max_turns": 20, "delegation.max_iterations": 8}
ENTRY = {"settings": {}}


class FakeCtx:
    """Just enough PluginContext for register()."""

    def __init__(self) -> None:
        self.hooks: dict[str, list] = {}
        self.middleware: dict[str, list] = {}
        self.commands: dict[str, dict] = {}

    def register_hook(self, name, handler, **_kw):
        self.hooks.setdefault(name, []).append(handler)

    def register_middleware(self, kind, callback, **_kw):
        if kind != "llm_request":
            raise ValueError(f"unsupported kind: {kind}")
        self.middleware.setdefault(kind, []).append(callback)

    def register_command(self, name, handler, **_kw):
        self.commands[name] = {"handler": handler, **_kw}

    def get_config(self, key, default=None):
        """Plugin-relative lookup, matching PluginsContext: reserved roots are rejected."""
        segments = str(key).split(".")
        if not key or segments[0].lower() in {"model", "plugins", "security", "settings"}:
            raise ValueError(
                "Expected a plugin-relative config key such as 'endpoint' or 'retry.policy'; "
                f"global, cross-plugin, and traversal paths are forbidden (got {key!r})"
            )
        node = ENTRY["settings"]
        for part in segments:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load(reload: bool = False):
    if reload and "iteration_budget_test" in sys.modules:
        del sys.modules["iteration_budget_test"]
    spec = importlib.util.spec_from_file_location("iteration_budget_test", PLUGIN_DIR / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["iteration_budget_test"] = module
    spec.loader.exec_module(module)
    module._config_lookup = lambda dotted: CONFIG.get(dotted)
    return module


FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + label + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def openai_payload(tool_text: str) -> dict:
    return {
        "model": "m",
        "messages": [
            {"role": "system", "content": "SYSTEM PROMPT MUST NOT CHANGE"},
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "calling a tool",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "earlier output"},
            {"role": "assistant", "content": "calling another tool",
             "tool_calls": [{"id": "c2", "type": "function",
                             "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c2", "content": tool_text},
        ],
    }


def anthropic_payload() -> dict:
    return {
        "model": "m",
        "system": "ANTHROPIC SYSTEM MUST NOT CHANGE",
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                                               "name": "terminal", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "tool output here"}]},
        ],
    }


def reset(mod, cfg_overrides=None, child_of: str = "", env_max: str | None = None) -> None:
    ENTRY["settings"] = dict(cfg_overrides or {})
    # Core (non-plugin) config is fixture-driven here so the test never reads the real config.yaml.
    mod._config_lookup = lambda dotted: CONFIG.get(dotted)
    if env_max is None:
        os.environ.pop("HERMES_MAX_ITERATIONS", None)
    else:
        os.environ["HERMES_MAX_ITERATIONS"] = env_max
    mod._TURNS.clear()
    mod._LATEST.clear()
    mod._UNCAPPED_SEEN.clear()
    if child_of:
        mod._CHILD_SESSIONS.add(child_of)


def framing(mod, session="s1", **extra):
    return mod._on_pre_llm_call(session_id=session, user_message="do the thing",
                                conversation_history=[], model="m", platform="matrix", **extra)


def drive(mod, session, turn, iterations, tools=("terminal",),
          api_mode="chat_completions", payload=None, start=1):
    """Fire middleware -> pre_api_request -> post_tool_call for iterations start..iterations."""
    seen = []
    for count in range(start, iterations + 1):
        request = payload if payload is not None else openai_payload(f"output {count}")
        request_id = f"{turn}-r{count}"
        result = mod._on_llm_request(request, session_id=session, task_id="task", turn_id=turn,
                                     api_request_id=request_id, api_mode=api_mode,
                                     api_call_count=count, model="m")
        mod._on_pre_api_request(session_id=session, task_id="task", turn_id=turn,
                                api_request_id=request_id, api_call_count=count,
                                platform="matrix", model="m", api_mode=api_mode)
        for name in tools:
            mod._on_post_tool_call(tool_name=name, tool_input={}, tool_result="ok",
                                   session_id=session, task_id="task", turn_id=turn,
                                   api_request_id=request_id, api_call_count=count)
        seen.append((count, request, result))
    return seen


def text_of(request: dict) -> str:
    return str(request["messages"][-1].get("content"))


M = "[iteration-budget]"
mod = load()
ctx = FakeCtx()
mod.register(ctx)

check("registers hooks + llm_request middleware + /budget",
      set(ctx.hooks) >= {"pre_llm_call", "pre_api_request", "post_tool_call", "pre_tool_call",
                         "subagent_start", "on_session_reset"}
      and "llm_request" in ctx.middleware and "budget" in ctx.commands,
      f"hooks={sorted(ctx.hooks)} commands={sorted(ctx.commands)}")

# ---------------------------------------------------------------- turn framing + phases
reset(mod)
first = framing(mod)
check("turn framing quotes the cap from agent.max_turns",
      bool(first) and "capped at 20" in first.get("context", ""),
      (first or {}).get("context", "")[:70])

seen = drive(mod, "s1", "t1", 21)
counts = {count: text_of(request) for count, request, _r in seen}
check("heartbeat count line at iteration 10", M in counts[10] and "10/20" in counts[10],
      counts[10][-95:])
check("iteration 11 stays quiet (heartbeat cadence)", M not in counts[11], counts[11][-60:])
check("warn at warn_ratio (14/20 = 0.70)", "converging" in counts[14] and "14/20" in counts[14],
      counts[14][-95:])
check("urgent at urgent_ratio (17/20 = 0.85)", "Close the step in flight" in counts[17],
      counts[17][-95:])
check("final at final_ratio (19/20 = 0.94)", "WRAP UP NOW" in counts[19], counts[19][-95:])
check("exhausted past the cap demands the answer", "SPENT" in counts[21]
      and "final answer now" in counts[21], counts[21][-95:])

request = seen[13][1]
check("system prompt untouched", "MUST NOT CHANGE" in request["messages"][0]["content"]
      and M not in request["messages"][0]["content"])
check("assistant output untouched", all(M not in str(m.get("content"))
      for m in request["messages"] if m.get("role") == "assistant"))
check("budget line rides the newest tool result only", M in request["messages"][-1]["content"]
      and M not in request["messages"][3]["content"])
check("middleware reports source + reason", seen[13][2]["source"] == "iteration-budget"
      and "budget warn" in seen[13][2]["reason"], str(seen[13][2]))

again = mod._on_llm_request(request, session_id="s1", task_id="task", turn_id="t1",
                            api_request_id="t1-r14", api_mode="chat_completions",
                            api_call_count=14, model="m")
check("injection is idempotent (one marker per carrier)", text_of(request).count(M) == 1,
      text_of(request)[-80:])

# ------------------------------------------------------------- execute_code refunds
seen = drive(mod, "s1", "t2", 10, tools=("execute_code",))
check("cheap execute_code rounds stay silent", all(r is None for _c, _r, r in seen))
seen = drive(mod, "s1", "t2", 24, tools=("terminal",), start=11)
check("refunded rounds lower the count (14/20, not 24/20)",
      "14/20" in text_of(seen[-1][1]) and "24/20" not in text_of(seen[-1][1]),
      text_of(seen[-1][1])[-95:])
seen = drive(mod, "s1", "t2", 25, tools=("terminal", "execute_code"), start=25)
check("mixed rounds (execute_code + another tool) are not refunded",
      "15/20" in text_of(seen[-1][1]), text_of(seen[-1][1])[-95:])

# ------------------------------------------------------------- retries + new turns
req = openai_payload("first attempt")
mod._on_llm_request(req, session_id="s1", task_id="task", turn_id="t4", api_request_id="t4-r1",
                    api_mode="chat_completions", api_call_count=1, model="m")
before = mod._used(mod._TURNS["s1"])
mod._on_llm_request(openai_payload("retry"), session_id="s1", task_id="task", turn_id="t4",
                    api_request_id="t4-r1", api_mode="chat_completions", api_call_count=1,
                    model="m")
after = mod._used(mod._TURNS["s1"])
check("retry (same turn_id + count) does not advance the count", before == after == 1,
      f"{before} -> {after}")
mod._on_llm_request(openai_payload("next turn"), session_id="s1", task_id="task", turn_id="t5",
                    api_request_id="t5-r1", api_mode="chat_completions", api_call_count=1,
                    model="m")
turn = mod._TURNS["s1"]
check("new turn_id restarts the record", turn.turn_id == "t5" and turn.last_count == 1
      and turn.refunds == 0, f"{turn.turn_id} count={turn.last_count} refunds={turn.refunds}")

# ---------------------------------------------------------------- uncapped sessions
saved = dict(CONFIG)
CONFIG["agent.max_turns"] = None
reset(mod)
first = framing(mod, "s_uncapped")
seen = drive(mod, "s_uncapped", "u1", 40)
check("uncapped session emits nothing", first is None
      and all(r is None for _c, _r, r in seen))
CONFIG.update(saved)

# --------------------------------------------------- HERMES_MAX_ITERATIONS precedence
saved_env = os.environ.get("HERMES_MAX_ITERATIONS")
reset(mod, env_max="30")
first = framing(mod, "s_env")
check("HERMES_MAX_ITERATIONS overrides agent.max_turns", bool(first)
      and "capped at 30" in first["context"], (first or {}).get("context", "")[:60])
reset(mod, {"max_iterations": 12})
first = framing(mod, "s_env2")
check("plugin max_iterations override wins", bool(first) and "capped at 12" in first["context"],
      (first or {}).get("context", "")[:60])
reset(mod, {"max_iterations": "unlimited"}, env_max="30")
first = framing(mod, "s_env3")
seen = drive(mod, "s_env3", "u3", 35)
check("max_iterations: 'unlimited' silences the plugin", first is None
      and all(r is None for _c, _r, r in seen))
if saved_env is None:
    os.environ.pop("HERMES_MAX_ITERATIONS", None)
else:
    os.environ["HERMES_MAX_ITERATIONS"] = saved_env

# ---------------------------------------------------------------- delegation children
reset(mod)
first = framing(mod, "child1", parent_session_id="s1")
seen = drive(mod, "child1", "tc", 9)
check("child cap comes from delegation.max_iterations (8)", bool(first)
      and "capped at 8" in first["context"] and "SPENT" in text_of(seen[-1][1]),
      text_of(seen[-1][1])[-70:])
mod._on_subagent_start(parent_session_id="s1", child_session_id="child2",
                       child_goal="probe", task_index=0, task_count=1)
child_cap = mod._resolve_cap("child2")
check("subagent_start marks children too", child_cap == 8, f"cap={child_cap}")

# ---------------------------------------------------------------- enforcement
# Semantics: block new tool calls once `used >= cap - enforce_margin`, i.e. when fewer
# iterations remain than the margin reserves for the wrap-up. Blocks stop nagging after
# max_enforce_blocks so a plugin can never park the turn (core finishes it its own way).
reset(mod, {"enforce": True, "enforce_margin": 1, "max_enforce_blocks": 2})
verdicts = []
for count in (18, 19, 20, 21):
    mod._on_llm_request(openai_payload("x"), session_id="s_enf", task_id="task", turn_id="te",
                        api_request_id=f"te-r{count}", api_mode="chat_completions",
                        api_call_count=count, model="m")
    verdicts.append(mod._on_pre_tool_call(tool_name="terminal", tool_input={},
                                          session_id="s_enf", task_id="task", turn_id="te",
                                          api_request_id=f"te-r{count}", api_call_count=count))
check("enforce runs tools while more than the margin remains (18/20, margin 1)",
      verdicts[0] is None, str(verdicts[0]))
check("enforce blocks once only the margin is left (19/20, margin 1)",
      bool(verdicts[1]) and verdicts[1]["action"] == "block" and M in verdicts[1]["message"],
      verdicts[1]["message"][:80] if verdicts[1] else "no block")
check("enforce blocks while the cap is spent (20/20)",
      bool(verdicts[2]) and verdicts[2]["action"] == "block",
      verdicts[2]["message"][:80] if verdicts[2] else "no block")
check("enforce stops nagging after max_enforce_blocks", verdicts[3] is None, str(verdicts[3]))

# margin 0 = block only when the budget is truly gone.
reset(mod, {"enforce": True, "enforce_margin": 0, "max_enforce_blocks": 5})
v0 = []
for count in (19, 20):
    mod._on_llm_request(openai_payload("x"), session_id="s_enf0", task_id="task", turn_id="t0",
                        api_request_id=f"t0-r{count}", api_mode="chat_completions",
                        api_call_count=count, model="m")
    v0.append(mod._on_pre_tool_call(tool_name="terminal", tool_input={}, session_id="s_enf0",
                                    task_id="task", turn_id="t0", api_request_id=f"t0-r{count}",
                                    api_call_count=count))
check("enforce_margin 0 blocks only at the spent cap", v0[0] is None and bool(v0[1]),
      f"19 -> {v0[0]}, 20 -> {'block' if v0[1] else None}")

# Real hook shape: pre_tool_call carries NO api_call_count (turn_id/session_id/tool_call_id
# only), so enforcement must fall back to the count the last pre_api_request recorded.
reset(mod, {"enforce": True, "enforce_margin": 0, "max_enforce_blocks": 5})
mod._on_llm_request(openai_payload("x"), session_id="s_enfreal", task_id="task", turn_id="tr",
                    api_request_id="tr-r20", api_mode="chat_completions",
                    api_call_count=20, model="m")
real_shape = mod._on_pre_tool_call(tool_name="terminal", tool_input={}, session_id="s_enfreal",
                                   task_id="task", tool_call_id="call-1", turn_id="tr",
                                   api_request_id="tr-r20", middleware_trace=[])
check("enforce works with the real pre_tool_call kwargs (no api_call_count)",
      bool(real_shape) and real_shape["action"] == "block",
      real_shape["message"][:60] if real_shape else "no block")
reset(mod)
check("enforce is off by default", mod._on_pre_tool_call(tool_name="terminal", tool_input={},
                                                         session_id="s_enf", task_id="task",
                                                         turn_id="te", api_request_id="x",
                                                         api_call_count=40) is None)

# ---------------------------------------------------------------- other wire shapes
reset(mod)
payload = anthropic_payload()
result = mod._on_llm_request(payload, session_id="s_anth", task_id="task", turn_id="ta",
                             api_request_id="ta-r15", api_mode="anthropic_messages",
                             api_call_count=15, model="m")
check("anthropic tool_result tail carries the line", bool(result)
      and M in str(payload["messages"][1]["content"][0]["content"]),
      str(payload["messages"][1]["content"][0]["content"])[-80:])
check("anthropic system field untouched", M not in payload["system"])

codex = {"model": "m", "input": [{"type": "function_call_output", "call_id": "c",
                                  "output": "codex tool output"}]}
result = mod._on_llm_request(codex, session_id="s_codex", task_id="task", turn_id="tx",
                             api_request_id="tx-r15", api_mode="codex_responses",
                             api_call_count=15, model="m")
check("codex_responses injection is off by default", result is None
      and M not in codex["input"][0]["output"])
reset(mod, {"inject_codex_input": True})
result = mod._on_llm_request(codex, session_id="s_codex", task_id="task", turn_id="tx",
                             api_request_id="tx-r16", api_mode="codex_responses",
                             api_call_count=16, model="m")
check("codex_responses injection works when opted in", bool(result)
      and M in codex["input"][0]["output"], codex["input"][0]["output"][-80:])

reset(mod, {"inject": False})
first = framing(mod, "s_quiet")
seen = drive(mod, "s_quiet", "tq", 19)
check("inject=false keeps framing + accounting, edits no payload",
      bool(first) and all(r is None for _c, _r, r in seen)
      and mod._LATEST["s_quiet"][0] == 19, str(mod._LATEST.get("s_quiet")))

# ---------------------------------------------------------------- /budget + reset
out = ctx.commands["budget"]["handler"]("", session_id="s_quiet")
check("/budget prints cap, used and remaining", "cap: 20" in out and "used this turn: 19" in out
      and "remaining: 1" in out, out.replace("\n", " | ")[:150])
data = json.loads(ctx.commands["budget"]["handler"]("json", session_id="s_quiet"))
check("/budget json mode round-trips", data.get("cap") == 20 and data.get("used") == 19
      and data.get("remaining") == 1, json.dumps(data)[:120])
check("/budget on an unknown session still reports the cap",
      "cap: 20" in ctx.commands["budget"]["handler"]("", session_id="nobody"))

mod._on_session_reset(session_id="s_quiet")
check("on_session_reset drops turn state", "s_quiet" not in mod._TURNS
      and "s_quiet" not in mod._LATEST)

# ---------------------------------------------------------------- fail-open hygiene
reset(mod)
try:
    for fn in (mod._on_pre_llm_call, mod._on_pre_api_request, mod._on_post_tool_call,
               mod._on_pre_tool_call, mod._on_subagent_start, mod._on_session_reset):
        fn(session_id=None, turn_id=None, api_request_id=None, api_call_count="not-a-number")
    survived = True
    error = ""
except Exception as exc:  # pragma: no cover - a crash here is the failure we are testing for
    survived = False
    error = repr(exc)
check("hooks tolerate malformed kwargs without raising", survived, error)
check("malformed counts make the tool hooks inert", all(
    fn(session_id="junk", turn_id=None, api_request_id=None, api_call_count="not-a-number") is None
    for fn in (mod._on_pre_api_request, mod._on_post_tool_call, mod._on_pre_tool_call)))
check("middleware tolerates a non-dict request",
      mod._on_llm_request("not a payload", session_id="s1", task_id="t", turn_id="t9",
                          api_request_id="r", api_mode="chat_completions",
                          api_call_count=19, model="m") is None)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("ALL CHECKS PASSED")
