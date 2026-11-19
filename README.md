# iteration-budget

A [Hermes](https://hermes-agent.nousresearch.com) plugin that makes the model
aware of its **per-turn iteration budget**, so it converges and writes a real
answer instead of being truncated mid-tool-call when the cap is reached.

Hermes caps one turn at N tool-calling iterations (`agent.max_turns`,
`HERMES_MAX_ITERATIONS`, or `--max-turns`) and hard-stops the turn at the cap.
Historically that was a silent truncation with no partial answer. This plugin
surfaces the whole budget from the first iteration and nudges the model toward
wrapping up as the cap approaches.

## What it does

- **Turn framing** (`pre_llm_call`): once per turn, states the cap and its cost
  model (a tool round costs 1, a reply with no tool call costs 0 and ends the turn).
- **Running count** (`llm_request` middleware): on every API call, appends a short
  status line to the **newest tool-result** — the same carrier Hermes core uses for
  its own budget warnings. It escalates through phases: heartbeat → warn → urgent →
  final → exhausted. It never touches the system prompt or the model's own messages.
- **execute_code refunds**: mirrors core's accounting so `execute_code`-only rounds
  don't count against the budget.
- **Delegation children**: marks `delegate_task` children so their cap comes from
  `delegation.max_iterations`.
- **Optional enforcement** (`pre_tool_call`, **off by default**): block new tool calls
  once the cap is spent so the model must answer. Stops after `max_enforce_blocks`
  so it can never park the turn — core still finishes it.
- **`/budget`** slash command: cap, used, remaining, phase thresholds, enforcement state
  (`/budget json` for machine-readable output).

Every hook is **fail-open**: an error in the plugin never breaks a turn.

## Install

```
hermes plugins install iteration-budget
```

## Configuration

All optional, under `plugins.entries.iteration-budget.settings`:

| key | default | meaning |
|---|---|---|
| `max_iterations` | follows Hermes | main-agent cap override; `"unlimited"` silences the plugin |
| `child_max_iterations` | follows `delegation.max_iterations` | delegate_task child cap override |
| `warn_ratio` / `urgent_ratio` / `final_ratio` | 0.70 / 0.85 / 0.94 | fraction of the cap where each phase begins |
| `heartbeat_every` | 10 | plain count line every N iterations in the roomy zone (0 = off) |
| `turn_framing` | true | state the cap once per turn |
| `inject` | true | append the running count to the outbound payload |
| `enforce` | false | block new tool calls once the cap is spent |
| `enforce_margin` / `max_enforce_blocks` | 1 / 3 | enforcement tuning |

## Behaviour disclosure

- **Modifies the outbound LLM request payload**: appends one status line to the
  newest user/tool-result message (never the system prompt or assistant messages),
  on `chat_completions` and `anthropic_messages` wire shapes (`codex_responses` is
  opt-in via `inject_codex_input`).
- Reads the `HERMES_MAX_ITERATIONS` environment variable, if set, to learn the cap,
  and reads core config (`agent.max_turns`, `delegation.max_iterations`) through
  `hermes_cli.config`.
- **No network calls, no file reads or writes, no credentials, no background
  processes, no telemetry.** Fully local. Uses public plugin APIs only — no runtime
  patching of Hermes core.

## Tests

```
python3 -I tests/selftest.py
```

No API calls and no core internals — drives the hooks and middleware directly with
fabricated payloads.

## License

MIT — see [LICENSE](LICENSE).
