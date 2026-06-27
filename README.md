# pi-omplike-advisor

A persistent **advisor** extension for [pi](https://github.com/badlogic/pi-mono):
a second model that reviews the main agent's work each turn and injects concise
advice inline. Port of oh-my-pi's advisor onto upstream pi's public extension
surface.

## What it does

The advisor is a long-lived, read-only `Agent` with its own model and read-only
tools (`read`/`grep`/`find`) plus one `advise` tool. It is fed the primary
agent's transcript one turn-delta at a time and may inject short advice back
into the conversation. It is **not** an executor: it cannot edit files, run
commands, or change session state.

Advice is emitted at three severities, with different delivery semantics:

- **nit** — delivered immediately (steered in and wakes an idle agent), tagged
  as raised about an earlier step. Low-stakes; mild staleness is fine.
- **concern** / **blocker** — always held on first emission, never steered
  immediately. Because review is asynchronous (seconds), high-severity advice is
  usually stale by the time it could land, so it is held and re-confirmed by the
  next review (the advisor re-raises survivors and stays silent on resolved
  ones).

While a high-severity note is held — or whenever a turn is about to idle — the
primary's next step is stalled (a **catch-up block**) so the advisor can catch
up. The wait backs off 15s → 30s → 60s … capped at 120s, is Escape-abortable,
and shows a notice. Nothing here is ever a hard interrupt; `abort()` is never
called.

## Installation

Add the package to your pi settings (`~/.pi/agent/settings.json`):

```json
{
  "packages": [
    "packages/pi-omplike-advisor"
  ]
}
```

(Or install from npm / a git checkout the same way you install other pi
packages.)

## Usage

- `/advisor` or `/advisor status` — show whether the advisor is on and which
  model it is using.
- `/advisor on` — enable the advisor (persisted; it is on by default unless
  explicitly turned off).
- `/advisor off` — disable the advisor (persisted).

## Configuration

### Advisor model

The advisor model defaults to **`openrouter/z-ai/glm-5.2`**. Override it by
adding an `advisor` entry to `modes.json` (project `.pi/modes.json` or global
`~/.pi/agent/modes.json`):

```json
{
  "modes": {
    "advisor": {
      "provider": "openrouter",
      "modelId": "z-ai/glm-5.2",
      "thinkingLevel": "low"
    }
  }
}
```

If the configured model can't be resolved, the advisor falls back to the current
session model.

### Project guidance (`WATCHDOG.md`)

If a `WATCHDOG.md` file exists in the working directory, its contents are
appended to the advisor's system prompt as advisor-only guidance (review
priorities, project traps, etc.). This lets you tune what the advisor watches
for without touching the main agent's prompt.

## Environment variables

- `ADVISOR_DEBUG=1` — verbose debug logging.
- `ADVISOR_NO_REVIEW=1` — skip live model review (keeps the deterministic
  `/advisor test` delivery path). Used by the test harness.

## Development

```bash
# fast, offline unit + loader + render tests
node packages/pi-omplike-advisor/extensions/advisor.test.mjs

# also run the live pi E2E harness (needs anthropic auth + network)
ADVISOR_E2E=1 node packages/pi-omplike-advisor/extensions/advisor.test.mjs
```
