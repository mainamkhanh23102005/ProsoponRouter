# Track 1 Router

Hybrid token-efficient routing agent for AMD Developer Hackathon ACT II Track 1.

The container reads tasks from `TASK_INPUT_PATH`, routes each task through a deterministic solver when confidence is high, escalates to Fireworks when needed, and always writes a schema-valid `RESULTS_OUTPUT_PATH`.

## Current State

This is the Phase 1 scaffold plus Phase 3-ready interfaces:

- Defensive task loading from either `[{...}]` or `{"tasks": [...]}`
- Config-only hackathon assumptions in `src/config.py`
- Conservative deterministic solvers that return `None` instead of guessing
- Fireworks client with `DRY_RUN=1` mode
- Atomic result writing and per-task stderr logs
- End-to-end unittest using sample tasks

## Unknown Official Facts

Before scoring, fill these values from the official guide or Discord:

- `ALLOWED_MODELS`
- `CHEAPEST_MODEL`
- `FIREWORKS_API_KEY_ENV`
- exact input task schema
- exact output result schema
- category labels and scoring method
- accuracy threshold
- latency limit

All of those assumptions are isolated in `src/config.py` or environment variables.

## Local Run

From this folder:

```powershell
$env:DRY_RUN = "1"
$env:TASK_INPUT_PATH = "$PWD\sample_input\tasks.json"
$env:RESULTS_OUTPUT_PATH = "$PWD\out\results.json"
python -m src.main
Get-Content .\out\results.json
```

## Tests

```powershell
python -m unittest discover -s tests
```

## Docker

```powershell
docker build -t track1-router .
docker run --rm `
  -e DRY_RUN=1 `
  -v "${PWD}\sample_input:/input" `
  -v "${PWD}\out:/output" `
  track1-router
```

## Architecture

```mermaid
flowchart LR
    A[Task JSON] --> B[Classify]
    B --> C[Deterministic solver]
    C -->|validated| D[Result]
    C -->|decline| E[Fireworks client]
    E -->|validated| D
    E -->|fail| F[Fallback answer]
    F --> D
```

## Category Policy

The initial policy is conservative until the eval harness proves free-path accuracy:

| Category | Free path | Default |
| --- | --- | --- |
| math | safe arithmetic parser | enabled |
| sentiment | VADER or small fallback lexicon | enabled |
| ner | regex extraction for emails, dates, money, percent | enabled |
| summarization | disabled until dev-gated | Fireworks |
| factual knowledge | disabled except trivial future cases | Fireworks |
| code debugging | disabled until validated | Fireworks |
| code generation | disabled | Fireworks |
| logical reasoning | disabled until finite-domain patterns exist | Fireworks |

