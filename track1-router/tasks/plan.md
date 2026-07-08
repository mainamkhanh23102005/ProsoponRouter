# Implementation Plan

## Phase 1 Scaffold

Build one complete local path: read sample tasks, classify, route through deterministic or DRY_RUN Fireworks paths, write results atomically, and prove the flow with a unittest.

Acceptance:

- `python -m src.main` writes `out/results.json`
- every input task gets one output entry
- the process exits 0 even when individual tasks fail
- stderr logs category, route, and token count per task

## Phase 2 Deterministic Solvers

Improve solvers one category at a time with adversarial unit tests. A solver may only return an answer when it can self-validate with high confidence.

Acceptance:

- at least 10 tests per enabled solver
- low-confidence cases return `None`
- enabled free paths meet the configured dev accuracy threshold

## Phase 3 Fireworks Integration

Replace DRY_RUN with real model calls using only config-approved models and tight prompt/output budgets.

Acceptance:

- missing credentials degrade to fallback, not crash
- token usage is logged from API responses
- `temperature=0` on every call
- no secrets are logged

## Phase 4 Eval and Policy

Generate a local dev dataset, score free path and model path per category, and update `POLICY` from measured results.

Acceptance:

- `python -m eval.score` prints category by path accuracy and token use
- README contains one-line policy justification per category
- free paths are disabled when they do not clear the threshold

