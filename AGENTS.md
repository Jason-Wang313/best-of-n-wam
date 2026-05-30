# Agent Notes

This repo uses the personal `context-handoff` skill for long-session continuity.

Before `/clear`, `/new`, or `/compact`, use:

`$context-handoff update docs/CODEX_HANDOFF.md before I clear context`

After clearing or starting a fresh session, use:

`$context-handoff resume from docs/CODEX_HANDOFF.md and verify against the repo`

Treat source files, tests, generated artifacts, and `scripts/claims_status.py` as the source of truth. Do not trust stale handoff claims without checking the repo. Keep `docs/CODEX_HANDOFF.md` factual, preserve only verified state, and mark unknowns as `UNKNOWN`.
