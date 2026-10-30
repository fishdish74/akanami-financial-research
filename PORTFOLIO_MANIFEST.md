# Akanami portfolio sharing manifest

This file records the intended public scope of **Akanami — AI-Assisted Financial Research System**.

Proposed GitHub repository: `akanami-financial-research`

## Included

- Python source for the collector, analytics, validation, rendering, and safety boundaries
- deterministic tests that do not depend on private evaluation captures
- `.env.example` with empty credential fields and disabled write gates
- fictional and synthetic fixtures
- an example entity registry containing no personal watchlist
- public-facing architecture and safety documentation
- 525 passing public-safe tests across 30 test modules

## Intentionally excluded

- the original `.git` directory and all original commits, tags, author metadata, and machine paths
- the working `.env` and all credential values
- the working SQLite database, WAL files, backups, and raw SEC/provider cache
- the private entity registry and research universe
- real-run evaluation artifacts, model captures, and generated reports
- production-specific automation configuration and frozen operational identifiers
- internal continuity handoffs, hardening logs, and private backlog/history notes
- screenshots, because no verified sanitized UI screenshots were available

The original private project has 61 test modules. The 31 omitted modules depend on excluded production/evaluation materials, operational configuration, continuity evidence, or test fixtures transitively tied to them. No claim is made that the entire original private suite was run or passes.

## Publication status

This repository is prepared locally only. It has not been pushed to GitHub and should not be made public until the owner reviews the exact staged file list and explicitly approves publication.
