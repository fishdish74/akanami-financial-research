# Akanami — AI-Assisted Financial Research System

Akanami is a safety-first Python system for collecting, filtering, analyzing, and rendering financial-research inputs. This portfolio edition is designed to run locally with synthetic data and without network access, paid model calls, or filesystem writes.

### Development Status: Work in Progress (Pre-V1)

Akanami is an actively developed AI-assisted financial research system. Core functionality has been implemented and tested, but the project has not yet satisfied its complete V1 acceptance criteria.

Some features remain incomplete or under independent review. The system is not production-ready and must not be relied upon for autonomous investment decisions.

This public repository is a sanitized portfolio demonstration of selected functionality. It does not contain the complete private development system, operational databases, or all internal evaluation and testing infrastructure.

The public demonstration is designed to run with synthetic data and without live model requests or production database writes.

The engineering principle is simple: AI may propose structured analysis, while trusted code owns validation, identifiers, budgets, paths, confidence rules, and write permissions.

## Engineering Philosophy

> Consistency without stubbornness. Rigor without unnecessary complexity. Progress without sacrificing correctness.

This philosophy guides Akanami's evidence-driven, contract-first methodology: keep implementation bounded, verify changes through regression testing and independent review, revise assumptions when evidence warrants it, and accept work only through controlled criteria that preserve correctness at every trust boundary.

> **Portfolio scope:** this repository contains source code, public-safe tests, and fictional fixtures. It does **not** contain the working database, provider caches, API credentials, private watchlist, production evaluation records, automation configuration, or original Git history.

Proposed GitHub repository: `akanami-financial-research`

## Implemented features

- Candidate normalization from manual fixtures, Google News RSS, and SEC submission metadata.
- Deterministic relevance, quality, and duplicate/event filters before optional paid analysis.
- Strict Pydantic contracts at source, model, and rendering boundaries.
- Structured AI-analysis support with code-owned identifiers, metadata, confidence derivation, usage accounting, and cost ceilings.
- Deterministic Markdown proposals and safe filename generation.
- Point-in-time financial data handling, quality checks, DCF scenarios, peer comparison, and opportunity/research-priority ranking logic.
- Synthetic forward-paper decision and evaluation logic with explicit persistence gates.
- Fail-closed controls for writes, redirects, malformed provider responses, incomplete pagination, incompatible periods, and unsafe paths.
- Offline fixtures and a public-safe automated test suite.

This is a research workflow, not a trading system. It does not connect to a broker or place trades.

## Current portfolio limitations

- The public demonstration is command-line only; no UI or sanitized screenshots are included.
- No live database, personal watchlist, provider cache, real evaluation capture, or generated research note is bundled.
- Network-backed sources and AI analysis require the user's own local credentials and may incur provider costs; they are not used by the documented offline demo.
- Production-specific automation, frozen operational identifiers, private continuity evidence, and live-run entry points were intentionally removed.
- Obsidian-oriented rendering and guarded note-write components are present, but complete operational vault integration remains outside this portfolio and under review.
- No LibreOffice integration is implemented or claimed in this sanitized repository.
- Complete V1 acceptance and operational verification remain unfinished.
- Rankings and heuristics support research triage. They are not forecasts, recommendations, or guarantees of factual accuracy.

## Possible next steps — not implemented here

- Add continuous integration for the public-safe suite.
- Package the CLI for simpler installation and versioned releases.
- Build a read-only local interface with sanitized demonstration outputs.
- Expand synthetic end-to-end fixtures and public architecture examples.
- Add live-source integration tests that run only with explicit credentials and spending limits.

## My role and AI-assisted engineering disclosure

I directed the system's requirements, architecture, safety constraints, review criteria, and acceptance decisions. I used AI coding tools to assist with implementation, test expansion, refactoring, and documentation. I reviewed the resulting behavior against the specified contracts and safety boundaries. I often review code manually, going through line by line finding hard to spot bugs. I do not claim to have manually authored every line.

## Development methodology

- **Evidence-driven research:** claims and calculations retain structured source context, quality status, and explicit missing-data handling.
- **Human-in-the-loop decisions:** automated components support research triage and analysis; they do not authorize autonomous investment action.
- **Provenance and reproducibility:** point-in-time inputs, deterministic fixtures, source identifiers, and versioned contracts make results inspectable and replayable.
- **Contract-first engineering:** schemas, bounded interfaces, and fail-closed write gates define behavior before data can cross trust boundaries.
- **Regression testing and independent review:** changes are evaluated against deterministic tests and separate review criteria; public test success is not treated as proof that the private V1 system is complete.

## Safety model

The default path is offline and read-only. A research-note write requires all of these conditions:

1. The caller selects `--write` rather than `--dry-run`.
2. Live AI analysis is explicitly enabled.
3. `MARKET_COLLECTOR_ENABLE_WRITES=1` is set locally.
4. Candidate, analysis, and rendered-note validation all pass.
5. The destination stays inside the allowed root.
6. The destination does not already exist.

The final writer re-checks the environment gate and uses an atomic create-if-absent operation. Secrets belong only in a local `.env`, which is ignored by Git.

See [docs/SAFETY.md](docs/SAFETY.md) for the trust boundaries and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data flow.

## Reproducible offline demonstration

Requirements: Python 3.11 or newer. The exact checked environment used Python 3.14.7 and the versions in `requirements-lock.txt`.

```sh
git clone https://github.com/YOUR-ACCOUNT/akanami-financial-research.git
cd akanami-financial-research
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
cp .env.example .env
```

No API key is needed. The example `.env` keeps every write gate disabled.

```sh
.venv/bin/python collector.py \
  --ticker NVDA \
  --company "NVIDIA Corporation" \
  --dry-run \
  --fixture offline-demo \
  --offline-analysis \
  --max-candidates 4 \
  --max-deep-calls 3
```

Expected safety properties: successful dry run, four synthetic candidates discovered, zero model requests, and zero files written.

Run the deterministic analytics demo:

```sh
.venv/bin/python collector.py demo
```

Its output is explicitly labeled fictional and reports zero files written.

## Testing

Run the public-safe suite with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The sanitized portfolio currently discovers and passes **525 tests across 30 test modules**.

The original private project has a larger suite: 31 of its 61 test modules are not present here because they depend on excluded real-run evaluation captures, frozen production identifiers, private operational configuration, internal continuity evidence, or transitive test fixtures tied to those materials. The 525-test result applies only to this sanitized repository. It does **not** claim that every original test was run or that the complete private suite passes.

## Dependencies and licensing

Direct runtime dependencies are the OpenAI Python library (Apache-2.0), Pydantic (MIT), and PyYAML (MIT). The lock file also resolves permissively licensed transitive packages under MIT, BSD-3-Clause, Apache-2.0, MPL-2.0, PSF-2.0, or dual-license terms. Dependencies are installed from package indexes and are not vendored into this repository.

No project license has been selected. Without a license, the code remains publicly viewable but is not generally granted for reuse, modification, or redistribution beyond rights supplied by the hosting platform. If broad reuse is desired later, MIT is the simplest permissive option; Apache-2.0 is an alternative when an explicit patent grant and NOTICE obligations are preferred. A license should be added only after an intentional owner decision.

## Repository map

| Path | Purpose |
|---|---|
| `collector.py` | Command-line orchestration and bounded run summary |
| `schemas.py` | Strict candidate and analysis contracts |
| `filters.py` | Cheap, inspectable pre-analysis rules |
| `dedup.py` | Canonical IDs and conservative event classification |
| `analysis.py` | Structured model boundary and offline analyzer |
| `renderer.py` | Deterministic Markdown and filename generation |
| `obsidian.py` | Atomic, no-overwrite filesystem boundary |
| `security.py` | Secret redaction and safe error records |
| `market_analytics/` | Point-in-time analytics, quality checks, DCF, ranking, and providers |
| `sources/` | Source adapters with bounded requests and parser isolation |
| `fixtures/` | Fictional or synthetic offline inputs |
| `tests/` | Deterministic unit and integration tests |

## Data and investment disclaimer

Fixture values are fictional and are not investment research. Network-backed sources require local configuration and may incur provider or model costs. Nothing in this repository is investment, legal, tax, or financial advice.
