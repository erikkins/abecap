---
name: AL2023 Lambda canary staged but NOT deployed
description: Python 3.12 / AL2023 container image pushed to ECR as canary tag; Dockerfile change is local-only (uncommitted). Nothing in prod changed yet.
type: project
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
## State as of Apr 15 2026

- **Image in ECR:** `149218244179.dkr.ecr.us-east-1.amazonaws.com/rigacap-prod-api:al2023-canary` (SHA `sha256:973df04c3ad109...`)
- **Dockerfile change is LOCAL ONLY** — `backend/Dockerfile.lambda` base bumped from `python:3.11` to `python:3.12` but NOT committed. Staying uncommitted keeps CI builds on AL2.
- **Rollback tag captured:** `84bc8ef353e5e53882acd126450e5a669b939202` (previous working Worker image)
- **Prod Lambdas still on AL2 / Python 3.11.** No traffic on canary.

**Why:** Need httpfs for DuckDB-over-S3 (currently /tmp-downloads parquet), 10-25% faster Python, modern glibc. But daily scan was fragile → deferred migration to a calm morning.

**How to apply:**
- Don't commit the Dockerfile change until we're ready to cut over
- Follow `docs/runbook-al2023-migration.md` (Worker-first canary, smoke tests, monitor)
- Bundle cutover with: (1) MMC get_all_assets fix, (2) /tmp workaround removal in `data_export.query_parquet`
- Rollback = re-point Worker at old SHA tag (30-60s, no data loss)
