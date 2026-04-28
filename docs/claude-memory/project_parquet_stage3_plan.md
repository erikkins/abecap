---
name: Parquet Stage 3 cutover plan
description: Detailed sequencing for migrating data reads from pickle to parquet, with parallel-read diff harness, staged cutover via feature flag, and 30-day rollback retention.
type: project
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
## Why this plan exists

Stage 1 (shadow write) has been running since Apr 14-15 — both pickle and parquet are written every daily scan. Stage 2 (AL2023/Python 3.12) deployed sometime before Apr 28. The remaining work (Stage 3 consumer migration, Stage 4 decommission) needs careful sequencing because there's no formal regression suite — we use the daily walk-forward as continuous integration, plus a dedicated divergence-events log during the parallel-read window.

## Sequencing (committed Apr 28 2026)

### Stage 3a — Parallel-read + diff harness ✅ SHIPPED Apr 28 2026

Code complete and deployed. `PARQUET_PARALLEL_READ=true` set on Worker via Terraform. First observation data accumulates with tonight's 4:30 PM ET daily scan.

**3a-1 ✅** `data_export_service.compare_pickle_to_parquet()` — async batched compare. Catches missing-on-either-side, row count diff, column set diff, index range diff, value diff (1e-6 relative tolerance for floats, exact for volume), compare errors. Indicator columns intentionally skipped from value compare. Sample of up to 5 mismatched cells per (symbol, column). Single batched DB commit. Returns summary `{compared, diverged, diverged_symbols, by_type}`. Commit `631171e`.

**3a-2 ✅** Wired into daily scan after both stores are written. `PARQUET_PARALLEL_READ` env var gate, try/except wrap so a divergence-log failure can never break the scan. Commit `e7cb5ef`.

**3a-3 ✅** `GET /api/admin/parquet-divergence?days=7&recent_limit=20` returns `by_type`, `top_symbols`, `by_day`, `recent_events`, `stage_status` (with auto-computed `ready_for_stage_3b` flag). Commit `ef870f8`.

**3a-bonus ✅** Daily pipeline health email now has a "Storage Migration → Parquet Divergence" row. GREEN/YELLOW/RED based on structural-vs-explainable types in last 24h. Commit `7dda6d8`.

**3a-4 (passive) — Two-week observation window.** Acceptance for moving to 3b: zero structural divergences (missing_in_*, row_count_diff, value_diff, compare_error) for 7 consecutive days, OR all observed divergences are explainable. Currently monitored automatically via the daily health email.

### Stage 3b — Read-from-parquet feature flag (~6-8h)

Add `READ_FROM_PARQUET=false` env var (default off). When true, scanner_service backs `data_cache` via per-symbol parquet lazy loads. **Set on Worker first** (smaller blast radius — only ~6 cron paths use it). Run for 1 week. Watch nightly WF + daily scan for regressions.

Rollback: flip env var, redeploy. ~5 min.

### Stage 3c — API + admin cutover (~2-3h)

After Worker stable for 1+ week, flip API + admin Lambdas. Side benefit: API Lambda can drop scanner_service module-level import (Stage 4 prep) since lazy parquet loads remove the "must be a worker to touch price data" constraint.

### Stage 4 — Decommission pickle (~3-4h)

Only after 30+ days of stable parquet reads with zero divergence events. Stop writing pickle, delete `s3://.../prices/all_data.pkl.gz` (keep one /backups copy forever), remove pickle import paths, remove guardrail/size-check code. Optional: remove dashboard.json cache if API can read parquet via DuckDB.

## Safety nets (three independent)

1. **3a-2 divergence log** — production traffic does the testing, mechanical
2. **Daily walk-forward result delta** — existing canary; subtle data corruption surfaces in equity-curve diff vs pickle baseline within one day
3. **Pickle retention** — kept writing through 3b/3c, kept readable for 30+ days post-cutover. Rollback is one env var.

## What's NOT in scope

- Removing the `dashboard.json` S3 cache — that's separate, only relevant if API Lambda gets fast enough at parquet reads to skip the cache layer
- Migrating WalkForwardSimulation/PeriodResult tables to parquet — those stay in Postgres
- TimescaleDB/QuestDB — explicitly skipped per the original storage-migration roadmap

## Decision points / off-ramps

- **After 3a-4:** if divergence rate is high or unexplainable, pause. Fix parquet writer first.
- **After 3b Worker:** if regression spike, revert (5 min) and don't proceed to 3c.
- **Before Stage 4:** require 30+ days zero-divergence + zero pipeline regressions. If anything's been flaky, extend retention.

## Estimated total

~25-30 hours of focused work, spread over ~6 weeks (most of it is observation windows, not active coding).
