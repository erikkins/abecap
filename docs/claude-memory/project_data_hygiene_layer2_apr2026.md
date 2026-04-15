---
name: Data hygiene Layer 2 — nightly corp-actions + ticker-reuse detection
description: Production pipeline for detecting splits, spinoffs, ticker reuses, and delistings before next-day signals fire
type: project
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
## Context

Apr 14 2026 deep data-integrity audit revealed 1326 "dirty" symbols in a 4527 universe (~29%), of which ~55 are in our tradeable $15+ universe. The Apr 14 fixes covered:
- 79 unadjusted splits force-refetched with `Adjustment.SPLIT`
- `market_data_provider.py` switched permanently to SPLIT adjustment (no more unadjusted appends)
- Path B abrupt-jump filter + long-gap (>10d) filter in `_is_data_quality_ok`
- Parquet + DuckDB diagnostic stack (additive, on-demand SQL queries)

**What's NOT handled yet:** forward-looking corporate actions. If NASDAQ recycles a symbol or a major split happens, we need to detect and respond BEFORE the next day's scan uses bad data.

## Gap: ticker reuse is the hardest case

Pure price-monitoring catches:
- Abrupt jumps (Path B: >65% single-day move)
- Date gaps (>10 days)
- DWAP-ratio extremes

But a ticker reuse where the new entity's price range overlaps the old entity's can slip through. Example: NYSE delists "PQR" at $22; 3 months later lists new PQR at $25. If Alpaca fills the gap with stale data or our fetch normalizes it, we get Frankenstein data silently.

## Required: asset-ID tracking

Alpaca's Asset API returns a stable UUID (`id` field) per entity — this survives ticker changes. Using asset_id as the primary key, not symbol, eliminates the reuse problem.

**What to store:**
Per symbol (in a new `symbol_metadata` table or S3 JSON):
- `symbol` (human label)
- `asset_id` (Alpaca UUID — the real identity)
- `first_listing_date` (from Alpaca)
- `status` (active / inactive / quarantined)
- `last_verified_at`

**What to check nightly:**
For every symbol in the universe:
1. Fetch current Alpaca asset record
2. If `current_asset_id` != `stored_asset_id` → ticker reuse detected → quarantine
3. If `current_first_listing_date` differs from stored by >7 days → suspect → quarantine
4. If asset no longer exists in Alpaca → delisted → mark inactive

## Nightly workflow (target 6 PM ET, after daily scan)

```
Step 1: Poll Alpaca corp-actions API
  - Query splits, dividends, spinoffs, mergers for the prior 24h
  - For each detected corporate action, queue action:
    * Split: force refetch affected symbol with Adjustment.SPLIT
    * Spinoff: admin review queue (manual handling)
    * Merger/cashout: remove symbol, close live positions
    * Delisting: remove symbol, mark inactive in model_positions

Step 2: Asset-ID integrity check
  - For every active symbol in universe:
    - Fetch Alpaca asset record
    - Compare stored asset_id vs current
    - Mismatch → quarantine (move to excluded_symbols table with reason='asset_id_changed')
  - Log admin alert if any symbol quarantined

Step 3: Post-actions parquet diagnose
  - Run data_export_service.diagnose_corruption()
  - Compute total_dirty_in_tradeable (intersection with $15+ universe)
  - If > threshold (e.g., 70), admin alert
  - Else, system is healthy for next day's scan

Step 4: Status digest to admin
  - Actions taken in last 24h (splits fixed, symbols quarantined, etc.)
  - Current universe stats (4500 → X clean)
  - Tomorrow's pre-scan ready confirmation
```

## Implementation pointers (for the session that builds this)

- New table: `symbol_metadata` in Postgres (or S3 JSON if we want to keep DB lean)
- New Lambda handler: `nightly_corp_actions_poll` invoked by EventBridge cron
- New service methods on `market_data_provider`:
  - `verify_asset_ids(symbols) -> Dict[sym, {ok: bool, reason: str}]`
  - `poll_corp_actions(since_hours=24) -> List[Event]`
- Integration: `data_quality_gate` service that runs post-scan, checks diagnostics, blocks next day's scan if red.

## Estimated effort
- Asset-ID tracking: 1-2 hours
- Corp-actions API integration: 90 minutes
- Nightly Lambda handler + EventBridge cron: 1 hour
- Admin email digest with action summary: 45 min
- **Total: ~4-5 hours** (dedicated session)

## Dependencies
- Alpaca asset endpoint (already accessible — same API key)
- Alpaca corp-actions endpoint (already tested — see reference_alpaca_corp_actions_sdk.md if that exists)
- Postgres schema migration for symbol_metadata (needs migration-first deploy pattern per CLAUDE.md)

## Long-term architectural ideal

Use `asset_id` (not `symbol`) as the parquet partition key. The human-readable "symbol" becomes a searchable label that can change without corrupting history. This is the Bloomberg/Refinitiv standard. Implement when we do Parquet Session 3 cutover.
