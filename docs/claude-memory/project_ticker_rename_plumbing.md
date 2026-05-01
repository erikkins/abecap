---
name: Production-grade ticker-rename / asset-ID plumbing
description: SQ→XYZ (Block) showed up as duplicate trades in WF results. Layer 2 catches future events but historical pickle has both tickers. Boring plumbing for next week.
type: project
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
**The bug surface (Apr 30 2026):** SQ and XYZ trades appear with identical entry/exit dates and prices in the 11y WF result. Block renamed SQ → XYZ in 2024. Our universe captured both, treating them as separate stocks. Result: same economic trade counted twice in 249 trailing-stop trade list.

**Why:** Universe is keyed on ticker. When a ticker change happens, the old and new symbols both end up in the universe during the transition window. Layer 2 (deployed Apr 15 2026, [project_data_hygiene_layer2_apr2026.md](project_data_hygiene_layer2_apr2026.md)) catches future ticker reuse going forward, but the existing pickle has historical artifacts.

**How to apply (next week, schedule alongside CB-in-production):**

Three layers of defense, do all three:

1. **Asset-ID is primary key, ticker is display attribute** — Alpaca returns a stable `asset_id` per company. Universe should dedupe on asset_id, not ticker. Ticker becomes the current display label that may change over time. ~half-day refactor.

2. **Verify Layer 2 covers ticker-change events explicitly** — corp-actions API includes name changes, ticker changes, mergers. Confirm the existing nightly poll catches these (not just splits/dividends). On detection: alias old → new in symbol-metadata service, merge histories under new symbol, exclude old from go-forward universe.

3. **Backfill cleanup utility** — one-shot job that walks the existing pickle, looks up each symbol's asset_id, identifies cases where two tickers share an asset_id, merges their bars under the current canonical ticker, drops duplicates.

**Quick fix for current data analyses (apply now in post-processing):**

Dedupe trades by `(entry_date, entry_price, exit_date, exit_price)` — identical tuples are the rename-duplicate signature.

**Other rename precedents to test against:**
- FB → META (2022-06-09)
- SQ → XYZ (2024)
- TWTR → X (delisted, but old data exists)
- Anything else discovered via Alpaca asset_id collisions

**Connected work:**
- Layer 2 corp-actions infra: already in place
- CB-in-production wiring: same week schedule
- Eventual parquet migration: opportunity to refactor asset-keying at the same time
