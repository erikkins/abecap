---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 26 2026 (FROZEN-UNIVERSE FIX shipped + heal path being reworked)

## ▶▶ GO SLOW — Erik sensitive after ASST data bug hit his 1st customer. Verify data before shipping. PARQUET not pickle. Never delete (additive only, keep rollback). Erik catches design flaws — LISTEN.

## ✅ DONE — frozen-universe root cause fixed + live
- ROOT: `universe_refresh` ranked 60d vol off FROZEN all_data.parquet (Jun 15) → wrote `universe-history/{date}.json` → `_scoped_parquet_load` (data_export.py:177) loads only that top-600 → membership frozen, surgers locked out. Was 64d stale, 206/600 (34%) wrong.
- FIX: `universe_refresh_v2` ranks CLEAN stock_universe_service list (NASDAQ/NYSE screener + EXCLUDED_PATTERNS = ETF-free) off FRESH fetch_raw_bars (~1.3min). `heal_newcomers` full-backfills surgers' PITFWU. Commits 58c5d41/69d8bec.
- RAN write+heal: wrote `signals/universe-history/2026-08-26.json`, healed all 206 (0 no_fetch). SCAN PREVIEW (new read-only `scan_preview` handler) = 20 buys / 5 watch, **0 suspicious**, all fresh+healthy px/dwap. Tomorrow's scan safe + WAY different (CAT/CAVA/CRWD/ISRG/LLY/MA/AXP etc now eligible).

## 🚫 BLOCKED — Erik must repoint LIVE weekly cron
- `rigacap-prod-universe-refresh` (SAT 20:00 UTC) still fires OLD `{universe_refresh:true}` (frozen). Terraform SSOT repointed to v2 (committed) but auto-mode BLOCKED live `aws events put-targets`. MUST repoint before Saturday or regresses. Cmd: `aws events put-targets --region us-east-1 --profile rigacap --rule rigacap-prod-universe-refresh --targets file:///tmp/uni_target.json`

## 🔁 HEAL PATH — being REWORKED (Erik's two catches were RIGHT, awaiting his go)
- Built yfinance-splice fallback (append_pitfwu_bars full=True + yf_backfill w/ 5% drift gate). **Erik correctly objected: yfinance is split-adjusted → can't go into raw PITFWU (double-adjust landmine).**
- PROBE FINDINGS (read-only `history_source_probe`): (1) yfinance auto_adjust=False is STILL split-adjusted (first_close 113.28 vs adj 99.24 = dividends only) → no true raw available. (2) **AZN 2026-02-02 boundary = a STOCK MERGER (Alpaca corp-actions: stock_mergers:1, NO split)** → AZN is a NEW asset identity post-merger (Erik's "different ticker?" right in substance). Alpaca serves only post-merger (143 bars); yfinance stitches across the merger with a fake 0.5 "split".
- DECISION (recommended, Erik to confirm): **RIP OUT yfinance-splice-into-PITFWU.** Make heal MERGER-AWARE: still-short symbol → check Alpaca corp-actions → stock_merger/symbol_change/spin_off = REFUSE stitch, leave GATED + flag reason; genuinely young = gated; only backfill confirmed pure-split truncation. AZN correctly stays gated ~few months until 250 bars. Its PITFWU = clean post-merger Alpaca-raw (143), yfinance splice NEVER ran w/ execute → nothing to undo.
- Among 6 short: AZN=merger, CBRS/EQPT/FRVO/SOLS/SUNB=young (72 bars in BOTH sources). yfinance helps NONE.

## ⏭️ NEXT
- Await Erik "yes" → rework heal to merger-aware (remove yf_backfill/fetch_yf_history + full=True yfinance path; add corp-action classifier that labels short reason, leaves gated).
- SEPARATE gap flagged: corp-actions calendar MISSING AZN event (calendar_splits empty) — broader calendar-completeness concern for any pure-split symbol that DOES span its split in Alpaca (would be unadjusted = ASST). Worth a calendar audit later.
- Cleanup later: read-only diag handlers (read_perf_test, fetch_scope_test, history_source_probe, scan_preview keepable). fold nasdaqtraded.txt ETF rule (hardening). X reply engine fix (paused).
- All ADDITIVE: all_data.parquet, old snapshots, existing PITFWU untouched → rollback available.
