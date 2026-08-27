---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 26 2026 (frozen-universe + heal + calendar ALL shipped; maximizer preview done)

## ▶▶ GO SLOW — Erik sensitive after ASST bug. Verify data. PARQUET not pickle. Never delete (additive; rollback). Erik catches design flaws — LISTEN. Calendar writes = safe (rebuild_calendar backs up + UNION-merges). Empty {} payloads are FALSY → handlers skip → pass truthy (e.g. {"x":{"run":true}}).

## ✅ ALL SHIPPED + LIVE
- **Frozen-universe fix:** `universe_refresh_v2` ranks CLEAN stock_universe_service list (NASDAQ/NYSE screener + EXCLUDED_PATTERNS, ETF-free) off FRESH fetch_raw_bars, not frozen all_data.parquet. Ran write+heal → snapshot 2026-08-26, 206 surgers healed. LIVE weekly cron `rigacap-prod-universe-refresh` REPOINTED to v2 (SAT 20:00 UTC).
- **Heal MERGER-AWARE** (Erik's 2 catches right: yfinance is split-adjusted; AZN 2026-02-02 = stock_MERGER not split). Ripped out yfinance splice. `classify_short_symbols` labels: corporate_action_boundary (merger/spinoff→gated), rename_continuity (SUNB — mature, history under old ticker), short_history (young). All gated, nothing synthetic in raw store.
- **Calendar completeness FIXED.** `calendar_audit` found 3 maladjusted (missing split + bars span ex-date): CRWD 4:1 fwd (SLIPPED display gate), WETO 100:1 rev, REAX 10:1 rev. Ran rebuild → re-audit dangerous_count=0. Ran ONE-TIME FULL baseline rebuild (4653 syms, 644 splits/498 syms, total 6655). Root cause = NO calendar cron → wired LIVE weekly `rigacap-prod-calendar-rebuild` (SAT 18:00 UTC, scope=full). Handler got scope=full/snapshot (get_universe is only ~80 curated names, would miss small-caps).

## 🔭 MAXIMIZER BREAKOUT PREDICTION (tomorrow) — done via read-only `maximizer_preview`
- regime = **rotating_bull** → breakout sleeve ACTIVE. **1 firing: WT** (WisdomTree $24.76, clean 2677 bars, fresh cross today, $56M vol, NOT held). Book 12/15 → room for 3. **BHVN on cusp** (0.4% to trigger, vol×5.34). Radar 8 approaching (BHVN/GEN/DBRG/TECK/BOX/SYF/HPQ/SHEL). WT/BHVN data verified clean (px/dwap 1.52/1.47). Timing nuance: breakout_signal fires on CROSS DAY, so WT (crossed today) is next-cycle entry; new crosses tomorrow = radar pool.

## 📌 get_universe() = LEGACY ~80 names (hardcoded NASDAQ_100+SP500_ADDITIONS in config.py). NOT the real universe — scanner.py:152 seeds it then load_full_universe/ensure_loaded REPLACES with dynamic ~4653. Only a fallback risk (fixed via scope=full). Cleanup: retire/repoint eventually.

## ⏭️ FOLLOW-UPS (none urgent)
- rename_continuity CARRY-FORWARD unbuilt (SUNB-type: map old→new ticker). Rare.
- Cleanup read-only diag handlers later: read_perf_test, fetch_scope_test, history_source_probe. KEEP: scan_preview, calendar_audit, maximizer_preview (useful pre-send checks). fold nasdaqtraded.txt ETF rule. X reply engine fix (paused).
- All ADDITIVE → rollback available. AZN stored = clean post-merger Alpaca-raw (143), gated as new identity.
