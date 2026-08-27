---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 26–27 2026 (data-integrity work DONE; now CHART cleanup inventory in flight)

## ▶▶ GO SLOW — Erik sensitive after ASST bug. VERIFY don't assume. PARQUET not pickle. Additive/rollback. Empty {} payloads FALSY→mangum→trips worker-errors alarm; pass truthy {"x":{"run":true}}.

## ✅ SHIPPED + LIVE (Aug 26) — all data-integrity work
- Frozen-universe fix (`universe_refresh_v2`, fresh fetch of clean ETF-free list); weekly cron `rigacap-prod-universe-refresh`→v2 (SAT 20:00). Merger-aware heal (`classify_short_symbols`; yfinance splice ripped out). Calendar completeness: fixed CRWD/WETO/REAX, full rebuild, NEW weekly cron `rigacap-prod-calendar-rebuild` SAT 18:00 scope=full.
- VERIFIED ready for Aug-27 PM scan: latest snapshot 2026-08-26.json (v2 fresh, ranked 2978) intact + PITFWU fresh thru 8-26. Rebuilt MANUALLY yesterday (cron 1st fires SAT 8-29). Scan reads latest snapshot, doesn't re-rank daily.

## 🎯 GRADE after Aug-27 PM scan — [[project_maximizer_breakout_prediction_aug26]]: WT firing, BHVN cusp. Run maximizer_preview + build_todays_actions.

## 🔧 IN FLIGHT — CHART CLEANUP inventory (Erik's new request)
- Erik wants: (1) remove vestigial markings (+20% profit-target line = OLD strategy; live t30v = 30% trail, no target), (2) explain/remove "unknown vertical lines", (3) NEW FEATURE: overlay PREVIOUS HOLDS (entry+exit markers) per symbol incl from walk-forwards.
- Ties to the pre-existing **t30v DISPLAY-PARITY SWEEP** TODO (chart +20% ref line was the last open item; also grep App.jsx+emails for 12%/+20%/profit_target/1.20).
- Launched 2 read-only Explore agents (results come back to THIS session via notification): (A) frontend chart-markings inventory (App.jsx / Chart* — recharts? all ReferenceLine/vertical/markers + which vestigial/unknown), (B) backend prior-holds DATA availability (TierFill/ModelPosition/MaximizerBookSnapshot/ensemble_signals + WF trade logs + any per-symbol history endpoint). When they return: synthesize inventory → scope changes with Erik BEFORE editing.

## ⏭️ OTHER TODOs (none urgent): retire legacy get_universe() (~80 names, fallback only); fold nasdaqtraded.txt ETF rule; rename_continuity carry-forward (SUNB); remove diag handlers read_perf_test/fetch_scope_test/history_source_probe (KEEP scan_preview/calendar_audit/maximizer_preview); X reply fix (paused); sell-alert 12% vs 30% parity (deferred Jul 7); docs refresh.
