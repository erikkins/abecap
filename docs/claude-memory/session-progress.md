---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 26–27 2026 (frozen-universe + heal + calendar ALL shipped; ready for Aug 27 PM scan)

## ▶▶ GO SLOW — Erik sensitive after ASST bug. Verify (don't assume). PARQUET not pickle. Never delete (additive; rollback). Empty {} payloads are FALSY → fall through to mangum → trips worker-errors alarm; pass truthy (e.g. {"x":{"run":true}}).

## ✅ SHIPPED + LIVE (Aug 26)
- **Frozen-universe fix:** `universe_refresh_v2` ranks CLEAN stock_universe_service list (screener + EXCLUDED_PATTERNS, ETF-free) off FRESH fetch_raw_bars, not frozen all_data.parquet. LIVE weekly cron `rigacap-prod-universe-refresh` REPOINTED to v2 (SAT 20:00 UTC).
- **Heal MERGER-AWARE:** ripped out yfinance splice (Erik: yfinance is split-adjusted, can't enter raw store; AZN 2026-02-02 = stock_MERGER = new identity). `classify_short_symbols`: corporate_action_boundary / rename_continuity (SUNB) / short_history. All gated, nothing synthetic.
- **Calendar completeness:** `calendar_audit` found CRWD 4:1(slipped display gate)/WETO/REAX missing+spanned → rebuilt → dangerous_count=0. Full baseline rebuild (4653 syms). NEW LIVE weekly cron `rigacap-prod-calendar-rebuild` SAT 18:00 UTC scope=full (root cause: no calendar cron existed).

## ✅ VERIFIED READY FOR AUG-27 PM SCAN (this morning Aug 27)
- Latest snapshot = `signals/universe-history/2026-08-26.json`, source=universe_refresh_v2_fresh_fetch, ranked 2978, top NVDA/INTC/NU/SOFI/PATH/T/SMCI/AAPL/AMZN/MU. INTACT (nothing stale overwrote). PITFWU fresh thru 2026-08-26. → this PM's scan uses fresh top-600 + fetches Aug-27 EOD. NOTE: rebuilt MANUALLY yesterday, NOT by cron (cron 1st fires SAT Aug 29); scan reads latest snapshot, doesn't re-rank daily.

## 🎯 GRADE TODAY/NEXT — Maximizer breakout prediction (Aug 26 close)
- [[project_maximizer_breakout_prediction_aug26]]: regime rotating_bull, **WT firing** (expect next-cycle entry), **BHVN on cusp** (0.4% to trigger, vol×5.34); radar 8 (GEN/DBRG/TECK/BOX/SYF/HPQ/SHEL). Grade via `{"maximizer_preview":{"run":true}}` + build_todays_actions after the Aug-27 scan.

## ⏭️ FOLLOW-UPS (none urgent)
- The Aug-27 PM scan = FIRST real run on fixed universe+calendar → likely BIG rebalance (206 newly eligible); scan_preview showed clean. Watch it.
- rename_continuity carry-forward unbuilt (SUNB). Retire legacy get_universe() (~80 names, only a fallback). Fold nasdaqtraded.txt ETF rule. Cleanup read-only diags (read_perf_test/fetch_scope_test/history_source_probe; KEEP scan_preview/calendar_audit/maximizer_preview). X reply fix (paused).
- False-alarm Aug 26: worker-errors alarm tripped by my {} maximizer_preview call → benign, cleared. Not a real issue.
