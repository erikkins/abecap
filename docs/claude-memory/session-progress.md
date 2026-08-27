---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 27 2026 EOD (tonight's scan+digest SENT clean; big cleanup queue open)

## ▶▶ GO SLOW. NO "DWAP"/"t30v"/"Consider adding" customer-facing. NO live filters/guards w/o full WF re-run. SPA reload after deploy. Embed debug in RESPONSE when logs won't surface. BUY=INSTRUCTION (mirror the book).

## ✅ TONIGHT — ALL CLEAN + SENT
- 4:30 scan clean (20 buys incl fresh CRCL/NOW/NVDA/FIG, rotating_bull, CRWD fixed via calendar rebuild, calendar_audit 0 dangerous). 6pm daily digest FIRED + sent OK. Maximizer bought DT/OKTA/RBRK (valid breakouts, real data; OKTA real earnings gap).
- Expo OTA published to `preview` channel → Erik's phone (portfolio-check count etc; reopen app ×2).

## ✅ DOUBLE-SIGNAL ALERT — KILLED (self-managed artifact; book is full + mirror model). Rule `rigacap-prod-double-signals` DISABLED (was cron(0 21)=5pm). It had been silently failing all session on a select NameError → I fixed it (266d9aa) → resumed → fired "Consider adding CRCL" to Jacob (trial=is_valid, double_signals pref default-on). Real signal, not data error; Erik: no correction to Jacob.
## ✅ DIGEST is already mirror-framed (email_service.py:641-647): served→"Our Book"+"Other Signals·Not in our book"; only a no-book FALLBACK else said "Consider adding" → NEUTRALIZED to "Today's Signals" (kept the branch — book can be None on build failure; deleting = NameError risk) (commit 5992614).
## ✅ SMCI=0 = `os` NameError in previous-holds (fixed 3da79ab). Widget tier cols consistent (count·best%). t30v leak fixed.

## 🧹 POST-EMAIL CLEANUP QUEUE (Erik OK'd, do next session):
1. Strip debug scaffolding: `/api/debug/mxholds` route, `_mx_debug` field + `[prevholds-mx]` print in previous-holds, `probe_maximizer_holds` handler; older read_perf_test/fetch_scope_test/history_source_probe.
2. Double-signal FULL removal: terraform (disable/remove rigacap-prod-double-signals rule) + remove check_double_signal_alerts handler + its "Consider adding" template (email_service.py ~3088).
3. ONE REGIME SOT: retire MarketAnalysisService `_mas` 5-regime (DASH-DIAG strong_bull divergence); point GET /regime (main.py:11827)+maximizer_preview at computed-once value (predict_transitions→regime_forecast_snapshots).
4. DST-aware EventBridge Scheduler migration (aws_scheduler_schedule + America/New_York), before Nov EST.
5. Smaller: in-chart M badge; scrub DWAP in perf_numbers.js comments; retire get_universe(); X-reply ticker fix (reply_scanner_service.py:366/437 `max by pnl_pct`).

## 🎯 GRADE next: [[project_maximizer_breakout_prediction_aug26]]; watch if OKTA reverts (Erik hunch).
