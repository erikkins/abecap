---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 27 2026 EOD (scan+digest SENT clean; cleanup queue open; Data Hygiene reviewed)

## ▶▶ GO SLOW. NO "DWAP"/"t30v"/"Consider adding" customer-facing. NO live filters/guards w/o full WF re-run. SPA reload after deploy. Embed debug in RESPONSE when logs won't surface. BUY=INSTRUCTION (mirror the book).

## ✅ TONIGHT SENT CLEAN: 4:30 scan (20 buys incl CRCL/NOW/NVDA/FIG, rotating_bull, CRWD fixed, calendar_audit 0 dangerous); 6pm digest fired OK; Maximizer bought DT/OKTA/RBRK (valid breakouts, real data). Expo OTA→preview channel (Erik's phone).
## ✅ DOUBLE-SIGNAL ALERT KILLED (rule rigacap-prod-double-signals DISABLED) — self-managed "Consider adding" artifact; had silently failed all session on select NameError (fixed 266d9aa) then fired to Jacob (trial=valid). Digest already mirror-framed; no-book fallback "Consider adding" NEUTRALIZED→"Today's Signals" (5992614, kept branch = book-None safety). SMCI os NameError fixed (3da79ab). Widget cols consistent. t30v leak fixed.

## 👀 DATA HYGIENE tab reviewed (Erik screenshot): 17 "icicles" = PITFWU bars frozen at 2026-06-15 (dropped out of scoped top-600 → append stopped; BY DESIGN). Not a risk: out-of-scope + re-entry now healed by universe_refresh_v2 heal_newcomers + display gate. Split calendar 6655 splits (=my rebuild ✓). Names: ASST(rev-split), AZN(merger, gated), **BRK.B (hyphen/dot Alpaca mapping BUG — real, worth fixing so it heals on re-entry)**.
- ⏭️ AWAITING Erik: (a) fix BRK.B mapping, (b) one-click-heal all 17 icicles, or (c) leave (harmless). My rec: fix BRK.B, leave the rest.

## 🧹 POST-EMAIL CLEANUP QUEUE (Erik OK'd, next session):
1. Strip debug scaffolding: /api/debug/mxholds route, _mx_debug + [prevholds-mx] print in previous-holds, probe_maximizer_holds; read_perf_test/fetch_scope_test/history_source_probe.
2. Double-signal FULL removal: terraform rule + check_double_signal_alerts handler + its "Consider adding" template (email_service.py ~3088).
3. ONE REGIME SOT: retire _mas 5-regime; point GET /regime (main.py:11827)+maximizer_preview at computed-once regime_forecast value.
4. DST-aware EventBridge Scheduler migration (America/New_York), before Nov EST.
5. Smaller: BRK.B mapping; in-chart M badge; scrub DWAP perf_numbers.js comments; retire get_universe(); X-reply ticker fix (reply_scanner_service.py:366/437).

## 🎯 GRADE next: [[project_maximizer_breakout_prediction_aug26]]; watch if OKTA reverts.
