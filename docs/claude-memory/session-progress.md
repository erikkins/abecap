---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 28 2026 (voice-guard shipped+verified; sector-rotation research scoped, incl. PREDICTABILITY lens)

## ▶▶ GO SLOW / BE PRECISE, no broken code. NO "DWAP"/"t30v"/"tape"/"Consider adding" customer-facing. NO model filter/param source change w/o full universe re-run. Research MUST map to prod penny-to-penny (only exposure-scaling reproduces). SPA reload after deploy. BUY=INSTRUCTION.

## ✅ SHIPPED to main (all CI/CD success): 0721a74 post-email cleanup+t30v-leak-fix; 06c7e1d sector-strength strip; **875d370** content-engine voice guard (ai_content_service._call_claude verify→retry→FAIL-CLOSED via voice_filters; was the ONE generator w/ prompt-only ban that let "tape" through) + full original post in engagement email (was [:200]…). VERIFIED LIVE: killed tape draft id 821; regeneration clean; emailed Erik 2 clean drafts (engagement_opportunities {insight_prob:0}→insights:2). insights:0 on first re-run = transient (generate_dynamic_insight swallows None @ai_content_service.py:503, concurrent w/ intraday_monitor), NOT the guard. Worker invoke: --function-name rigacap-prod-worker --profile rigacap sync + --cli-read-timeout 600 + Bash timeout 600000 works.

## ⏳ AWAITING ERIK GREENLIGHT — SECTOR ROTATION research (his curiosity→now excited: "predict next hot sector = $$$"). Told him honestly: pure prediction is hard/coin-flip beyond momentum persistence; real edge = EARLY-ROTATION DETECTION (RS acceleration/2nd-deriv, within-sector breadth thrust, rank migration, regime-conditioned). Analog exists: market_regime.predict_transitions() (market-level transition probs) → sector version = leadership-change probs.
- **Phase 1 (READ-ONLY worker job + chart artifact), now w/ PREDICTABILITY LENS:** (1) does sector leadership persist vs revert & at what horizon; (2) DO leading indicators lead — does RS accel/breadth precede RS level turning (=is it forecastable at all); (3) count independent rotation events in 21y (small-sample/overfit guard). Overlay on regime timeline. Backtester ALREADY has symbol_sectors + _compute_sector_medians + _get_sector_rs + levers (sector_rs_score/filter, sector_rs_regime_gated w/ rotating_bull allowed, max_sector_entries) — NO new plumbing.
- **Phase 2 (only if #2 shows real lead):** build sector-transition-prob model, WF a tilt, judge Sharpe/MaxDD multi-start, prod-day-step validate. CAVEATS given: sector CAPS already hurt t30v every metric (Jul20); momentum already rides rotation → edge (if any) = DD/crowding mgmt not raw return.
- NEXT ACTION: offered to run Phase 1 w/ predictability lens; awaiting Erik "go".

## ⏭️ Other open: regime half of _mas (ONE REGIME SOT — safe reporting unification vs scanner.py:675 needs re-run). Queued: DST EventBridge Scheduler (pre-Nov); scrub DWAP perf_numbers.js; retire get_universe(); X-reply ticker fix; perf-numbers SSOT audit (plan file). Local venv Py3.9 str|None import fails = RED HERRING (prod Py3.12).
