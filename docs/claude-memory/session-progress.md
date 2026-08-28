---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 28 2026 (voice-guard shipped+verified live; sector-rotation research proposed)

## ▶▶ GO SLOW / BE PRECISE, no broken code. NO "DWAP"/"t30v"/"tape"/"Consider adding" customer-facing. NO model filter/param source change w/o full universe re-run. Research MUST map to prod penny-to-penny (only exposure-scaling reproduces). SPA reload after deploy. BUY=INSTRUCTION.

## ✅ SHIPPED to main (all CI/CD success): 0721a74 post-email cleanup+t30v-leak-fix; 06c7e1d sector-strength strip; **875d370** content-engine voice guard (ai_content_service._call_claude now verify→retry→FAIL-CLOSED via voice_filters; was the ONE generator w/ prompt-only ban that let "tape" through) + full original post in engagement email (was [:200]…).

## ✅ VERIFIED LIVE via worker invokes (deploy done 16:13): killed stale tape draft id 821; test_ai_content + generate_research_insight both produce CLEAN no-"tape" copy; emailed Erik the 2 clean drafts (engagement_opportunities {insight_prob:0} → insights:2, opps:2). NOTE: first engagement re-run showed insights:0 = transient (generate_dynamic_insight swallows None at ai_content_service.py:503, ran concurrent w/ intraday_monitor) — NOT the guard (which logs; was silent). Worker invoke pattern: --function-name rigacap-prod-worker --profile rigacap, sync w/ --cli-read-timeout 600 + Bash timeout 600000 works fine (memory's "2-min SIGTERM" fear didn't bite).

## ⏳ AWAITING ERIK GREENLIGHT — Sector-Rotation research (his curiosity Q). Proposed staged plan:
- Phase 1 (cheap, READ-ONLY worker job + chart artifact): "Sector Rotation Observatory" — per-sector rolling RS over 21y history, leaders/laggards, rotation cadence/persistence/secular drift, OVERLAID on regime timeline (what Rotating Bull looks like under the covers). Backtester ALREADY has symbol_sectors + _compute_sector_medians + _get_sector_rs + levers (sector_rs_score/filter, sector_rs_regime_gated w/ rotating_bull in allowed set, max_sector_entries) — NO new plumbing.
- Phase 2 (only if pattern found): WF sweep w/ those levers, judge Sharpe/MaxDD, multi-start. CAVEATS surfaced to Erik: sector CAPS already hurt t30v every metric (Jul20); momentum already rides rotation implicitly → edge (if any) is DD/crowding mgmt not raw return.

## ⏭️ Other open: regime half of _mas (ONE REGIME SOT — safe reporting-layer unification vs scanner.py:675 needs re-run). Queued: DST EventBridge Scheduler (pre-Nov); scrub DWAP in perf_numbers.js; retire get_universe(); X-reply ticker fix; perf-numbers SSOT audit (plan file). Local venv=Py3.9 str|None import fails = RED HERRING (prod Py3.12).
