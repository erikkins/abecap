---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Mon Aug 10 2026

## Frozen spec (load-bearing)
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP. Deploy=push main→"Deploy RigaCap" GHA. **SMOKE-TEST a live endpoint after EVERY deploy.** fastapi.Form/File unusable (no python-multipart; parse bodies manually). NEVER `terraform apply` w/o plan review. NEVER `lambda update-function-configuration --environment`. Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. Reply scan event: `{"scan_replies":{"since_hours":N,"dry_run":false,"clear_existing":true}}` → creates drafts + sends approval emails to erik@rigacap.com.
- MODELS latest = Claude 5 family + Opus 4.8. newsletter=opus-4-8, ai_content=sonnet-5, reply_scanner=sonnet-4-6.

## ✅ THIS SESSION (Aug 10)
- REPLY ENGINE de-mechanized (04626f9) then intra-batch anti-repeat (530044a, DEPLOYED+smoke-tested): each accepted draft inserted into `_recent_replies` so the NEXT reply in the same scan avoids its shape/angle/wording. Per Erik: repeat tickers across threads is FINE + human — just make each reply a distinct take, NOT a same-symbol block.
- Ran real scan (24h): 7 drafts, 7 approval emails SENT to Erik.

## ▶ IN FLIGHT / DECISION (asked Erik)
- Residual after re-scan: SHAPE now varies + Preserver replies are distinct, BUT all 4 MAXIMIZER replies (2×TSM, 2×SNDK) hit the SAME thesis+closer ("the entry is the easy part / exit rule matters more"). Thesis-monotony (different from the shape gap we fixed), Maximizer-specific. RECOMMENDED fix: give Maximizer voice a ROTATION of distinct theses (let winners run / giveback math / conviction / patience through mid-hold dip / rebalance cadence) + avoid_block penalizes thesis+closer reuse harder (keep repeat tickers allowed). Asked Erik: apply the thesis-rotation + re-send, or approve today's batch as-is.

## ▶ NEXT (queued)
- DOCS REFRESH ([[project_docs_refresh]]) — main planned task, not started. Reuse layout, write fresh, 2-tier/growth-forward/claret/walk-forward/SSOT.
- Plan file unified-sauteeing-whale.md: VERIFY public return numbers (consistency + correctness) then centralize into perf_numbers.* SSOT — Gate A read-only first, STOP for Erik sign-off. Not started.
- "A holding week" label for Maximizer (~29-day holds) — discuss. GROWTH: testimonials/social-proof; churn; ads recheck.
