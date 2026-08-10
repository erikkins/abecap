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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP. Deploy=push main→"Deploy RigaCap" GHA. **SMOKE-TEST a live endpoint (curl api.rigacap.com/api/market-data-status → 200) after EVERY deploy.** fastapi.Form/File unusable (no python-multipart). NEVER `terraform apply` w/o plan review. NEVER `lambda update-function-configuration --environment`. Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. Reply scan: `{"scan_replies":{"since_hours":24,"dry_run":false,"clear_existing":true}}` → drafts + approval emails to erik@rigacap.com. Payload at /tmp/scan_payload.json.
- MODELS latest = Claude 5 + Opus 4.8. newsletter=opus-4-8, ai_content=sonnet-5, reply_scanner=sonnet-4-6.

## ✅ THIS SESSION (Aug 10) — REPLY ENGINE fully de-mechanized + DONE
- 04626f9: VARY THE SHAPE (broke the single-skeleton template).
- 530044a: intra-batch anti-repeat (each accepted draft → `_recent_replies` so next reply in same scan differs). Repeat tickers ALLOWED per Erik — just each reply a distinct take, NOT a same-symbol block.
- c2b3026: MAXIMIZER thesis-rotation (was 4/4 converging on "exit>entry"; now 6-angle menu: let-winners-run / giveback-math / in-before-crowd / patience-through-shakeout / position-don't-predict / exit-is-edge[sparingly]) + avoid_block fights THESIS-level repeat + global anti-judgy/teachy tone ("confident+contrarian worth-the-click, said as someone who made the mistake, not grading the reader"). All 3 DEPLOYED+smoke-tested.
- Re-ran scan twice → final batch (7 drafts, 7 emails to Erik): maximizer now 2 distinct theses, tone un-preachy, 3× AMZN each a distinct take. Erik reviewing inbox. If good, reply engine is DONE.

## ▶ NEXT (queued, in order)
- DOCS REFRESH ([[project_docs_refresh]]) — Erik's stated next task. Reuse layout, write fresh, 2-tier/growth-forward/claret/walk-forward/SSOT. Signal-intelligence stays confidential.
- Plan unified-sauteeing-whale.md: verify public return numbers (consistency+correctness) → centralize into perf_numbers.* SSOT. Gate A read-only, STOP for sign-off.
- "A holding week" label for Maximizer (~29-day holds) — discuss. GROWTH: testimonials; churn; ads recheck.
