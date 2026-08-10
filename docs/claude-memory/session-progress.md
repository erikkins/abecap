---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Sun/Mon Aug 10 2026

## Frozen spec (load-bearing)
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP. Deploy=push main→"Deploy RigaCap" GHA. **SMOKE-TEST a live endpoint after EVERY deploy.** fastapi.Form/File unusable (no python-multipart; parse bodies manually). NEVER `terraform apply` w/o plan review. Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. `{"db_read":"SQL"}`, `{"run_migration":true,"sql":...}`.
- MODELS latest = Claude 5 family + Opus 4.8. newsletter=opus-4-8, ai_content=sonnet-5, reply_scanner=sonnet-4-6.
- Post types: LAUNCH CARDS = image on ALL 3 (X/Threads/IG). INSIGHT autoposts = text on X/Threads, generated card on IG only.

## ✅ THIS SESSION (Aug 10)
- REPLY ENGINE de-mechanized (commit 04626f9): drafts all cloned ONE skeleton (confession → "$TICKER flagged DATE, +X% since" → terse maxim) b/c single KO few-shot + fixed ordering; anti-repeat varied words not SHAPE. Added "VARY THE SHAPE" section (3 on-voice examples w/ different structures incl one with NO ticker/number), anti-template rules (rotate opener/result-placement/closer; often drop the ticker), + structural anti-repeat in avoid_block. Dry-run confirmed: shapes now vary, no flag/date/% template.
- (Sat, still relevant) newsletter LOCKED for Sunday; DB deadlock-retry middleware live; admin avatar→claret; "Post Goes Live" email fixed twice (card image full-URL render + platform mislabel Threads-as-Instagram); real-wins feed wired.

## ▶ IN FLIGHT / DECISION (asked Erik)
- Reply engine residual: a single scan batch can CLUSTER (today's dry-run = all 3 Maximizer/"exit is the edge", 2× the same stock $TSM diff accounts). Cause: within one scan, replies only anti-repeat vs PRIOR DAYS, not vs EACH OTHER (recent_replies loaded once at scan start). OFFERED intra-batch de-dup (no 2 replies same symbol / not 3 same angle per scan). Erik deciding: add it, or leave clustering to sort across the day's scans. Shape fix itself is done + good.

## ▶ NEXT (queued)
- DOCS REFRESH ([[project_docs_refresh]]) — the main planned task, not started. Reuse layout, write fresh, 2-tier/growth-forward/claret/walk-forward/SSOT.
- "A holding week" label for Maximizer (discuss). GROWTH: testimonials/social-proof; churn; ads recheck.
