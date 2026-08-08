---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Sat Aug 8 2026

## Frozen spec (load-bearing)
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP. Deploy=push main→"Deploy RigaCap" GHA. **SMOKE-TEST a live endpoint after EVERY deploy** (Fri Form outage). fastapi.Form/File unusable (no python-multipart; parse bodies manually). NEVER `terraform apply` w/o plan review (ignore_changes protects Lambdas). Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. `{"db_read":"SQL"}`, `{"run_migration":true,"sql":...}`.
- MODELS: latest=Claude 5 family + Opus 4.8. Newsletter=claude-opus-4-8; ai_content=claude-sonnet-5; reply_scanner=claude-sonnet-4-6. Default new AI work to latest/most-capable.
- Two post types differ: LAUNCH CARDS post the image on ALL 3 (X/Threads/IG). INSIGHT autoposts = TEXT on X/Threads, generated quote-card only on IG.

## ✅ SHIPPED TODAY (Sat)
- NEWSLETTER "Market, Measured" world-class rework + Erik LOCKED it → sends Sunday. 2-tier growth-forward voice, tiers LEAD STORY (§02), Opus 4.8, real market color (gold) from briefing but COUNTS authoritative from structured data, §03 bold full first sentence, §04 no invented trades. Anti-hallucination: never fabricate news/dates/causation/per-stock-per-day anecdotes/sub-counts; only cite provided headlines + REAL-WINS feed.
- REAL-WINS FEED: generator now pulls this week's actual closed winners (model_positions, no ticker, real %) → §04 cites a TRUE win or stays principle-based (0 wins this week yet).
- DB DEADLOCK-RETRY middleware (main.py DeadlockRetryMiddleware): retries request up to 2x on Postgres DeadlockDetectedError → no more 500/alarm from transient lock contention (root: post-deploy API-Gateway flush into cold Lambdas hitting same write path; exact query in RDS log). Verified 0 deadlocks post-deploy.
- Admin social preview avatar: /icon-halo.svg (navy, retired) → /icon-bitone.png (claret). 
- "Post Goes Live" email chart card: presigned a full CloudFront URL (launch cards) → mangled/broken; now uses full http URLs directly. Fixed.

## ▶ NEXT / OPEN
- MONDAY: docs refresh ([[project_docs_refresh]]); "A holding week" label for Maximizer.
- Fri/earlier: tier-preview fixes shipped (real subs unaffected); launch cards scheduled Sun/Tue/Thu; autopost cadence live.
- Deadlock exact-query root cause = RDS Postgres log (not chased; retry masks it safely). GROWTH: testimonials/social-proof; churn; ads recheck.
