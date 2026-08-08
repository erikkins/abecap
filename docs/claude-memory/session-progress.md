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
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/DWAP. Deploy=push main→"Deploy RigaCap" GHA. NEVER `terraform apply` w/o plan review (ignore_changes protects Lambdas). **SMOKE-TEST a live endpoint after EVERY deploy** (fastapi.Form broke prod Fri ~10min; fastapi.Form/File unusable — no python-multipart; parse bodies manually). Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. `{"db_read":"SQL"}`, `{"run_migration":true,"sql":...}`.
- MODELS: latest = Claude 5 family + Opus 4.8 (claude-opus-4-8), Sonnet 5 (claude-sonnet-5), Haiku 4.5. Newsletter now on claude-opus-4-8; ai_content on claude-sonnet-5; reply_scanner still claude-sonnet-4-6.

## ✅ TODAY: NEWSLETTER "Market, Measured" REWORK (this Sunday's draft = world-class)
- Was: Preserver-only, repetitive, no Maximizer, no tiers story; §04 HALLUCINATED "Senate stopgap cleared Thursday, system re-evaluated within the hour". Root causes: prompt POSITIONING was June-Preserver-only; §04 fed real Google-News RSS but model embellished (fake date + causal claim); §01/§03 lifted unverifiable specifics ("gold +2.3%", "14 signals from 9", "a ride-hailing name") from the AI market briefing; §04 invented trades ("tech stock jumped 20% Wed").
- FIXED (many deploys, latest b632d95): 2-tier growth-forward POSITIONING + growth topics; news guardrail (cite provided headlines only, never fabricate date/causation); market COLOR (gold/sectors) allowed from real-data briefing but COUNTS authoritative only from structured data; counts=CORE book (Maximizer is a SEPARATE book — don't attribute); global "aggregate-data-only, never invent per-stock/per-day anecdote"; §04 no-invented-trades; §03 bold ENTIRE first sentence; Opus 4.8.
- Regenerated w/ tiers LEAD STORY (via {"generate_newsletter":{"lead_story":...,"force":true}}). v6 draft (newsletter/drafts/2026-08-09.json) verified clean: correct counts, real color, both tiers, no fabrication. UNLOCKED — Erik reviews/locks; publishes Sunday.

## ▶ IN FLIGHT / NEXT (asked Erik)
- **REAL-WINS FEED (Erik green-lighting):** feed the generator this week's ACTUAL closed winners (ModelPosition closed <7d, pnl>0, generic no-ticker + real %) so §04/§01 can tell a TRUE win story instead of a banned anecdote. Same pattern as the news feed. Not built yet.
- §04 opens "A reader emailed asking…" = framing device (flagged; Erik may want changed).
- **STILL OPEN (Fri):** "post went live" confirmation emails don't render the chart-card image — needs a chase (email img/presigned-URL handling).
- MONDAY: docs refresh ([[project_docs_refresh]]); "A holding week" label for Maximizer.
- Fri autoposts/launch cards live; tier-preview fixes shipped (real subs unaffected).
