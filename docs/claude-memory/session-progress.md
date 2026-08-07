---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 7 2026 (EOD)

## Frozen spec (load-bearing)
- "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/sleeve/DWAP; steer Maximizer. Deploy=push main→"Deploy RigaCap" GHA. NEVER push+dispatch same commit. NEVER `terraform apply` w/o reviewing plan (ignore_changes protects Lambdas now). **NEW RULE: smoke-test a live endpoint after EVERY deploy — GHA-green ≠ app-imports (a fastapi.Form import broke prod ~10min today).** fastapi.Form/File NOT usable (no python-multipart in image) — parse bodies manually.
- Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. `{"db_read":"SQL"}`, `{"run_migration":true,"sql":...}`. OVERLAY SSOT=perf_numbers. BRAND claret/paper, Fraunces+IBM Plex (bundled TTFs in backend/app/assets/fonts/).

## ✅ SHIPPED TODAY (huge session)
- Reply scanner: plain voice, forks, anti-repeat, X-403 deep-link workaround, repeat-click guard.
- Cards: launch 1-5 + og + X banner growth-forward, on-brand, 2x-crisp.
- OWN-SOCIAL AUTOPOST → X+Threads+IG M/W/F, alternating tiers, autopost+kill+EDIT email. IG posts an editorial quote card (chart_card_generator.generate_text_card — Fraunces/Plex, sentence-broken, widow+function-word line rules, centered box). Sentiment guardrail (no "recovered in 2yrs"). Anti-repeat all post types. LIVE-PROVEN: IG+Threads posted end-to-end.
- Launch cards 1/2/3 scheduled X+Threads+IG Sun/Tue/Thu (off the M/W/F days).
- DR: env backup scripts + weekly worker handler + terraform blueprint (15 keys). Terraform apply now SAFE (ignore_changes). Bios canonical (design/brand/profiles/profile-bios.md). Edit-email feature (hotfixed the Form outage).

## ✅ TIER-PREVIEW FIXES SHIPPED (end of day, all deployed + smoke-tested 200)
- MISSED-OPPS (a08a197): frontend only set missedOpportunities `if length>0` → never cleared → admin switching tiers saw stale Preserver "$16,855" on Maximizer. Fixed: 3 sites `|| []`.
- THIS-WEEK + post-trade dashboard (ef3c3a8): frontend didn't forward ?preview_tier= → admin preview showed real tier's book. Fixed: forward preview_tier (backend was already tier-scoped + admin-gated).
- **BOTH were ADMIN-PREVIEW-ONLY. Real subscribers were NOT affected** (their tier resolves from subscription, not the URL param; single-tier users never had a cross-tier stale cache). Confirmed to Erik.

## ▶ MONDAY / NEXT
- DISCUSS: is "A holding week" the right label for Maximizer? It holds each name ~29 days, so a weekly "this week" frame may not fit the hold model. (Erik's ask.)
- Tonight: good-copy insight auto-posts Threads(733)/IG(734) after cooldown. Launch cards fire Sun/Tue/Thu.
- Optional: make missed-opps genuinely per-tier (Preserver trailing-stop catches vs Maximizer breakout), not the generic shared backtest.
- GROWTH: testimonials/social-proof; churn prevention; ads recheck ~Aug 8. Reply paid-X-tier OPEN.

## KEY LESSON TODAY: fastapi.Form broke prod import (~10min outage) → NEW RULE: smoke-test a live endpoint after EVERY deploy; frontend changes build-verified locally first. Both now habit.
