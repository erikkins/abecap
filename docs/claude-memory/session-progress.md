---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 7 2026

## Frozen spec (load-bearing)
- Public copy: "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/sleeve/DWAP; steer Maximizer. Deploy = push main → "Deploy RigaCap" GHA (or `gh workflow run`). NEVER push+dispatch same commit (serialize). NEVER `aws lambda update-function-configuration --environment` casually. **terraform apply SAFE now** (ignore_changes on both Lambdas; [[project_terraform_apply_unsafe]] resolved). Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. `{"db_read":"SQL"}` (ts ::text); `{"run_migration":true,"sql":...}`.
- OVERLAY SSOT = perf_numbers.js/.py. One-click links → api.rigacap.com. BRAND=claret/paper #F5F1E8/#7A2430, Fraunces (display) + IBM Plex Sans (body) + IBM Plex Mono (data/URLs). Bios canonical: design/brand/profiles/profile-bios.md.

## ✅ SHIPPED THIS SESSION (all live)
- REPLY scanner: plain voice, anti-repeat, no-ellipsis, 2 tier forks; X 403 → deep-link to X composer ([[project_x_api_reply_403]]); repeat-click "already opened" guard.
- CARDS: launch 1-5 + og-card growth-forward, 2x-crisp, paper/claret; X banner knob 1500×500.
- OWN-SOCIAL CADENCE → **NOW ALL 3 PLATFORMS (X + Threads + Instagram)**, M/W/F, alternating Preserver/Maximizer, autopost + 1-click kill. IG posts an on-brand EDITORIAL QUOTE CARD (chart_card_generator.generate_text_card, rebuilt: bundled Fraunces+Plex TTFs in backend/app/assets/fonts/, centered wordmark, sentence-broken Plex body, pixel-wrap + widow pull-up + function-word break rule, box-centered text, mono url). publish_post auto-gens the IG card. Handler generates once → 1 post/platform (same scheduled_for); heads-up email names all 3; ONE kill cancels all (cancel-email cascades to same-time research_insight siblings). Insight prompt now enforces one-idea-per-sentence/no-nested-clauses. Anti-repeat on all own-post types. Dry-run verified (commit d2b3ce4 deployed).
- DR: env backup scripts + weekly worker handler {"backup_lambda_env"} + EventBridge; terraform blueprint (15 live-only keys declared, tfvars populated). Bios refreshed. Stale drafts cleaned.

## ✅ LIVE TEST DONE + LAUNCH CARDS SCHEDULED (Aug 7 eve)
- Fan-out LIVE-tested: IG (722) + Threads (721) POSTED — **IG card path PROVEN end-to-end** (render→S3→Graph API). X (720) failed = X API credits depleted → Erik enabled AUTO-RECHARGE (X works now, confirmed by 732). IG "can't share links" restriction → Erik verified-human, lifted.
- COPY BUG caught live: generator produced net-negative "…13.7% drawdown. Recovered in about 2 years" (reads as 2yr underwater). FIXED (commit 5d34c45): sentiment guardrail in generate_research_insight prompt — never frame drawdown/recovery-duration as reassurance; frame depth-vs-market + discipline. Dry-runs now confident (e.g. "13.7%, a fraction of the market's 55%"). Cancelled bad X post 720; Erik deletes old Threads 721 + IG 722 manually.
- TODAY'S post: inserted candidate-2 good copy to X/Threads/IG (732/733/734). X posted now; Threads+IG queued behind PLATFORM_COOLDOWN (threads 180min, ig 120min, twitter 30) from the test post → auto-publish tonight (~IG 8:15pm ET, Threads 9:15pm ET) via 15-min publish cron.
- LAUNCH CARDS scheduled (9 posts): launch-1/2/3 → X+Threads+IG on Sun Aug9 / Tue Aug11 / Thu Aug13 16:00 UTC (noon ET). Staggered off the M/W/F autopost days. post_type='launch_card', image_s3_key=https://rigacap.com/launch-cards/launch-N.png (CDN, verified 200). Captions have NO url (bio /ig link carries it; keeps IG link-filter calm).

## ▶ NEXT / IN FLIGHT
- FAST-FOLLOW (Erik's idea): "Edit this" option in the autopost heads-up email — kill OR tweak-a-word-and-regen the card before it posts. Real little build (edit endpoint + regen). Offered; Erik to decide now-vs-next-session.
- Confirm Threads/IG auto-post tonight after cooldown; first real M/W/F fan-out. Erik pastes new bios (design/brand/profiles/profile-bios.md).
- GROWTH swings: testimonials/social-proof; churn prevention; ads recheck ~Aug 8. Reply paid-X-tier OPEN.

## Notes
- Billing portal 400 = NOT a bug (erikkins@gmail.com orphan; don't touch billing w/o ask).
