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
- Public copy: "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/sleeve/DWAP; steer toward Maximizer. Deploy = push main → "Deploy RigaCap" GHA (or `gh workflow run "Deploy RigaCap" --ref main`). NEVER push+dispatch same commit (ResourceConflictException; serialize). NEVER `aws lambda update-function-configuration --environment`. **NEVER `terraform apply`** (worker/api Lambda drifted → apply reverts env+image = outage; [[project_terraform_apply_unsafe]]). Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. `{"db_read":"SQL"}` (ts ::text); `{"run_migration":true,"sql":...}`.
- OVERLAY SSOT = perf_numbers.js/.py. One-click links → api.rigacap.com. Admin email → erik@rigacap.com. BRAND=claret/paper (Fraunces + IBM Plex); mono only for data, never prose.

## ✅ SHIPPED THIS SESSION (all live)
- REPLY SCANNER: plain voice, 5-reply-day anti-repeat, no-ellipsis, 2 tier forks. X 403 → deep-link to X pre-filled composer ([[project_x_api_reply_403]]); Erik posts via 1 tap.
- CARDS: launch 1-5 growth-forward, fonts fixed (mono→serif/sans), 2x-crisp; og-card rebuilt claret/paper CENTERED+crop-safe (link-preview at rigacap.com/og-card.png, NOT header). Meta tags → walk-forward+SSOT.
- OWN-SOCIAL CADENCE: autopost + 1-click kill, Mon/Wed/Fri, alternating Preserver/Maximizer. Handler {"autopost_own_social":{}} (guard fixed b975447: empty-dict was falsy). EventBridge rigacap-prod-autopost-social. Live TEST post 715 scheduled 23:27 UTC (~7:27PM ET Aug 7) — confirm it posts.
- X BANNER: 1500×500 (3:1 — X zoom-crops other ratios), single-tone bg, divider stops at frame, native-size sharp (blur was X compression, unavoidable). design/brand/profiles/x-banner.png + x-banner-source.html. Erik uploaded, accepted.
- TERRAFORM: imported 3 CLI rules (reply-scan day/eve + autopost) into state + main.tf; nothing deleted.
- ANTI-REPEAT extended: generate_post (trade_result/we_called_it/loss_review) now takes avoid_texts 8-day lookback like replies + research_insight; generate_social_posts feeds it.
- **TERRAFORM NOW SAFE (commit): added `lifecycle{ignore_changes=[image_uri,environment]}` to BOTH Lambdas.** Root cause: live 55/52 env keys vs main.tf 41/39 (TIER_SERVING, MAXPP prices, META_FB_*, shadow flags, ADS_INGEST_SECRET…) + CI image + rotated tokens → blind apply would drop keys/revert = outage. ignore_changes = idiomatic fix (env/image managed out-of-band; no secret transcription; survives token rotation). Full plan now 0 add/0 destroy/1 COSMETIC (monthly_recap target `{"_":1}`→`{}`; recap stays disabled). Did NOT apply. tfvars left stale (safe — env ignored). [[project_terraform_apply_unsafe]] now RESOLVED for the clobber risk.
- STALE DRAFTS CLEANED: deleted 23 (20 stuck research_insight drafts + 3 killed monthly_recap). Kept 8 active contextual_reply (today; = dedup + anti-repeat history) + scheduled 715. Social tab clean.

## ▶ NEXT / IN FLIGHT
- Confirm 715 posts ~7:27PM ET Aug 7 + first real M/W/F autopost fires clean.
- GROWTH swings (Erik's north star 1→5→…→500 subs): testimonials/social-proof section (landing); churn prevention (cancel survey + win-back); ads recheck search terms ~Aug 8.
- Reply auto-post paid X tier = OPEN (verify before paying). og-card/launch cards live; banner = Erik uploads manually.
- OPTIONAL later: refresh stale terraform.tfvars values for accuracy (not needed for safety now).

## Notes
- Billing portal 400 = NOT a bug (erikkins@gmail.com orphan; don't touch billing w/o ask).
