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
- TERRAFORM: imported 3 CLI rules (reply-scan day/eve + autopost) into state + main.tf; nothing deleted. **Plan surfaced worker Lambda drift → apply unsafe (see frozen spec).**
- ANTI-REPEAT extended: generate_post (trade_result/we_called_it/loss_review) now takes avoid_texts 8-day lookback like replies + research_insight; generate_social_posts feeds it.

## ▶ NEXT / IN FLIGHT
- Confirm 715 posts ~7:27PM ET + first real M/W/F autopost fires clean.
- RECONCILE worker+api Lambda env+image in main.tf → makes `terraform apply` safe again (pull live via get-function-configuration). Until then: import/plan OK, never apply.
- Clean up 20 stale research_insight drafts (optional).
- GROWTH swings: testimonials/social-proof section; churn prevention (cancel survey + win-back); ads recheck search terms ~Aug 8.
- Reply auto-post paid X tier = OPEN (verify before paying).

## Notes
- Billing portal 400 = NOT a bug (erikkins@gmail.com orphan; don't touch billing w/o ask).
