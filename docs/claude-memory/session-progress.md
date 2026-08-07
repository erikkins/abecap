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
- Public copy: "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/sleeve/DWAP; steer Maximizer. Deploy = push main → "Deploy RigaCap" GHA (or `gh workflow run`). NEVER push+dispatch same commit (serialize). NEVER `aws lambda update-function-configuration --environment` casually (restore script is the guarded exception). Worker=rigacap-prod-worker, AWS_PROFILE=rigacap. `{"db_read":"SQL"}` (ts ::text); `{"run_migration":true,"sql":...}`.
- OVERLAY SSOT = perf_numbers.js/.py. One-click links → api.rigacap.com. BRAND=claret/paper (Fraunces + IBM Plex; mono only for data). X banner=1500x500 native.
- TERRAFORM SAFE NOW ([[project_terraform_apply_unsafe]] resolved): both Lambdas have `lifecycle{ignore_changes=[image_uri,environment]}`. Full plan = 0-add/0-destroy/1-cosmetic (monthly_recap target). Still review plans; keep ignore_changes. tfvars gitignored (secrets local only).

## ✅ SHIPPED THIS SESSION (all live)
- REPLY scanner: plain voice, 5-day anti-repeat, no-ellipsis, 2 tier forks; X 403 → deep-link to X pre-filled composer ([[project_x_api_reply_403]]). Deep-link double-post FIX: repeat click shows "already opened" interstitial (+&force=1 reopen).
- CARDS: launch 1-5 growth-forward + 2x-crisp; og-card claret/paper centered/crop-safe (link-preview, auto-served); X banner knob rebuilt 1500x500. Anti-repeat extended to all own-post types (generate_post avoid_texts).
- OWN-SOCIAL CADENCE: autopost + 1-click kill, M/W/F alternating Preserver/Maximizer ({"autopost_own_social":{}}, EventBridge rigacap-prod-autopost-social). Live test post 715 scheduled ~7:27PM ET Aug 7.
- STALE DRAFTS cleaned (23 deleted; kept active reply history + 715).
- DR HARDENING: scripts/backup-lambda-env.sh + restore-lambda-env.sh (SSE S3 dr/lambda-env/, baseline uploaded worker55/api52). Weekly worker handler {"backup_lambda_env":true} + EventBridge rigacap-prod-backup-lambda-env cron(0 7 ? * SUN). main.tf blueprint: 15 live-only keys declared (var refs + vars, secrets sensitive), tfvars populated from live (gitignored). All CLI EventBridge rules imported into terraform.

## ▶ ACTIVE: MULTI-PLATFORM AUTOPOST (X + Threads + Instagram) — mid-build, NOT deployed/committed
- Goal: M/W/F autopost fans out to all 3 (was twitter-only). Twitter+Threads = text; Instagram needs an image → publish_post ALREADY auto-generates a card via chart_card_generator.generate_text_card + uploads to S3 (so IG works once we create an IG post row).
- ✅ DONE (uncommitted, backend/app/services/chart_card_generator.py + backend/app/assets/fonts/): rebuilt generate_text_card into an on-brand editorial card. Bundled Fraunces + IBM Plex Sans/Mono TTFs (OFL) + registered in matplotlib. Iterated on Erik's design notes (4 rounds): claret/paper, NO giant quote mark, wordmark "RigaCap." CENTERED as a unit (claret period overlaid at measured right edge), body = IBM Plex Sans (site body font; Fraunces=wordmark only), broken BY SENTENCE with gaps, PIXEL-MEASURED wrap w/ widow pull-up (absorb ≤2-word trailing lines within a hard edge-limit — no frame overrun), block vertically centered between equal+longer claret kicker rules, footer "Walk-forward tested — signals only" + rigacap.com. Previews /tmp/ig_{maximizer,preserver,short}.png look sharp. **Awaiting Erik's final OK, then commit.**
- FONT NOTE: site uses Fraunces (font-display) medium/opsz-48 + IBM Plex Sans body; matplotlib renders Fraunces variable-default (heavier) — acceptable for the small wordmark. Card font measurement uses renderer.get_text_width_height_descent.
- ⬜ TODO after Erik OKs card: (1) modify autopost handler (main.py ~5529) to generate insight ONCE then create 1 SocialPost per platform [twitter,threads,instagram] scheduled together; (2) send_autopost_notice → note all 3 platforms + ONE kill link; (3) cancel-email endpoint → cascade-cancel same-scheduled_for research_insight siblings (one click kills all 3); (4) compile+deploy; (5) dry-run/live verify. Files: backend/main.py, email_service.send_autopost_notice, app/api/social.py cancel_post_via_email.

## ▶ NEXT / IN FLIGHT
- Bios: canonical positioning-led copy in design/brand/profiles/profile-bios.md (Erik pastes to X/IG/Threads). Confirm 715 posted ~7:27PM ET.
- OPTIONAL: already-replied-on-X interception (skip tweets @rigacap replied to manually).
- GROWTH swings: testimonials/social-proof; churn prevention; ads recheck ~Aug 8.
- Reply auto-post paid X tier = OPEN (verify before paying).

## Notes
- Billing portal 400 = NOT a bug (erikkins@gmail.com orphan; don't touch billing w/o ask).
