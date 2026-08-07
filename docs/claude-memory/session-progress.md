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
- Public copy: "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/sleeve/DWAP; steer toward Maximizer. Deploy = push main → "Deploy RigaCap" GHA (or `gh workflow run "Deploy RigaCap" --ref main`). **NEVER push + dispatch same commit → Lambda ResourceConflictException; serialize.** NEVER `aws lambda update-function-configuration --environment`. Worker=rigacap-prod-worker (ARN arn:aws:lambda:us-east-1:149218244179:function:rigacap-prod-worker), AWS_PROFILE=rigacap. `{"db_read":"SQL"}` (cast ts ::text); `{"run_migration":true,"sql":...}`.
- OVERLAY SSOT = perf_numbers.js/.py: Maximizer 5yr 31.4/1.51/−14.9, 21yr 13.5/0.93/−20.8; Preserver 5yr 13.0/1.28/−12.9, 21yr 7.7/0.87/−13.7; SPY 5yr 14.2/−25.4, 21yr 9.8/−55; 2008 both ~flat vs SPY −37.7; down-mo capture ~−0.97 vs SPY −3.85. One-click links MUST use api.rigacap.com. Admin email → erik@rigacap.com. BRAND=claret/paper (Fraunces + IBM Plex); mono ONLY for data/tables, never prose.

## ✅ SHIPPED THIS SESSION (all live)
- REPLY SCANNER: plain-voice, discipline-first, 5-reply-day anti-repeat, NO ellipsis (rejects '...' any length), 2 tier FORKS (classify_tier by thread register). **X 403: Free tier can't API-reply to 3rd parties → approval email is a DEEP LINK to X pre-filled composer (compose-email→intent/tweet); Erik taps+Posts. Erik LOVES it.** See [[project_x_api_reply_403]].
- CARDS: launch 1-5 refreshed GROWTH-FORWARD (1=+31%/−15% hook, 2=bear defense, 3=21yr long-game w/ DD circled — NOT return, so it doesn't clash w/ card1's 31%, 4=how-it-works, 5=teaser). Fixed mono-on-prose fonts. Rendered 2x (crisp when X downscales). og-card rebuilt claret/paper, CENTERED+crop-safe, growth-forward; meta tags backtested/19%→walk-forward+SSOT. Generators: design/brand/social-launch-cards.html + design/og-card-source.html. Cards use SAME Google Fonts as site.
- **OWN-SOCIAL CADENCE (just shipped):** autopost + 1-click KILL, Mon/Wed/Fri, alternating Preserver/Maximizer. Handler {"autopost_own_social":{tier?,dry_run?,window_hours?}} → generate_research_insight(tier, avoid_texts=last 8 days) → status=scheduled +4h → send_autopost_notice heads-up+kill email (reuses cancel-email) → publish cron auto-posts unless killed (own-posts NOT 403-blocked). EventBridge rigacap-prod-autopost-social cron(0 13 ? * MON,WED,FRI) → worker (CLI-created, +lambda perm). Dry-run verified both voices excellent + forked.

## ▶ NEXT / IN FLIGHT
- OFFER: fire one LIVE autopost now (schedules a real post +4h w/ kill link) so Erik tests the full flow end-to-end — his call (real post to X).
- HEADER BANNER (x-banner.png 1500×500, the knob) still blurry + no numbers — separate asset, source generator not located. Options: re-sharpen knob at 2x, or rebuild growth-forward. Erik's call.
- 20 stale research_insight drafts left in DB (won't post; harmless). Could cancel for tidiness.
- TERRAFORM RECONCILE: import CLI-made rules (2 reply-scan + autopost-social); remove monthly_recap block (~1809).
- Reply flow: after wk1 Erik still taps deep-link (approve-first by nature). Paid X tier for reply auto-post = OPEN, verify before paying.

## Notes
- Billing portal 400 = NOT a bug (erikkins@gmail.com orphan; don't touch billing w/o ask). Ads: rigacap-signals-2tier live, recheck search terms ~Aug 8.
