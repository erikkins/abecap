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
- Public copy: "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/sleeve/DWAP; steer toward Maximizer. Deploy = push main → "Deploy RigaCap" GHA (or `gh workflow run "Deploy RigaCap" --ref main`). **NEVER push + dispatch same commit → Lambda ResourceConflictException; serialize.** NEVER `aws lambda update-function-configuration --environment`. Worker=rigacap-prod-worker, AWS_PROFILE=rigacap, us-east-1. `{"db_read":"SQL"}` (cast ts ::text); `{"run_migration":true,"sql":...}`.
- OVERLAY SSOT = perf_numbers.js/.py (Maximizer 5yr 31.4/1.51/−14.9, 21yr 13.5/0.93/−20.8; Preserver 5yr 13.0/1.28/−12.9, 21yr 7.7/0.87/−13.7; SPY 5yr 14.2/−25.4, 21yr 9.8/−55; 2008 both ~0.1 vs SPY −37.7; down-mo capture Max −0.97 vs SPY −3.85). One-click links MUST use api.rigacap.com. Admin/test email → erik@rigacap.com. BRAND = claret/paper editorial (Fraunces + IBM Plex).

## ✅ SHIPPED THIS SESSION (all live)
- REPLY SCANNER: plain-spoken Erik voice, open-with-discipline, anti-repeat over last ~5 reply-days, NO ellipsis/truncation (rejects ANY '...' regardless of length now), batch approval email, scan 4x/day, monthly recap KILLED.
- **X 403 KEY FINDING + FIX:** X Free API tier can't auto-post replies to 3rd-party tweets → approval email is now a DEEP LINK (`GET /compose-email?token=` → 302 to `twitter.com/intent/tweet?in_reply_to&text=` pre-filled) → Erik taps, hits Post. $0, no 403, no paste. Erik LOVES it, posted 4 replies. Own-posts still API-post fine (only replies blocked). See [[project_x_api_reply_403]].
- **2 FORKS:** classify_tier() picks ONE voice by thread register (FOMO/breakout→Maximizer; fear/loss→Preserver; neutral→Preserver). Angle overlay in prompt, no product names. Colour tier chip in email.
- **LAUNCH CARDS refreshed (commit 508d383, growth-forward per Erik):** 1=+31%/yr·−15%DD hook, 2=bear defense (2008 flat / ¼ downside), 3=21yr long-game table (Maximizer leads, return circled), 4=how-it-works (unchanged), 5=teaser (unchanged). All from SSOT, walk-forward labeled. Generator=design/brand/social-launch-cards.html; render via per-card isolation (CLAUDE.md method); copied to frontend/public/launch-cards/launch-1..5.png. Erik to download + post.

## ▶ NEXT / IN FLIGHT (Erik to pick)
- (a) **og-card.png refresh** — the Jun-11 link-preview card is the real "June leftover" Erik flagged ("ew"); still says "backtest"/"19%". Regen to match growth-forward set. Need to locate its generator.
- (b) **OWN-SOCIAL CADENCE** (other half of Erik's ask, so feed isn't stale) — ai_content_service makes good research_insight posts but stuck in draft; wire 2-3x/wk schedule + fork Preserver/Maximizer + one-click approve→AUTO-PUBLISH (own-posts NOT 403-blocked). Different from killed monthly recap (that was repellent stat-dump).
- Erik test the $KO deep-link end-to-end (may already have). Paid X tier (~$200/mo) for true reply auto-post = OPEN, verify before paying.
- TERRAFORM: import 2 CLI reply-scan rules; remove monthly_recap block (~1809). Newsletter anti-repeat window widen (Erik "curious").

## Notes
- Billing portal 400 = NOT a bug (erikkins@gmail.com orphan; don't touch billing w/o ask). Ads: rigacap-signals-2tier live, recheck search terms ~Aug 8.
