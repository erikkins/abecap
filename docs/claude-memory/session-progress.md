---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 6 2026 (late)

## Frozen spec (load-bearing)
- Public copy: "walk-forward" not "backtest"; never expose t30v/Core/Ensemble/sleeve/DWAP; steer toward Maximizer (+$100/mo upsell). Deploy = push main → "Deploy RigaCap". NEVER `aws lambda update-function-configuration --environment`. Worker invoke = rigacap-prod-worker, AWS_PROFILE=rigacap. Read DB via `{"db_read":"SQL"}` (cast timestamps ::text — datetime not JSON-serializable). Arbitrary SQL write via `{"run_migration":true,"sql":"..."}` (commits).
- OVERLAY SSOT = frontend/src/perf_numbers.js + backend/app/services/perf_numbers.py: Preserver 21yr 7.7/0.87/−13.7, 5yr 13.0/1.28/−12.9, typ +10.7%. Maximizer 21yr 13.5/0.93/−20.8, 5yr 31.4/1.51/−14.9, typ +26.5%, 2022 −6.4%. SPY 21yr 9.8/−55. All customer-facing numbers (site, emails, social cards, portal card) now single-source from this.

## GH ACTIONS INCIDENT (Aug 6) — mostly recovered. If push doesn't trigger a run (webhook throttle): `gh workflow run "Deploy RigaCap" --ref main` (workflow_dispatch bypasses webhook — added to deploy.yml on: block). DON'T fire 2 deploys close together → Lambda ResourceConflictException (serialize).

## ✅ SHIPPED THIS SESSION (all live unless noted)
- Daily report: grounded per-tier blurb (cites only shown tickers) + symbol-set cache key; signal actionability (fresh/actionable/extended · Late); BOOK-FIRST — "Our Book" mirror leads, "Other Signals — not in our book" preserves valid signals for self-directed subs; Maximizer always shows full book; newsletter confirmation email.
- Mobile-first: dashboard regime bar stacks; Preserver book = gauge cards (entry▸today▸HWM cushion). (Marketing-page "clipping" was a headless-screenshot artifact — pages ARE responsive; Erik's real phone = ground truth.)
- Number accuracy: all backend emails/newsletter/social + launch PNGs + portal Simulated-Portfolio card = overlay SSOT. Portal card: real trailing-365 ROLLING headline + 5yr & 21yr overlay foundations (tier_serving._tier_card/_read_rolling_wf/_overlay_line; App.jsx foundations render).
- MAXIMIZER ADD-ON LAUNCHED: "Add Maximizer" button threads maximizer flag → checkout bundles add-on; founding strikethrough pricing (Preserver ~~$129~~$59, Maximizer ~~+$100~~+$79). Toggle safety verified (exits are Position.source-scoped, tier-independent — never orphaned).
- STALE-UI-ON-IDLE FIXED: 15-min token had no proactive refresh → idle poll got 200 public payload → logged-out downgrade. AuthContext now refreshes token every 11min + on tab-focus.
- WINDING DOWN section (deploying, run 31135558775 / commit 78f452c, watcher b189p3sll): served tiers now show off-tier held positions (book ≠ current signal_source) w/ their exit guidance + transition note, so lingering breakout/t30v names stay visible until they siphon off (Erik requirement).

## ▶ TOMORROW (Erik): REVISIT SOCIAL INTELLIGENCE ENGINE + VOICE / AUTOPOSTS
- Scope: the "We Called It" content pipeline + the voice of autoposts. Files: ai_content_service.py (Claude-generated posts, 3 types trade_result/missed_opportunity/we_called_it + voice), post_scheduler_service.py (scheduling windows), engagement_service.py (reply/engagement voice), newsletter_generator_service.py, components/SocialTab.jsx (admin approve/schedule UI). Auto-publish: Twitter API v2 + IG Graph (FB Page token). Approval: T-24h/T-1h emails + one-click cancel.
- Context: today I updated ai_content/newsletter NUMBERS → overlay SSOT, but the VOICE + the 2-TIER framing of the post library are NOT done — SocialTab post library still has stale 8.3/19 + isn't 2-tier (Preserver/Maximizer steer). Likely tomorrow: refresh autopost voice (honest-bold, steer→Maximizer, walk-forward-not-backtest, no internal terms), 2-tier post library rewrite, review generation prompts + engagement voice, maybe re-baseline sample posts. Also regen og-card.png ("backtest"/"19%" stale) if not already.

## Notes / non-blockers
- BILLING PORTAL 400 = NOT a bug: erikkins@gmail.com is an orphan (DB active + sub_1T0Vx but not in Stripe prod). Real payers get cus_ at checkout. To test portal use erikkins+test@gmail.com (cus_Ul2h, valid). DON'T cross-link cus ids. Erik: "don't fix what's not broken" — don't touch billing code w/o ask.
- ⚠️ POSSIBLE REAL BUG (later, don't guess): Erik "submitting +test overwrote the other one" — signup may clobber an existing customer link.
- ADS: rigacap-signals-2tier live, negatives list attached Aug 6 (recheck search terms ~Aug 8), $350/2wk gate.
- CLEANUP scratch scripts (breakout_*/maximizer_*/tier_*_today/*.bak/shapes_tpe.db/scratch json). KEEP overlay_canonical.json + perf_numbers.
