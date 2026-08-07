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

## ▶ ACTIVE (Aug 7): SOCIAL INTELLIGENCE ENGINE + VOICE — growth goal (Erik: 1→5→10→50→100→500 subs; social = zero-CAC discovery). Erik NEW: FORK Preserver vs Maximizer messaging (two buyers).
- AUDIT DONE (read ai_content_service.py SYSTEM_PROMPT + 14 real social_posts via db_read). VERDICT: VOICE ALREADY TOP-NOTCH — research_insight posts editorial/honest/discipline-led/human ("The calm days aren't the test"; "Doing nothing is a position"; "Defensible beats flattering"). SYSTEM_PROMPT strong (Erik-founder voice, lead-with-discipline, own losses, overlay numbers correct, anti-AI-tells). Generator is NOT the problem.
- 3 GAPS: (A) NO tier forking — all leans Preserver, nothing hunts Maximizer growth persona (Erik's ask). (B) Monthly recap = growth-repellent stat-dump (auto-template, non-AI: "July 0/11 winners, best −15.6%") — needs editorial framing. (C) THROUGHPUT: AI drafts stuck in `draft` (11/14 recent research_insight unpublished) → verify publish pipeline not stalled (content that doesn't ship = no subs).
- Priority: C→A→B. Files: ai_content_service.py (social_posts.text_content), post_scheduler_service.py, engagement_service.py, reply_scanner_service.py, social_posting_service.py, SocialTab.jsx.

## ▶ ENGAGEMENT REPLIES = the strongest growth lever (Erik's confession: he's an "awkward poster", hasn't been posting the manual-email opps; his north-star reply = behavioral empathy from lived experience: "Watching −8% become −22% telling yourself it's still a 'thesis hold' — I've done it, the loss is the same number regardless of the story"). 
- TWO engagement systems: (1) engagement_service.py = OLD, emails drafts daily 9AM for MANUAL post (Erik likes the email, hates copy-paste); BETTER voice prompt (empathy/temperature-down/skip-hard). (2) reply_scanner_service.py = NEWER, PERSISTS contextual_reply drafts (event `{"scan_replies":{...}}`, uses Twitter user-timeline not paid search) + social_posting_service.post_to_twitter supports reply_to_tweet_id → one-click-approve→auto-publish plumbing EXISTS. But reply_scanner's voice is WEAKER (ticker-match "we flagged $X +56.6%" flex).
- **DRY-RUN (verify C, DONE):** reply_scanner works — 24 accts/106 tweets/6 replies. BUT drafts are RETURN-LED ticker-flexes ("Our system flagged GOOGL, +56.6%") that VIOLATE our own voice rules (lead-with-discipline-not-returns, never smug "we called it", don't pitch) AND are NOT Erik's behavioral-empathy north star. Auto-posting these = self-promotion, worse than silence.
- **ERIK DECISIONS:** approve-first for WEEK 1 (nothing posts until 1-click approve in email — no copy-paste), then flip to default-post+1-click-kill; scan 2–3×/day. KILL the monthly-recap autopost (useless + repellent 0/11 stat-dump, <10 followers). 
- **VOICE RULE (Erik):** OPEN with discipline; a positive flex is OK but only as SECONDARY understated evidence, never the hook. **DONE:** reworked _build_reply_system_prompt (reply_scanner_service.py:28) to open-with-discipline + lived-experience + result-secondary + SKIP-if-only-a-flex; added SKIP guard in _generate_reply (~830) + short-text guard. Committed f20f7a3, deploying (watcher bc42oq1v0). **NEXT: dry-run `{"scan_replies":{"dry_run":true,"since_hours":24}}` on worker to show new drafts.** Then wire 2–3×/day scan + one-click-APPROVE email (approve-first wk1) + auto-publish reply; kill monthly recap.
- **2 FORKS (Erik reminder) — do after base voice validated:** for REPLIES, fork = tier-voice by THREAD REGISTER (anxiety/loss thread → Preserver "loss is the same number" voice; momentum/FOMO/growth thread → Maximizer "growth w/ seatbelt"). For OUR OWN posts = two targeted versions (Preserver vs Maximizer buyer). social_posts needs a tier column for this.
- Also pending: og-card.png regen; SocialTab 2-tier library; tier-forking (A) after engagement.

## Notes / non-blockers
- BILLING PORTAL 400 = NOT a bug: erikkins@gmail.com is an orphan (DB active + sub_1T0Vx but not in Stripe prod). Real payers get cus_ at checkout. To test portal use erikkins+test@gmail.com (cus_Ul2h, valid). DON'T cross-link cus ids. Erik: "don't fix what's not broken" — don't touch billing code w/o ask.
- ⚠️ POSSIBLE REAL BUG (later, don't guess): Erik "submitting +test overwrote the other one" — signup may clobber an existing customer link.
- ADS: rigacap-signals-2tier live, negatives list attached Aug 6 (recheck search terms ~Aug 8), $350/2wk gate.
- CLEANUP scratch scripts (breakout_*/maximizer_*/tier_*_today/*.bak/shapes_tpe.db/scratch json). KEEP overlay_canonical.json + perf_numbers.
