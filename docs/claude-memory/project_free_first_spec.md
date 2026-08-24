---
name: project_free_first_spec
description: "BUILD SPEC — free-first, earn-the-card-later conversion model (states, free/paid line, gating, QA, drip)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Free-First Conversion Model — BUILD SPEC (sketched Aug 24 2026; DISCUSS→build)

## 0. Rationale
0 real external subs (the "6" = Erik+beta comps). Unknown brand can't borrow trusted-incumbent (Fool/SA) card-friction — the card ask is a WALL for cold cautious traffic. Ads bring OLD+MALE (55+≈65%) capital-preservers who buy TRUST not FOMO; 83% mobile. 7-day CC trial = too short (value horizon = weeks–months: 29d holds, ~weekly signals) AND too much first-touch friction. Fix: earn trust via a FREE experience, ask for the card LATER (30-day money-back guarantee). Pivot at 0 subs = free / zero revenue blast-radius → do it now.

## 1. Data model — states
- ADD `free`; RETIRE `trial`. Flow: `free` → `active` (+ `canceled`/`expired`/`past_due` all DROP TO the free view, never hard-lock).
- `free` = registered, no card, LIMITED/DELAYED view, PERPETUAL (no clock). Explicit `status='free'` Subscription record created at register (for cohort/drip/analytics — NOT "free = no row").
- `active` = paid, full. 30-day money-back = REFUND POLICY on active, not a state.
- REDEFINE `is_valid()` = paid full-access (`active` + grace) ONLY. `free` sits explicitly BELOW valid (entitled limited tier, not "invalid/locked"). This one change flips the app from "locked out" → "free tier."
- Migration = non-event: internal/beta stay comped; nothing real to migrate.

## HARD RULE (Erik, emphatic Aug 24): FREE = PROOF ONLY. ZERO ACTIONABLE.
Free shows RESULTS, never a placeable trade. All executable (entries/weights/exits/realtime/live names) = PAID.
- ⚠️ REFINED (Aug 24): a fixed "2-week delay" is NOT proof-only — a still-OPEN winning holding's NAME is actionable insight ("buy the current winner"). So NAMES REVEAL ONLY ON POSITION CLOSE, not a fixed lag:
  - Maximizer: closes on ~29d time-stop → names reveal ~29d out (Erik: "Maximizer almost needs 29-day delay").
  - Preserver: reveals on exit (trailing-stop/regime — variable, can be months).
  - = the "We Called It" CLOSED-trade ledger (already generated). "Names after they've moved" = "names after the trade FINISHED, w/ full result."
- CURRENT open book: COUNTS + performance ONLY, NO names ("holds 18 · +X% · 2 new today 🔒") — non-actionable, creates unlock-desire. No open-holding names even delayed.
- Free tier = closed-ledger (named proof, both tiers) + current counts (no names) + free market read (no live tickers) + full track record. ZERO live names/entries/weights/exits/signals.

## 2. The free/paid line — ✅ DECIDED (Aug 24)
- FREE: full LIVE market read + regime; full track record; full "We Called It" CLOSED-trade ledger; AND the book/signals with NAMES VISIBLE on a **2-WEEK DELAY** (Erik: "show names after they've moved, 2 weeks feels right") — free user sees "we bought NVDA 2wk ago, +Y% since". NO live weights, NO sell/exit alerts.
- PAID: real-time — today's live signals (tickers), mirror book (holdings+WEIGHTS+entries), sell/exit alerts, both-tier realtime.
- Principle: FREE = "the system WAS right" (proof, 2wk-delayed names + closed ledger); PAID = "what to do NOW" (realtime, actionable, exits).
- Free-riding = handled/acceptable: a 2wk-old name is PAST the clean entry (= "extended/chasing" per our own reads) AND free-riders get NO exits/weights → stale + exit-blind = degraded, not the product; it's itself the upgrade pitch. NUANCE: 2wk delay bites harder on Maximizer (29d hold→near-spent) than Preserver (multi-month→more mirror-able on a lag); if Preserver free-riding shows in data, dial = lengthen Preserver delay / show Preserver closed-only. Not solving now.
- Anti-leak: free payload = as-of (today−14d); paid = as-of today. Choke point serves the lagged vs live payload; free literally never fetches today's names.
- ✅ Free-tier LEAD personalization CONFIRMED. ✅ 30-day money-back CONFIRMED.

## 3. Free-user tier experience (Erik's Q — resolved)
- Free is NOT a tier-entitlement choice; sits below paid. Show the WHOLE engine DELAYED (both postures).
- LEAD personalized by entry door: /should-i-sell → Preserver-first (Maximizer teased); /momentum → Maximizer-first (Preserver floor teased). Optional one-tap "protect/grow" toggle (teaches one-engine-two-settings + soft-qualifies).
- Tier CHOICE (Preserver base vs +Maximizer + pricing) happens at UPGRADE only.

## 2b. Free-tier MARKET READ (⚠️ caught Aug 24 — the read LEAKS live tickers)
- The daily market read NAMES today's live entries/continuing/extended tickers (e.g. "CAG and PATH are the two clean new entries today... XOM day 10 +12% extended"). So "free gets the full live read" was a HOLE — that IS the paid signal content.
- FIX: a purpose-generated FREE-TIER read (distinct generation, NOT regex-strip which yields "[unlock] and [unlock]…"): same market/regime/behavioral color, activity described GENERICALLY ("added 2 fresh names + 1 return today — 🔒 unlock to see them; 2 holdings extended = chasing risk"), + optional 2wk-delayed/closed names as PROOF ("2 weeks ago flagged NVDA, +18% since"). ZERO live tickers. Paid read = the real one w/ tickers.

## 4. State gating & ANTI-LEAK (safety — "never send free users active prices")
- PROSE-SURFACE AUDIT (generalized from the market-read catch): gating can't just cover the structured signals list — EVERY free-facing prose surface that could name a live ticker (market read, daily-email body, AI briefings) needs a ticker-free free version via the choke point. Fail-closed guard scans free emails for ANY live ticker (not just structured fields).
- ONE entitlement choke point: `entitlement(user) -> 'paid'|'free'` from REAL state. Every content producer (dashboard payload, daily digest, drip) asks ONCE + assembles state-appropriate payload. FREE PAYLOAD OMITS paid fields at the DATA layer (absent, not CSS-hidden) — can't leak what's never fetched.
- Fail-closed pre-send guard: if recipient is free, ASSERT rendered email has ZERO paid-only fields (tickers/weights/entry-price/active-pricing) → else HOLD send + alert (like the stale-data freshness gate).
- Serve via `get_current_user_optional` (EXISTS) for tier-aware payloads. Expiry → graceful free view, NOT 403.

## 5. QA / state testability (Erik requirement)
- `?preview_state=free|active|trial|expired|past_due|canceled` (admin-gated, parallels `preview_tier`) → inspect each state's UI; compose w/ preview_tier.
- `force_state` on admin email/drip trigger + `target_emails=[erik@rigacap.com]` → send each state's email on demand (no real subs / no waiting). Same for drip (`force_state`+`force_step`).
- state × tier × email-type MATRIX + automated assertion (free render has NO tickers/book/price; active does; expired=resubscribe). Regression net.

## 6. Email capture → newsletter (never waste a lead)
- ANY captured email → newsletter enroll: explicit opt-in (EXISTS), free-account register (auto), ABANDONED signup (email entered step-1, didn't finish → enroll = funnel-leak RECOVERY).
- Do-it-right: transparent framing (abandoners: "you started signing up — here's the free read"), one-tap unsubscribe (CAN-SPAM ok, US audience), suppress hard-bounces (deliverability/protect SPF-DKIM-DMARC), DE-DUP by state (don't double-send newsletter+digest; newsletter-unsub ≠ kill transactional/account email).

## 7. Signup flow change
- register → create FREE account (status='free'), NO auto-Stripe-checkout (today App.jsx auto-routes register→Stripe). Land in free view.
- Two-step modal: step1 = email or Google/Apple (CAPTURE email immediately → newsletter enroll), step2 = password. "Credit card required" appears ONLY at the upgrade step, never at signup.
- Upgrade = card step: 30-day money-back guarantee + tier choice + Introductory pricing.

## 8. Offer/copy reframes
- Founding/First-100 (LandingPageV2) → "Introductory pricing" (keep 12-mo rate-lock, DROP FOMO/empty "first 100" — signals empty which is literally true at 0 subs). "Direct line to founder" stays adviser-tier only.
- /should-i-sell: lead with the no-card free path + trust/proof at the CTA (currently thin "Start 7-day trial"); newsletter promoted (not buried soft-catch).
- 7-day CC trial → free tier (no-card) + 30-day money-back on the PAID upgrade. (No full-access free trial; the guarantee replaces it.)

## 9. Drip rewrite (state-aware, free-first)
- RETIRE trial-clock drip (D5 "trial ending" / D8 win-back-20%). NEW free-nurture→paid: D1 welcome+free read+how-it-works · D3 "what the book did this week" (delayed, upgrade to see names) · D6 proof/"we called it" · D10+ upgrade ask (money-back + tier choice) · ongoing weekly value + periodic nudges. State-aware (free-nurture vs active vs lapsed=resubscribe). Testable via force_state/force_step.

## 10. Build sequence (each gates off #1–2)
1. ✅ BUILT (Aug 24, code only, NOT deployed — behavior-identical no-op): `Subscription.is_valid()` redefined = paid-only (`active`+period, or carded Stripe `trial` w/ stripe_subscription_id; no-card trial/`free`/lapsed → NOT valid) + `Subscription.entitlement()` method + `security.resolve_entitlement(user, sub)` = THE choke point (admin→paid, valid→paid, else free). Wired into `/dashboard` (signals.py:2053 — replaced has_valid_sub w/ resolve_entitlement; free seam tagged `entitlement:'free'`). VERIFIED no-op: of 9 sub rows only 2 valid today (akins@cookma, erikkins@gmail — active+future period); 4 "comped beta" (erik@amberland, jglynn, arnist, jpatrickfrei) ALREADY lapsed (comp period_end past → is_valid False under old code too) + 1 no-card trial (erikkins@me) long-expired → NOBODY's validity changes. No migration/live-write needed. create_trial is DEAD in live src (only backend/package/ stale); register creates NO sub; trial rows only from Stripe (carded) or admin comp. OPEN: 4 lapsed beta comps → under free-first they gracefully become free-view once #2 ships (Erik: comp-extend or leave?).
2. Tier-aware serving: free payload (delayed/proof) vs full via optional-auth; graceful drop-to-free (not 403).
3. Register→free (stop auto-Stripe); two-step modal + step-1 email→newsletter enroll.
4. QA harness: preview_state + force_state + assertions.
5. Upgrade flow: 30-day money-back, tier choice, Introductory pricing; offer/copy reframes.
6. Drip rewrite (state-aware) + abandoned-signup enrollment.
7. /should-i-sell + /momentum + landing copy → free-first.

## OPEN DECISIONS (Erik)
- Free/paid line: confirm "delayed-proof free / realtime paid" + how delayed.
- Free-tier lead: entry-door personalization + optional protect/grow toggle — OK?
- Money-back length (30d) + no full-access free trial (free tier IS the no-card path) — confirm.
- Introductory pricing exact numbers (keep $59→$129 12-mo-lock, or new intro rate?).
- Related: [[project_pricing_founding_jun23]] (existing founding system to unwind), [[project_sis_funnel_watch]] (the leak this fixes).
