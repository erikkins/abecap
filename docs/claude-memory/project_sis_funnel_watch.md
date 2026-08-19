---
name: project_sis_funnel_watch
description: /should-i-sell ad-landing funnel — baseline numbers + the redesign trigger Erik set
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# /should-i-sell conversion funnel — WATCH (baseline set Aug 15 2026)

Erik: "let it run a bit to see if we get any scrollers. if not, we may have to redesign the page."

## Baseline — Aug 11–15 2026 (5 days, 44 landers; 30 paid/gclid ~68%, 34 mobile ~77%)
- Landed 44 → **scroll_50 (past fold) 16 = 36%** → reach_cta (saw offer) 8 = 18% → cta click 2 → signup_open 2 → **checkout_redirect 0** → newsletter 0; bounce 11 (25%).
- Engagement was a ONE-DAY BLIP: all CTA clicks + signups happened Aug 12 (2→2). Aug 13–15 = 0 CTA clicks, bounce rising (5 of 8 on Aug 15). Steady ~9 landers/day.
- ~30 paid clicks × ~$6–7 CPC ≈ $180–200 for 0 checkouts so far.

## UPDATE Aug 19 — fold-through RECOVERED; leak moved to offer→CTA + (maybe) signup
- 7-day funnel /should-i-sell: 80 landed → 37 fold (**46%**, up from 36% baseline) → 19 saw_offer → **2 CTA clicks (11% of offer-viewers)** → 3 signup_open → 0 submit/success/stripe. /momentum: 12 landed (mostly test/organic, ~0 paid) — starved, needs impression volume before it's readable.
- Search terms (14d): targeting WORKING — crash/protection intent dominates ("stock market crash", "is the stock market going to crash", "how to protect against a stock market crash"), ~$2.70 CPC, good CTR, all on-thesis Preserve. Momentum only "momentum stocks list" (5 impr). $367 spend / 132 clicks / 0 conv — but conv tracking was blind (fixed) + it's a PAGE problem not targeting.
- NEW read: hero/fold no longer prime suspect (46%). Real leak = **offer→CTA (11%)** → redesign should target the OFFER/CTA/trust block, not the top. signup→Stripe leg TOO NEW to judge (signup_submit/success events ~2 days old, only 3 opens, 1 may be Erik).
- Erik's signup hypothesis (revisit later, NOT now): single-field email-first/progressive signup + make Google/Apple one-tap prominent (LoginModal has name+email+password+Turnstile form). Sub-funnel will localize modal-friction vs registration-fail vs Stripe-handoff once volume accrues. DECISION: wait a couple more days, then revisit.

## THE TRIGGER (Erik's rule)
- Watch metric = **scroll_50 ÷ landed** (fold-through), baseline 36%. Give it ~1 more week or ~150+ landers (so it's not one-day noise).
- If fold-through still ~mid-30s or worse → REDESIGN above-the-fold (dominant leak, mostly paid mobile — 64% never scroll).
- If it climbs to ~50%+ → page is fine, leak is deeper (offer/CTA/signup→Stripe: 0/2 reached Stripe).

## How to re-pull (worker db_read on page_views; path='/should-i-sell')
- Funnel: `SELECT event,count(*) FROM page_views WHERE path='/should-i-sell' GROUP BY event`
- Events (PageView.event): pageview, scroll_50, reach_cta, cta_hero, cta_trial, signup_open, checkout_redirect, newsletter_submit, bounce. Paid=gclid NOT NULL; is_mobile bool. Endpoint: GET /api/admin/pageviews/summary → sis_funnel (admin.py ~4775).

## Known small gaps (non-blocking)
- `country` column empty (CloudFront geo header not wired). Funnel events only began firing Aug 11–12, so earliest days undercount scroll/bounce.
- Ties to [[project_ad_conversion_tracking_jun24]] — the pre-Stripe leak (land→signup→pay) is the recurring theme.
