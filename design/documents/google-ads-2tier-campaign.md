# Google Ads — 2-Tier High-Intent Campaign (refreshed Jul 31 2026 for launch)

> Replaces the paused `stability-search-test-a` (defense-only, informational intent,
> $695 / 134 clicks / **0 signups**). New thesis: target people shopping for a
> **signal *service/system*** — not researchers wanting a free how-to answer.
>
> **Jul 31 refresh:** landing pages are now re-baselined to the honest **overlay** numbers
> (live as of this push) and the positioning is **honest-bold** — transparency is the moat
> (a 3rd-party allocator graded the research 9/10 *because* of the disclosure). And per
> product strategy we now **steer toward Maximizer** (the +$100/mo upsell) wherever honest.

## Canonical numbers for ad copy (overlay, walk-forward — say "walk-forward," never "backtest")
- **Preserver** — typical year **+10.7%**, worst drop **−13.7%** over 21 years (S&P: 9.8% / −55%).
  Market-like return at a fraction of the drawdown. The "sleep at night" tier.
- **Maximizer** — typical year **+26.5%**, 5-yr **+31%/yr** vs the S&P's 14%, worst drop
  **−14.9%** (shallower than the S&P's −25%). In 2022 it lost just **−6.4%** while the S&P fell 20%.
- Never expose internal terms (t30v/Ensemble/sleeve/DWAP). Plain-language risk ("worst drop").

## Why the old one failed
Keywords like `when to sell stocks`, `trailing stop loss`, `how to protect from a crash`
are **informational** — free-answer seekers, not buyers. Zero signups was predictable.
This rebuild skews to **commercial intent** (service / subscription / signals) and adds
the brand-new **Maximize** angle unlocked by the 2-tier story.

---

## Campaign settings
- **Name:** `rigacap-signals-2tier`
- **Type:** Search only (no Display/Search Partners to start — cleaner data)
- **Networks:** Google Search only; Search Partners OFF; Display OFF
- **Budget:** $25/day to start (same as before) — but gate scaling on conversions
- **Bidding:** **Maximize Clicks with a $6.50 CPC cap** until ~15–30 conversions
  accumulate, THEN switch to **Maximize Conversions / tCPA**. (Can't run tCPA with 0
  conversion history.)
- **Locations:** US only
- **Ad schedule:** all day to start; the old data showed even hourly spread
- **Devices:** all, but expect ~80% mobile → mobile landing must be tight
- **⚠️ Conversion tracking:** import **GA4 `sign_up`** (and `purchase`) as Ads
  conversion actions; set **`sign_up` = PRIMARY**, `purchase` = secondary. Do this
  BEFORE spend resumes so the relaunch is measurable from click one.
  (Note: `begin_checkout` was sending stale $39/$349 — do NOT use it as the conversion.)

---

## Ad Group 1 — "Preserve" (defensive buyer intent)
**Landing:** `/track-record` (Preserve-anchored) via `/x|/ig|/t` vanity per platform N/A for search — use plain `rigacap.com/track-record?utm_source=google&utm_medium=cpc&utm_campaign=preserve`

### Keywords (start PHRASE; head terms EXACT)
- "stock signal service"
- "stock buy sell signals"
- "stock alert service"
- "systematic investing service"
- "rules based investing system"
- "momentum signal subscription"
- "portfolio risk management service"
- "trailing stop signal service"
- [capital preservation service]
- [systematic momentum strategy]
- (keep the 2 winners from old set: "protect portfolio from crash",
  "prepare portfolio for recession" — 5.6% CTR — but expect low purchase intent;
  watch cost-per-signup once tracking is live)

### Headlines (≤30 char)
`A Signal Service, Not Tips` · `Buy & Sell Signals` · `You Execute, We Signal` ·
`Market Returns, Less Pain` · `A System You Can Hold` · `Built for the Next Crash` ·
`Discipline, Not Tips` · `Risk-Managed Momentum` · `21 Years, Walk-Forward` ·
`For $250k+ Investors` · `Capital Preservation` · `Signals You Can Execute` ·
`Regime-Aware Investing` · `Stop Panic-Selling` · `Cancel Anytime`

### Descriptions (≤90 char)
- `A systematic momentum signal service built so you never get a reason to sell low.`
- `Market-like returns at a third the drawdown — 14% worst over 21 yrs vs the S&P's 55%.`
- `We publish the numbers that don't flatter us — survivorship-free, walk-forward. See it.`
- `Buy and sell signals you execute at your own broker. Discipline is the product.`
- `For investors who know what a trailing stop is and want a system, not a tip sheet.`

---

## Ad Group 2 — "Maximize" (aggressive-growth buyer intent — NEW)
**Landing:** landing/Maximize-anchored, `utm_campaign=maximize`

### Keywords (start PHRASE; head terms EXACT)
- "momentum trading signals"
- "momentum stock signals"
- "stock signals subscription"
- "best stock signal service"
- "growth stock signals"
- "algorithmic trading signals"
- "quant momentum strategy"
- "systematic momentum signals"
- [momentum signal service]
- [aggressive growth signals]

### Headlines (≤30 char)
`Growth Signal Service` · `Aggressive Momentum` · `Beat the S&P, Less Risk` ·
`In 2022 We Lost 6%` · `~31%/yr Over 5 Years` · `Growth With a Seatbelt` ·
`Dial the Risk Up` · `Push When It Pays` · `Buy & Sell Signals` ·
`One Engine, Two Settings` · `21-Year Walk-Forward` · `You Execute, We Signal` ·
`The Maximizer Setting` · `Not a Hot Stock Tip` · `Cancel Anytime`

### Descriptions (≤90 char)
- `Beat the market on return and drawdown — ~31%/yr over 5 years, a smaller worst loss.`
- `In 2022 the S&P fell 20%. Our aggressive setting lost 6%. Growth with a seatbelt.`
- `An aggressive momentum signal service with a volatility brake — growth on trend.`
- `Buy and sell signals you execute at your broker. Walk-forward tested, 21 years.`
- `One engine, two settings — Preserve or Maximize. You choose how hard to push.`

---

## Shared negative keyword list (`rigacap-negatives`)
`free` · `telegram` · `discord` · `reddit` · `penny` · `day trade` · `day trading` ·
`crypto` · `bitcoin` · `forex` · `options` · `course` · `class` · `book` · `pdf` ·
`youtube` · `jobs` · `most volatile stocks` · `high volatility stocks` ·
`volatile stocks today` · `penny stocks`
> Keep bare `volatile` NON-negative — "is the market volatile now" is on-thesis.

## Ad extensions (add at campaign level)
- **Sitelinks:** Track Record · How It Works · For Advisers · Pricing
- **Callouts:** Walk-forward tested · You execute at your broker · 21-year record ·
  Survivorship-free data · Numbers you can verify · Cancel anytime · Signals only — you keep control
- **Structured snippet (Header: "Types"):** Preserve, Maximize

---

## Launch gate / channel discipline
- At ~$6.50 CPC, even a strong 2% click→signup ≈ **$325/signup** before trial→paid.
  Set a **2-week / $350 kill-or-scale checkpoint**, same as the original test.
- Decision rule: if high-intent keywords ALSO produce 0 signups at ~$150 spend →
  the problem is the **land→signup funnel or price**, not the keywords → stop and fix
  the funnel before spending more.
- Bigger strategic question (unresolved): is paid search the right channel for a
  $129/mo product vs. doubling down on social/newsletter/adviser motion that's working?
