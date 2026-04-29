# RigaCap Editorial Pipeline

> **What this file is:** the markdown sibling of the editorial CMS described in §12 of `MarketingNewsletterStrategyCLAUDE.md`. Every entry maps 1:1 to the `EditorialItem` TypeScript schema in that doc. Until the DynamoDB CMS + admin UI is built, this file is the source of truth for the IDEA / DEVELOPING / OUTLINED backlog. When the CMS ships, port these entries mechanically (one section → one PutItem).
>
> **What this file is NOT:** a draft repository. Once an item reaches DRAFTED stage, the body lives in its appropriate destination (newsletter draft S3 key, blog post repo, drip email template, etc.) and this file just tracks the metadata + reference link. Drafts in markdown are fine for early stages; once a draft is being actively edited, link out.
>
> **Maintenance cadence (per §12 of the strategy doc):**
> - **Sunday 30 min:** Triage IDEA items, advance any that are ready, confirm the week's writing commitment.
> - **First Sunday of month, 60 min:** Calendar planning, theme audit, archive what's been published.
> - **Quarterly, 2 hours:** Strategy alignment, compliance audit, performance review (which themes resonate?).
>
> **Schema fields in use** (matching `EditorialItem` from §12):
> `id` · `title` · `description` · `stage` · `channel` · `theme[]` · `priority` · `outline` · `references` · `estimatedWordCount` · `marketingRuleNotes` · `audienceFilter`
>
> Stages: `IDEA` → `DEVELOPING` → `OUTLINED` → `DRAFTED` → `EDITED` → `SCHEDULED` → `PUBLISHED` → `ARCHIVED`
> Channels: `NEWSLETTER` · `REGIME_REPORT` · `BLOG` · `METHODOLOGY` · `SOCIAL_X` · `SOCIAL_FB` · `SOCIAL_IG` · `SOCIAL_LINKEDIN` · `DRIP_EMAIL` · `REENGAGEMENT` · `PODCAST_PITCH` · `INTERNAL_DOC`
> Priorities: `P0` (do this week) · `P1` (this month) · `P2` (this quarter) · `P3` (someday)

---

## Index

### Newsletter §02 candidates (educational rotation)
- [CG-EXPLAINED-NL](#cg-explained-nl) — Cascade Guard, in plain English
- [FRICTION-VS-SIM-NL](#friction-vs-sim-nl) — Why we publish a *lower* number than the backtest
- [MEDIAN-VS-MEAN-NL](#median-vs-mean-nl) — How one outlier can carry an "average"
- [CARRY-RULE-NL](#carry-rule-nl) — A binary rule worth +5pp
- [RESTRAINT-BEARS-NL](#restraint-bears-nl) — Why we don't build a bear strategy
- [LUCK-FACTOR-NL](#luck-factor-nl) — Don't trust 18 months of any strategy. Including ours.
- [SHARPE-CRITIQUE-NL](#sharpe-critique-nl) — Why the Sharpe ratio is over-fetishized
- [NO-PREDICTIONS-NL](#no-predictions-nl) — Why we don't predict — and what we do instead
- [CASH-POSITION-NL](#cash-position-nl) — Cash isn't a position. Except when it is.
- [START-DATE-VARIANCE-NL](#start-date-variance-nl) — Same strategy, multiple start dates, 50pp swings

### Blog / longform
- [DATA-INTEGRITY-BLOG](#data-integrity-blog) — The day we caught our own backtest lying
- [BACKTEST-INTEGRITY-BLOG](#backtest-integrity-blog) — Survivorship bias in retail signal services
- [METHODOLOGY-DEEP-DIVE](#methodology-deep-dive) — Methodology page expansion

### Drip email enhancements
- [DR-002-FRICTION-UPDATE](#dr-002-friction-update) — Add friction-adjusted framing to "What you're paying for"
- [DR-003-QUIET-EXPANSION](#dr-003-quiet-expansion) — Sharpen the "you haven't seen a signal yet" reframe

### Social long-form (LinkedIn / IG carousel)
- [REFRESH-NUMBERS-SOCIAL](#refresh-numbers-social) — The +204% → +160% honesty post
- [MEGACAP-PATTERN-SOCIAL](#megacap-pattern-social) — A 12-event observation about post-shock recovery
- [VARIANCE-HONESTY-SOCIAL](#variance-honesty-social) — When the same strategy returns +252% or +109% based on start date

### Regime report material
- [PER-REGIME-DISCLOSURE-RR](#per-regime-disclosure-rr) — Per-regime track record (publish when data is ready)

### Internal / methodology docs
- [REGIME-SUBSTRATEGY-INTERNAL](#regime-substrategy-internal) — Why we evaluated and shelved per-regime sub-strategies

---

## Newsletter §02 candidates

### CG-EXPLAINED-NL
- **title:** "Cascade Guard, in plain English."
- **description:** What Cascade Guard is, when it fires, why pausing trading is itself a strategic decision. For paid subscribers who see "system paused" notifications but don't grasp the mechanics.
- **stage:** IDEA
- **channel:** NEWSLETTER (§02), with BLOG candidate sibling
- **theme:** discipline, methodology
- **priority:** P1
- **estimatedWordCount:** 220
- **outline:**
  1. The seven market regimes, briefly — most investors think bull or bear; reality has more granularity.
  2. What "panic_crash" looks like in market data, not in vibes.
  3. Why a pause is a position, not the absence of one.
  4. The cost of being early to re-enter (one anonymized historical example).
- **references:** `design/documents/rigacap-signal-intelligence.html` Cascade Guard section
- **marketingRuleNotes:** No specific tickers. No return claims tied to the pause. Use marketing name "Cascade Guard," not internal "circuit breaker."

### FRICTION-VS-SIM-NL
- **title:** "Why we publish a lower number than the backtest shows."
- **description:** The friction-adjusted vs simulation distinction. Why a 21.5% claim is harder to make and more honest than a 23% one. Trust-building piece.
- **stage:** IDEA
- **channel:** NEWSLETTER (§02)
- **theme:** discipline, transparency, methodology
- **priority:** P1
- **estimatedWordCount:** 230
- **outline:**
  1. Backtest reality: simulations don't include slippage, taxes, or human behavior.
  2. The friction haircut: what we subtract and why.
  3. Why most retail services *don't* do this — and why we do.
  4. The honest math: what the gap actually is.
- **references:** `docs/MarketingNewsletterStrategyCLAUDE.md` §15
- **marketingRuleNotes:** Cite specific friction-adjusted numbers from §15 only. No range claims.

### MEDIAN-VS-MEAN-NL
- **title:** "How one outlier can carry an average — and why median tells the truth."
- **description:** Educational piece on how a single lucky-window result can lift a backtest's headline number. Inoculates readers against industry hype numbers.
- **stage:** IDEA
- **channel:** NEWSLETTER (§02)
- **theme:** statistics, methodology, transparency
- **priority:** P2
- **estimatedWordCount:** 240
- **outline:**
  1. The arithmetic mean: simple and seductive.
  2. The geometric mean: more conservative, technically correct for compound returns.
  3. The median: what most subscribers actually experience.
  4. A real-world example (anonymized): one outlier shifts a "average" by 30+ percentage points.
  5. What we choose to publish, and why.
- **references:** Internal session notes (the +204% provenance investigation)
- **marketingRuleNotes:** Anonymize the specific scenarios — no implication that any specific competitor inflates numbers.

### CARRY-RULE-NL
- **title:** "A simple rule that added five percentage points."
- **description:** Story of a binary rule change to Cascade Guard pause logic. How a one-line decision (does the pause carry past a rebalance boundary?) was worth more than weeks of parameter optimization.
- **stage:** IDEA
- **channel:** NEWSLETTER (§02), with potential BLOG sibling
- **theme:** methodology, simplicity-as-a-feature
- **priority:** P1
- **estimatedWordCount:** 220
- **outline:**
  1. What we noticed: pauses were getting reset at our rebalance boundaries (an artifact, not a design choice).
  2. The fix: pause carries through boundaries. Same rule, different boundary semantic.
  3. The result, validated across multiple start dates.
  4. The lesson: simple rules beat complex optimization, especially with limited data.
- **references:** Memory note `feedback_wf_prod_parity.md`; this week's WF runs
- **marketingRuleNotes:** No specific dollar return claims; describe in percentage points.

### RESTRAINT-BEARS-NL
- **title:** "Why we don't build a bear-market strategy."
- **description:** The discipline of *not* over-engineering. Eleven years has only ~5 true bear regimes — too few samples for any optimization to mean anything. What we do instead.
- **stage:** IDEA
- **channel:** NEWSLETTER (§02), with BLOG sibling
- **theme:** discipline, statistical-honesty, contrarian
- **priority:** P1
- **estimatedWordCount:** 240
- **outline:**
  1. The temptation: build a sub-strategy for bear markets, sell it as "downside protection."
  2. The math: 4-5 events is too few. Anything we'd build would be lottery odds at best.
  3. What we do instead: pause via Cascade Guard, hold cash, accept being sidelined.
  4. The trade-off: we miss bear-market upside (rare but real). We accept the trade-off because the alternative is overfitting.
  5. The brand value: restraint when restraint is warranted.
- **references:** Memory note `project_regime_substrategy_decision.md`
- **marketingRuleNotes:** Frame as a methodology choice, not a performance claim.

### LUCK-FACTOR-NL
- **title:** "Don't trust 18 months of any strategy. Including ours."
- **description:** Behavioral piece. The first-year survivor effect. Why short track records — even good ones — are not signal.
- **stage:** IDEA
- **channel:** NEWSLETTER (§02)
- **theme:** statistics, behavioral, contrarian
- **priority:** P2
- **estimatedWordCount:** 230
- **outline:**
  1. The seductive 18-month track record problem.
  2. Sample-size statistics: how many data points it takes to differentiate skill from luck.
  3. Why backtests across multiple start dates help (variance disclosure).
  4. What our 11-year walk-forward actually shows — and what it can't tell us.
  5. The honest disclosure: 5 years is enough for confidence; 18 months is not.
- **references:** Multi-start-date variance findings from the 8-date carry-on run
- **marketingRuleNotes:** Cite full 5-10y windows where possible.

### SHARPE-CRITIQUE-NL
- **title:** "What the Sharpe ratio doesn't tell you."
- **description:** Industry observation piece. Why the Sharpe is over-fetishized in retail finance. What metrics tell you more.
- **stage:** IDEA
- **channel:** NEWSLETTER (§02)
- **theme:** methodology, contrarian, statistics
- **priority:** P3
- **estimatedWordCount:** 230
- **outline:**
  1. What Sharpe actually measures.
  2. What it misses: skewness, drawdown duration, regime-dependence.
  3. Why a 0.92 Sharpe with great drawdown control beats a 1.5 Sharpe with hidden tail risk.
  4. What we look at instead.
- **references:** None yet.
- **marketingRuleNotes:** Don't disparage specific competitor metrics; frame as methodology preference.

### NO-PREDICTIONS-NL
- **title:** "We don't predict. Here's what we do instead."
- **description:** Brand-defining piece. Why predictions are the wrong frame and "respond to data" is the right one. Echoes the marketing doc's voice rules.
- **stage:** IDEA
- **channel:** NEWSLETTER (§02)
- **theme:** discipline, brand, methodology
- **priority:** P2
- **estimatedWordCount:** 220
- **outline:**
  1. Why every "the market will..." prediction is a coin flip dressed up.
  2. The alternative: respond, don't predict. What that looks like in code.
  3. The cost: we miss "called the top" moments. We don't optimize for those.
  4. The benefit: we don't blow up when our prediction is wrong.
- **references:** `docs/MarketingNewsletterStrategyCLAUDE.md` §11 (forbidden phrases)
- **marketingRuleNotes:** Reinforces the brand voice rules; no return claims.

### CASH-POSITION-NL
- **title:** "Cash isn't a position. Except when it is."
- **description:** Behavioral piece on opportunity cost during pause periods.
- **stage:** IDEA
- **channel:** NEWSLETTER (§02)
- **theme:** behavioral, methodology
- **priority:** P3
- **estimatedWordCount:** 220
- **outline:**
  1. The "I'm in cash, I'm doing nothing" psychological trap.
  2. Why cash *is* a position — measured against the alternative of being long during a regime that punishes longs.
  3. How we frame pause periods to subscribers.
- **references:** Cascade Guard methodology section
- **marketingRuleNotes:** Avoid implying cash always outperforms during pauses.

### START-DATE-VARIANCE-NL
- **title:** "Same strategy, many start dates, fifty-point swings."
- **description:** Methodology transparency piece. Real-world variance disclosure as a trust-builder.
- **stage:** IDEA
- **channel:** NEWSLETTER (§02), with SOCIAL_LINKEDIN sibling
- **theme:** methodology, transparency, statistics
- **priority:** P2
- **estimatedWordCount:** 230
- **outline:**
  1. The same five-year window, run from multiple start dates.
  2. The result spread: from +109% to +252%.
  3. Why this isn't a flaw — it's the truth about path-dependency.
  4. What we publish (the average) vs what subscribers experience (one specific path).
- **references:** `docs/numbers-citations-registry.md`
- **marketingRuleNotes:** **NEVER cite the specific number of start dates** (per `feedback_no_7_dates.md`). Use "multiple" or "many." Cite full distribution, not just best/worst.

---

## Blog / longform

### DATA-INTEGRITY-BLOG
- **title:** "The day we caught our own backtest lying."
- **description:** Long-form essay on the pickle-corruption discovery, the over-fit number, and the decision to refresh and republish lower numbers. The most powerful trust move in the pipeline.
- **stage:** IDEA
- **channel:** BLOG, with podcast-pitch potential
- **theme:** discipline, transparency, brand-defining
- **priority:** P1
- **estimatedWordCount:** 1,200
- **outline:**
  1. The setup: how we built our walk-forward infrastructure.
  2. The discovery: an indicator-corruption bug that inflated one start date by 30+ pp.
  3. The choice: republish lower honest numbers or quietly fix and never mention it.
  4. The math: what changed (avg +204% → +160%, MDD 32% → 20.4%).
  5. The principle: lower honest numbers > higher inflated numbers. Always.
  6. What this means for any subscriber who joined under the old numbers.
- **references:** Memory notes on Trial 37 over-fit, clean data refresh
- **marketingRuleNotes:** Compliance-sensitive — likely needs attorney review before publishing. Material change in disclosed performance.
- **attorneyReviewed:** false (must complete before publish)

### BACKTEST-INTEGRITY-BLOG
- **title:** "What's wrong with most backtest claims in retail signal services."
- **description:** Industry-observation longform. Survivorship bias, over-fitting, cherry-picked windows, friction-blind sims.
- **stage:** IDEA
- **channel:** BLOG
- **theme:** methodology, contrarian, brand-defining
- **priority:** P2
- **estimatedWordCount:** 1,500
- **outline:**
  1. Why "5 years, 350% return" claims are usually meaningless.
  2. The five common failures: survivorship, in-sample optimization, friction-blind, cherry-picked window, no walk-forward.
  3. How to evaluate a backtest claim as a subscriber.
  4. What we do differently — line by line.
- **references:** None yet.
- **marketingRuleNotes:** Don't name specific competitors. Frame as industry critique.

### METHODOLOGY-DEEP-DIVE
- **title:** "Methodology page — full expansion."
- **description:** Update the public `/methodology` page with the carry-on rule, regime detection details, and refreshed numbers. Foundational content, not a one-off post.
- **stage:** IDEA
- **channel:** METHODOLOGY (the public methodology page)
- **theme:** methodology, transparency
- **priority:** P1
- **estimatedWordCount:** 2,000+
- **outline:**
  1. Strategy entry conditions (breakout timing, momentum quality, volume confirmation). NEVER name DWAP in public copy — use "proprietary timing reference" or similar.
  2. Cascade Guard mechanics + the carry-on rule.
  3. Position sizing + rebalance cadence.
  4. The 7 regimes — what they are, how detected, what we do in each (light touch given the substrategy decision).
  5. Walk-forward methodology + multi-start-date variance.
  6. Friction-adjusted vs simulation: what we publish and why.
- **references:** `docs/numbers-citations-registry.md`; existing `/methodology` page
- **marketingRuleNotes:** Coordinate with `numbers-citations-registry.md` — must update in lockstep with any number refresh.

---

## Drip email enhancements

### DR-002-FRICTION-UPDATE
- **title:** "Update DR-002 ('What you're paying for') with friction-adjusted framing."
- **description:** Existing drip email is missing the strongest trust hook (we publish lower, more honest numbers). One-paragraph addition.
- **stage:** IDEA
- **channel:** DRIP_EMAIL
- **theme:** transparency, lifecycle
- **priority:** P2
- **estimatedWordCount:** +80 (insert)
- **outline:**
  - Add a paragraph after the current value-prop section: "Most services quote simulation returns. We quote friction-adjusted: the number after slippage and behavior. Our 21.5% annualized claim is the conservative one — and we'd rather be conservative and accurate than aggressive and approximate."
- **references:** `backend/app/services/email_service.py` DR-002 section
- **marketingRuleNotes:** Numbers must match `numbers-citations-registry.md` canonical column.

### DR-003-QUIET-EXPANSION
- **title:** "Sharpen DR-003 ('You haven't seen a signal yet') with regime-context."
- **description:** Current quiet-period reframe is good but generic. Add a one-line context tag from the active regime when the email fires.
- **stage:** IDEA
- **channel:** DRIP_EMAIL
- **theme:** lifecycle, methodology
- **priority:** P3
- **estimatedWordCount:** +50 (insert)
- **outline:**
  - Pull active regime from `regime_forecast_snapshots`. Insert: "Right now we're in a [regime_label] regime — that historically produces [N] signals per [timeframe]. Quiet now is exactly what the system should be doing."
- **references:** `backend/app/services/email_service.py` DR-003; `regime_forecast_snapshots` table
- **marketingRuleNotes:** Don't promise signal frequency; describe historical pattern.

---

## Social long-form

### REFRESH-NUMBERS-SOCIAL
- **title:** "We just refreshed our published returns from +204% to +160%. Here's why we're proud of the smaller number."
- **description:** LinkedIn / IG carousel. The honesty story compressed to social-friendly format.
- **stage:** IDEA
- **channel:** SOCIAL_LINKEDIN, SOCIAL_IG
- **theme:** transparency, brand-defining
- **priority:** P1 (publish around the same time as the data integrity blog post)
- **estimatedWordCount:** 400 (LinkedIn long-form) / 8-slide carousel (IG)
- **outline:**
  1. Slide 1: Headline. "We caught our own backtest exaggerating. We fixed it. Here's what changed."
  2. Slide 2: The discovery — corrupted indicator data inflated one start date.
  3. Slide 3: The new (lower, honest) numbers — +160%, 21.1% ann, 20.4% MaxDD.
  4. Slide 4: What didn't change — risk profile is actually *better* than what we modeled.
  5. Slide 5: Why we're republishing rather than quietly fixing.
  6. Slide 6: CTA — "we'd rather be conservative and accurate."
- **references:** [DATA-INTEGRITY-BLOG](#data-integrity-blog) (publish in lockstep)
- **marketingRuleNotes:** Same numbers as canonical registry; attorney review on the blog post extends to this if synchronized.

### MEGACAP-PATTERN-SOCIAL
- **title:** "What twelve historical shocks tell us about post-shock recovery."
- **description:** Educational social piece. The mega-cap recovery pattern (NVDA +11%, AAPL +8%, etc.) — without disclosing the specific holdings or implying a tradeable strategy.
- **stage:** IDEA
- **channel:** SOCIAL_LINKEDIN, SOCIAL_X
- **theme:** methodology, education
- **priority:** P3
- **estimatedWordCount:** 300 (LinkedIn) / 280 (X thread)
- **outline:**
  1. The data: 12 vol shocks across 11 years.
  2. The pattern: large-caps drop similarly to the index but recover meaningfully harder.
  3. The caveat: this is a 12-event sample. Direction is consistent; magnitude isn't.
  4. The lesson: post-shock periods reward patience more than they reward conviction.
- **references:** This session's bear-window inspection script + output
- **marketingRuleNotes:** Don't name specific tickers. Refer to "large-cap basket" or "index leaders." No implied trade recommendation.

### VARIANCE-HONESTY-SOCIAL
- **title:** "When the same strategy returns either +252% or +109% based on start date alone."
- **description:** Honesty post on path-dependency. Anchored to our actual 8-date variance.
- **stage:** IDEA
- **channel:** SOCIAL_LINKEDIN, SOCIAL_X
- **theme:** transparency, methodology
- **priority:** P2
- **estimatedWordCount:** 350 (LinkedIn)
- **outline:**
  1. We ran the same five-year window from eight start dates.
  2. The spread: +109% (worst) to +252% (best). All positive. Average +160%.
  3. Why this isn't a strategy flaw — it's the truth about how compound returns and entry timing interact.
  4. What this means for a subscriber: your specific path will differ from the headline. We'll publish all of them.
- **references:** `docs/numbers-citations-registry.md`; 8-date carry-on results
- **marketingRuleNotes:** Cite full distribution; don't selectively quote.

---

## Regime report material

### PER-REGIME-DISCLOSURE-RR
- **title:** "Per-regime track record — quarterly disclosure."
- **description:** Per the [regime-substrategy decision](#regime-substrategy-internal), publishing per-regime stats is a unique trust play even without per-regime sub-strategies. This is the disclosure piece.
- **stage:** IDEA (blocked on data — needs the regime-PnL attribution script run)
- **channel:** REGIME_REPORT
- **theme:** transparency, methodology
- **priority:** P2 (after registry refresh + clean-data trade-level recompute)
- **estimatedWordCount:** 600
- **outline:**
  1. Brief: the seven regimes.
  2. Trades by regime (count, win rate, avg return per trade).
  3. Contribution to total PnL by regime.
  4. Where we add alpha vs the index, regime by regime.
  5. The honest soft-spot disclosure (likely range_bound).
- **references:** Memory note `project_regime_substrategy_decision.md`
- **marketingRuleNotes:** Specific historical performance by regime — Marketing Rule applies. Likely attorney review before first publish.

---

## Internal / methodology docs

### REGIME-SUBSTRATEGY-INTERNAL
- **title:** "Why we evaluated and shelved per-regime sub-strategies (Apr 2026)."
- **description:** Internal record of the Apr 2026 evaluation. Not for external publication; reference doc for any future session that proposes building bear/range-bound sub-strategies.
- **stage:** PUBLISHED (lives at `~/.claude/projects/-Users-erikkins-CODE-stocker-app/memory/project_regime_substrategy_decision.md`)
- **channel:** INTERNAL_DOC
- **theme:** methodology, decision-record
- **priority:** P0 (already done)
- **estimatedWordCount:** ~600 (existing)
- **references:** Memory note `project_regime_substrategy_decision.md`

---

## Themes glossary

When tagging an entry's `theme[]`, prefer terms from this list to keep the index searchable:

- `discipline` — restraint, process, what we *don't* do
- `methodology` — how the system works, why
- `transparency` — disclosure, honesty about limits/variance
- `statistics` — math, sample size, distributions
- `behavioral` — investor psychology, cognitive traps
- `contrarian` — challenges industry consensus
- `brand` — voice, identity, positioning
- `lifecycle` — onboarding, retention, drip
- `education` — teaches a concept (current §02 default)
- `decision-record` — internal "why we chose X" memos
