# RigaCap — Compliance Brief: "Portfolio Overlay" Feature (for counsel review)

**Prepared for:** [Securities counsel]
**Prepared by:** Erik Kins, RigaCap, LLC
**Date:** August 2026
**Purpose:** Pre-clear the framing, display, and disclaimers for a proposed feature *before* we build/ship it. We want to confirm it stays within RigaCap's current non-adviser posture. **We are not asking for a fairness opinion on the strategy — only whether the feature and its language keep us outside the definition of an "investment adviser."**

---

## 1. Who we are today (current posture)
RigaCap, LLC is **not** a registered investment adviser. We publish a **single, impersonal model portfolio and a set of rules-based signals** that are identical for every subscriber — a bona-fide, regular publication. We rely on the **publisher's exclusion** to the Investment Advisers Act (§202(a)(11)(D); *Lowe v. SEC*, 472 U.S. 181 (1985)): our content is impersonal, not tailored to any individual's situation, financial goals, or risk tolerance. Every surface carries "signals/information only — not individualized advice; execute at your own broker" and a past-performance disclaimer (language you previously reviewed).

## 2. What we want to add
A **"Portfolio Overlay"** onboarding tool. A prospect or user provides a list of tickers they own or watch (by CSV/paste initially; possibly a read-only brokerage connection via an aggregator later). We then show them how **our already-published, impersonal signals and universe** intersect with those tickers — as an information/education layer, **never as a recommendation to buy, sell, or hold anything.**

## 3. What it WILL do (we believe = impersonal, factual, non-advice)
- **Universe membership (factual):** "X of your N tickers are in the universe RigaCap tracks."
- **Historical signal overlap (aggregate, retrospective):** "Y of your tickers have triggered a RigaCap signal at some point in our published walk-forward history." Shown first as a **count**, not per-ticker directives.
- **Retrospective, mechanical rule application (educational):** "Here is what our published rules *did* with these tickers over our historical record" — describing how our already-public system behaved, framed in the past tense, applied mechanically and identically for everyone.
- **Signal matching, not new advice:** any per-ticker view simply **intersects the user's list with signals we already publish impersonally to all subscribers.** The signals exist independently of the user; we are only highlighting overlap.

## 4. What it will NOT do (the lines we intend to stay behind)
- No prescriptive or individualized statements: never "you should sell X," "buy Y," "rebalance to Z."
- No tailoring to the individual's stated goals, risk tolerance, tax situation, time horizon, or account size.
- No forward-looking, personalized recommendation of any kind.
- No trade execution and no discretionary authority (any future brokerage connection would be **read-only**).
- No holding ourselves out as an adviser, planner, or manager.

## 5. Display tiers (value + compliance escalate together)
1. **Public (no account):** aggregate **counts only** — "in universe" / "ever signaled." Instant, impersonal, a teaser.
2. **Free account:** per-ticker **historical/factual** overlay ("in universe? / has this ever been a signal?") + retrospective "what our rules did," all impersonal.
3. **Paid:** the full impersonal signal service (same as today).

## 6. Data handling
Users may submit their holdings (tickers, and optionally email). We intend to **store submissions** (tied to email if given, else an anonymous session ID) for product analytics, lead follow-up, and to power the overlay. Holdings lists can be sensitive; our Privacy Policy will disclose collection, use, retention, and that data is not sold. **Question for counsel:** any additional disclosure/consent needed for storing a user-submitted holdings list (non-account-linked), and does a read-only brokerage connection (via an aggregator such as SnapTrade) change that analysis?

## 7. Questions for counsel
1. Does the **aggregate-count** public display (Tier 1) stay clearly within the publisher's exclusion?
2. Does the **per-ticker historical/"has ever been a signal"** view (Tier 2) remain impersonal, or does applying our lens to a user-specific list risk being deemed individualized advice — and if borderline, what specific framing/disclaimers cure it?
3. Is a **read-only** brokerage connection (data aggregation only, no execution, no discretion) compatible with our non-adviser posture, or does pulling a specific user's account tip the analysis?
4. What **exact disclaimer language** should sit on the overlay UI and any overlay-driven emails?
5. Any state-level (vs. federal) adviser considerations we should be aware of for a broad U.S. consumer audience?

## 8. What we're asking for
A go / no-go (with any required wording) on Tiers 1 and 2 of the overlay, plus guidance on the read-only brokerage connection for a later phase. We will not ship any tier publicly until the framing and disclaimers are blessed.

---
*Internal note: this brief states RigaCap's intent and questions; it is not legal advice and is subject to counsel's review.*
