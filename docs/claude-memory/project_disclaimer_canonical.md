---
name: Canonical disclaimer language
description: The publisher's-exemption-aligned disclaimer language to use across all customer-facing surfaces, plus the three Lowe factors that the publisher's exemption depends on.
type: project
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
Per attorney opinion (Apr 30 2026, Erik's uncle), RigaCap operates under the **publisher's exemption** to the Investment Advisers Act of 1940 (§202(a)(11)(d)). No SEC registration required. No California registration required (Cal. Corp. Code §25009(a) mirrors federal exemption). Lingley v. Seeking Alpha (2024) is a friendly precedent — Seeking Alpha gives more info than RigaCap and was held exempt.

**Why:** Decided Apr 30 2026 after attorney response. The exemption depends on three Lowe factors that RigaCap must keep satisfying — language alone doesn't preserve the exemption, but customer-facing language must align with operating reality.

**The three Lowe factors RigaCap must preserve:**
1. **Impersonal advice** — same signals to all subscribers, not tailored to individual portfolios. Never offer per-subscriber recommendations.
2. **Bona fide commentary** — methodology-driven analysis, not promotional touting of specific securities. Marketing the product itself is fine; touting individual securities is not.
3. **General and regular circulation** — weekly newsletter + scheduled signal cadence. Skipping weeks weakens the claim. The Saturday/Sunday newsletter cron is legally load-bearing, not just marketing-load-bearing.

**How to apply:**

When updating customer-facing surfaces, use one of three canonical disclaimer forms:

**LONG FORM** (legal Terms, primary landing/methodology/track-record footers):
> RigaCap, LLC is not a registered investment advisor, broker-dealer, or financial planner. The signals, data, content, and methodology provided through the Service are for information purposes only and are not a solicitation to invest, purchase, or sell securities in which RigaCap has an interest. They do not constitute investment advice, financial advice, trading advice, or any other kind of advice.

**SHORT FORM** (page footers, blog footers, email body footers):
> For information purposes only. Not a solicitation to invest, purchase, or sell securities in which RigaCap has an interest. RigaCap, LLC is not a registered investment advisor. Past performance does not guarantee future results.

**MICRO FORM** (copyright lines, header bars, where space is at a premium):
> &copy; YYYY RigaCap, LLC · For information purposes only

**Key phrases that must appear** (any combination depending on space):
- "for information purposes only"
- "not a solicitation to invest, purchase, or sell securities in which RigaCap has an interest"
- "RigaCap, LLC is not a registered investment advisor"
- "past performance does not guarantee future results"

**The "in which RigaCap has an interest" clause is forward-looking.** Erik holds no positions in signaled securities currently, but the clause preserves optionality. Before any first personal/business trade in a signaled security, Erik should consult uncle on structure (timing, disclosure, holding-period rules) — anti-scalping (§206) and anti-touting (§17(b)) still apply even with the publisher's exemption.

**Skip these surfaces** — they're not customer-facing:
- AI prompt instructions in `ai_content_service.py`, `signals.py`, `reply_scanner_service.py`, `instagram_comment_service.py` — these tell Claude/the AI not to give advice. They should stay as-is.

**The full attorney opinion** lives in Erik's email (Apr 30 2026 from uncle Juris). If new surfaces need disclaimer language, return to this file for the canonical forms — don't paraphrase from prior conversation drift.
