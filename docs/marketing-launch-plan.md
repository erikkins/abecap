# RigaCap Marketing Launch Plan

## Phase 1: Foundation (Week 1)

### Social Intro Posts (X + Instagram)
Post 1-2 brand introduction messages establishing who RigaCap is:
- "We built a trading system that combines three signals into one. Here's how it works."
- Walk-forward performance teaser (289% 5-year, no hindsight bias)
- Link to /track-record

### Follow & Engage Target Accounts
Follow the accounts from `social-target-list.md`. Start engaging manually:
- Like/retweet relevant market commentary
- Reply to posts mentioning stocks we've recently signaled
- Goal: get on their radar before any promotional content

### SEO Baseline
- robots.txt + sitemap.xml deployed (DONE)
- Structured data (JSON-LD) added (DONE)
- Canonical URL set (DONE)
- Monitor Google Search Console for indexing

---

## Phase 2: Regime Intelligence Posts (Week 2-3)

### Market Regime Content
Use the 7-regime detection system as thought leadership:
- "The market just shifted from weak_bull to rotating_bull. Here's what that means for stock selection."
- "Our system detected panic_crash conditions 2 days before the VIX spike. This is why regime detection matters."
- Post regime shifts as they happen (real-time credibility)

### "We Called It" Engine Activation
- Start publishing AI-generated trade result posts from walk-forward data
- Schedule 2-3 posts/week through the admin approval pipeline
- Mix: 60% trade results, 20% missed opportunities, 20% regime calls

---

## Phase 3: Social Proof Amplification (Week 3-4)

### Contextual Reply Strategy (NEW - Auto-Generated)
When followed accounts post about stocks we've signaled:
1. System detects the mention (keyword match against our signal history)
2. Claude API generates a contextual reply linking our "We Called It" post
3. Admin reviews and posts (one-click from notification)

Example flow:
- @TastyTrade posts: "NVDA breaking out of consolidation, watching for entry"
- RigaCap reply (auto-generated): "Our ensemble flagged NVDA on Jan 15 when all three factors aligned. It's up 12% since. Full breakdown: [link to We Called It post]"

This is the highest-leverage social activity — uses their audience, their context, and our real track record.

### Reply Content Types
1. **Direct match**: They mention a stock we signaled -> link the specific trade result
2. **Regime alignment**: They discuss market conditions -> share our regime detection angle
3. **Strategy overlap**: They discuss momentum/technical analysis -> position ensemble as evolution

---

## Phase 4: Growth Engine (Month 2+)

### Content/SEO
- Blog posts on rigacap.com (market commentary, strategy explainers)
- "How our ensemble detected the [X] rotation" long-form posts
- Target keywords: "trading signals", "momentum trading system", "market regime detection"

### Referral Program Promotion
- Highlight "Give a month, get a month" in social posts
- Beta testers get referral codes in welcome email (already built)

### Performance Dashboard for Users
- Personal ROI tracking ("am I making money?")
- This becomes the ultimate retention + word-of-mouth driver

---

## Key Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Landing page -> trial | 3-5% | GA4 |
| Trial -> paid | 3-5% | Stripe |
| Monthly churn | <5% | Stripe |
| Social impressions/week | 10K+ | Twitter Analytics |
| Engagement rate | >2% | Twitter Analytics |
| "We Called It" click-through | >1% | UTM tracking |
| Contextual reply -> profile visit | Track | Twitter Analytics |

---

## Social Reply Automation Architecture (To Build)

### Data Flow
```
Cron (hourly) -> Fetch recent posts from followed accounts (Twitter API v2)
    -> Match stock symbols against signal history (DB query)
    -> For matches: Claude API generates contextual reply
    -> Save as draft in social_posts table (status: pending_reply)
    -> Admin notification email with preview + approve/skip buttons
    -> On approve: publish via Twitter API
```

### Requirements
- Twitter API v2: `GET /2/users/:id/tweets` for followed accounts' recent posts
- Symbol extraction: regex + NLP to detect stock mentions ($NVDA, "Nvidia", etc.)
- Signal history lookup: match against walk-forward trades + recent buy signals
- Claude API prompt: include their tweet text, our trade data, and brand voice guidelines
- Admin UI: new "Replies" tab in social admin showing pending contextual replies

### Implementation Priority
This is the highest-ROI social feature. Build after the "We Called It" engine is running smoothly (Phase 2).
