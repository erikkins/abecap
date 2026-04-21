---
name: Social content — balanced honesty rule (every 4th post)
description: Every 4th social post must cover a loss, stopped-out trade, quiet week, or strategy limitation. Trust-building differentiator vs pump-and-dump services.
type: project
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
**Rule:** The Social Intelligence Engine must enforce a standing ratio — every 4th post must be about one of:
- A losing trade or stopped-out position
- A week with no signals / system was quiet
- A strategy limitation we're actively working on

**Why:** This isn't just compliance hygiene. "Here's a trade that hit our stop this week and why we're okay with that" is content almost no competitor will make. It's the trust-building content that converts sophisticated skeptics into paying subscribers. It differentiates RigaCap from every pump-and-dump signal service.

**How to apply:** When building the `ai_content_service.py` post generation queue, enforce a 3:1 ratio — after 3 winner/opportunity/we-called-it posts, the system must generate a transparency post. Post types to add: `losing_trade`, `quiet_week`, `strategy_limitation`. The scheduler should track the ratio and block winner posts if overdue for a transparency post.

Noted Apr 20, 2026. Not yet built.
