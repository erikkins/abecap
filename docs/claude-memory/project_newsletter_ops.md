---
name: Newsletter operations and timing
description: Market Measured weekly ops — Saturday generate, Sunday send, no weekend news refresh
type: project
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
**Schedule:**
- Saturday 8 PM ET: Lambda generates draft from Friday close data, saves to S3, emails admin
- Admin gets Last Look window (Saturday night / Sunday morning) to edit or leave as-is
- Sunday 10 AM ET: auto-lock and send to free list if admin hasn't acted

**Weekend news gap:** Accepted by design. The newsletter is written from Friday's close. If something massive happens Saturday/Sunday, skip the send and write a manual note — more powerful than auto-updating. The system responds to data, not headlines. That's the brand.

**Why:** Option 3 from the timing discussion (Apr 25 2026). Options 1 (Sunday AM generate) and 2 (partial refresh) were rejected — tighter timing reduces edit window, and auto-refreshing undermines the "we respond to data" thesis.

**How to apply:** EventBridge crons needed: `cron(0 0 ? * SUN *)` for generation (Sat 8 PM EDT = Sun 00:00 UTC) and `cron(0 14 ? * SUN *)` for auto-send (Sun 10 AM EDT = 14:00 UTC).
