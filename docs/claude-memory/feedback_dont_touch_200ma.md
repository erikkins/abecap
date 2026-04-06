---
name: Don't add buffer to SPY 200MA exit
description: The SPY < 200MA regime exit works even on razor-thin crosses — SNDK Mar 19 2026 proved it saved profits before a selloff
type: feedback
---

Do NOT add a buffer zone to the SPY < 200MA regime exit. Even a 0.04% cross (Mar 19 2026) correctly triggered an exit that saved +16.7% on SNDK before SPY dropped -1.8% over the next few days.

**Why:** What looks like a whipsaw is actually an early warning. The 200MA cross is a leading indicator of short-term weakness. Adding a 1-2% buffer would delay exits and lose profits.

**How to apply:** Be very cautious about changing this. The SNDK case argues against a buffer. But worth testing in a WF run to see the data — just don't deploy without validation. On the to-do list for investigation.
