---
name: Never invoke email handlers without verifying target_emails support
description: Always check if a Lambda email handler respects target_emails before invoking — blasted 7 regime subscribers by mistake
type: feedback
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
NEVER invoke a Lambda handler that sends emails without first verifying it supports `target_emails` filtering. The `weekly_regime_report` handler does NOT support `target_emails` — it sends to ALL active subscribers. Invoking it "to test" blasted 7 real subscribers.

**Why:** April 25 2026 — invoked `weekly_regime_report` with `target_emails: ["erik@rigacap.com"]` thinking it would only send to Erik. It sent to all 7 regime subscribers. User was rightfully angry.

**How to apply:** Before ANY email Lambda invoke: (1) Read the handler code to confirm it checks `target_emails`, (2) If it doesn't, use the `test_emails` handler instead, (3) NEVER assume a handler supports filtering just because other handlers do.
