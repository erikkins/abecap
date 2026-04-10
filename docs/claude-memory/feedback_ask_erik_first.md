---
name: Ask Erik before assuming code paths
description: When unsure which code branch/path to modify, ask Erik rather than guessing and debugging for 30 minutes.
type: feedback
originSessionId: 7dc69abd-ade1-4ef8-b901-42d3cee7df53
---
Ask Erik about code paths before assuming.

**Why:** Spent 30 min debugging RS Leaders not firing because I put the code in the `momentum` branch instead of `ensemble`. Erik knows the system runs ensemble (strategy 5) and could have pointed to the right branch instantly.

**How to apply:** When adding new features to the backtester or WF service, ask which strategy type / code path to target. Don't assume from reading the code — Erik knows what's live.
