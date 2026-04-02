---
name: WF launch protocol
description: Always include periods_limit=1 in WF payloads and test ONE job before launching batches
type: feedback
---

**ALWAYS include `periods_limit: 1` in walk-forward job payloads.** Without it, the Lambda tries to run all ~131 periods in one 900s invocation and times out. The previous successful batches all used periods_limit=1.

**Why:** On Apr 1-2 2026, launched 28+ jobs without periods_limit. ALL timed out at 900s on period 1. Jobs registered in DB as "running" but had zero progress. Wasted ~14 hours before discovering the issue.

**How to apply:** 
1. Every WF payload MUST include `"periods_limit": 1`
2. Launch ONE test job first, verify it chains (check state file growing in S3)
3. Only then launch the full batch
4. Consider making periods_limit=1 the DEFAULT instead of 0
