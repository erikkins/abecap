---
name: Don't use dated "Added April 2026" callouts in design docs
description: Date-stamped "added" or "new" callouts sprinkled throughout docs clutter the reader and go stale. Git history is the source of truth for when something was added.
type: feedback
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
Don't write things like "(added Apr 15 2026)", "new Apr 2026", "(Apr 2026)" etc. in the design HTML docs (`design/documents/*.html`) or PDFs. Erik doesn't want them.

**Why:** They clutter the prose, draw the eye away from the actual content, and go stale immediately. Once a feature is 6 months old, the date stamp is meaningless context. Git history + commit messages are the real record of when something was added.

**How to apply:**
- When writing or updating the design docs, describe features as they currently exist — no "newly added" or "Apr 2026" markers
- If a historical callout is relevant, put it in the session's commit message, memory file, or runbook — not the user-facing doc
- When updating TPE/marketing/signal-intelligence docs with new results, just state the numbers as they are now
- Existing "(added Apr 2026)" callouts already in docs should be cleaned up next time we pass through those files

**When it IS OK to mention a date:**
- Track record tables showing year-by-year historical performance (data is inherently dated)
- Specific market events referenced in narratives ("2022 bear market", "Feb 2021 squeeze")
- Citing a specific walk-forward window ("5-year window ending April 2026")

The memory/runbook/commit layers carry the session-dated context. The user-facing docs should read as evergreen.
