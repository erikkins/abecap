---
name: feedback_no_tape_brand_voice
description: "Brand voice: NEVER use the word 'tape' for the market anywhere in copy"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

Never use the word **"tape"** to mean the market/price action — anywhere in user-facing copy (landing pages, emails, portal, newsletter, AI-generated posts). Erik doesn't use the term.

**Why:** brand voice / register. Erik flagged it Aug 17 2026 after I wrote "when the tape turns choppy" on /momentum. It's a long-standing rule — `maximizer_service.py` (~line 225) already instructs the AI briefing generator: *"NEVER use the word 'tape' for the market — say 'market', 'action', or ..."*

**How to apply:** say **"the market"**, **"market action"**, **"markets"**, or **"conditions"** instead. Keep the rest of the sentence — the fix is just the word (e.g. "when the tape turns choppy" → "when the market turns choppy"; "our read on the tape" → "our read on the market"). When adding any new marketing/email/UI copy, grep new strings for "tape" before shipping.
