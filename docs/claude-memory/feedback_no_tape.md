---
name: No "tape" in brand voice
description: Never use the word "tape" in any RigaCap content — it's trader jargon
type: feedback
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
Never use the word "tape" anywhere in RigaCap content, prompts, section names, or generated text. It's trader jargon ("reading the tape," "the tape says," etc.) and violates the brand voice.

**Why:** Erik considers it exactly the kind of Wall Street insider language that makes RigaCap sound like every other signal service. The brand voice is "curious founder at dinner," not "trader on a desk."

**How to apply:** Grep for "tape" before any commit touching email, AI prompts, social content, or newsletter code. The AI content prompts already list it as banned jargon — this extends to human-written section names and descriptions too.

**ENFORCEMENT (Jul 30 2026 — the real lesson): a prompt instruction is NOT enough — negative prompts LEAK.** Every AI-generated user-facing text path MUST run its output through `backend/app/services/voice_filters.py` (`contains_banned` / `generate_with_voice_filter`, which retries then falls back). The Maximizer daily briefing leaked "tape" (Jul 29) because `generate_maximizer_briefing` was prompt-only and bypassed the filter. Fixed by wrapping it in `generate_with_voice_filter` + guarding the base `market_context` briefing (signals.py) with `contains_banned` → drop to the clean deterministic fallback on any hit. RULE: any NEW AI text path (briefings, emails, social, replies, newsletters) must go through voice_filters, not just carry the instruction in the prompt.
