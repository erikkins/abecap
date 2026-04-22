---
name: AI engagement responses — founder tone, no jargon
description: Social engagement AI must use founder-earnest tone, observe same exclusion rules as posts (no "tape", etc.)
type: feedback
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
AI-generated engagement responses (replies, comments) must match the founder's voice: earnest, direct, non-salesy. Observe the same language exclusions as social posts — no trader jargon like "tape", "printing", etc.

**Why:** Consistency between posts and replies builds trust. Jargon-heavy replies undermine the approachable founder brand.

**How to apply:** When generating engagement responses in `ai_content_service.py` or social publishing flows, apply the same tone/exclusion filters used for post generation.
