---
name: NEVER mention specific number of walk-forward start dates anywhere
description: CRITICAL — never cite "7", "8", "12" or any specific count of WF start dates. Use "multiple" or "many." This rule applies to ALL surfaces: public AND internal. Repeatedly violated; treat as load-bearing.
type: feedback
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
**CRITICAL — repeatedly violated rule.** Never cite the specific number of walk-forward start dates (7, 8, 12, or any number) anywhere — public OR internal docs.

**Use only:** "multiple start dates", "many start dates", "across start dates", "regardless of when you started"

**Never:** "8 start dates", "7 different windows", "avg of 8", "across 8", "8 dates", etc.

**Why:** (1) The specific count sounds amateurish and underwhelming — sample size of 7 or 8 is small and exposes us to "your sample is too small" criticism. (2) The number is an implementation detail, not a selling point. (3) Internal docs sometimes get quoted externally; consistent rule across all surfaces eliminates risk.

**How to apply:**
- All marketing copy, landing/track-record/blog pages, social posts, emails, scripts, and templates: use "multiple" or "many."
- Internal docs (registry, editorial pipeline, marketing strategy doc, CLAUDE.md): also use "multiple" — eliminates leak risk.
- Surface map templates and JSON: use the `wf_5y_num_start_dates_label` key (= "multiple start dates"), NOT the `wf_5y_num_start_dates` integer.
- When generating any new content, **scan output for digit-counts of dates before showing to user.**
- Only place a count is acceptable: internal compute scripts that need the actual integer for math (CSV row counts, stat aggregation). Never in any string that could surface.

**Violations to remember:**
- 2026-04-28 session: introduced "8 start dates" in TrackRecordPageV2 + LandingPageV2 perf tables and supporting paragraphs, plus surface_map block templates. User flagged with caps-lock frustration ("I HAVE SAID THIS MANY TIMES BEFORE"). Fixed across all surfaces same session.
