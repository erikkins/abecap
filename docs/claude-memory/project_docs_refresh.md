---
name: project_docs_refresh
description: "NEXT-WEEK task (Aug 2026): refresh all design/documents to current posture — reuse layout, rewrite content fresh"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

**Task (Erik, Fri Aug 7 2026, for next week):** update all the PDF/HTML docs under `design/documents/`. Approach preference: **grab the layout concepts + strong thought-blocks and write fresh, rather than editorialize each section.** Reuse the scaffolding/structure; rewrite the copy to current posture.

**The docs (all drifted — pre-date the current era; grep found navy/gold + "backtest" in several):**
- Investor report — `rigacap-investor-report.html/.pdf` (+ `-v2.html`) (Jun 10-11)
- Market pricing — `rigacap-market-pricing-analysis-2026.html/.pdf` (Jun 11) — Erik: "might be ok," verify not rewrite
- Marketing playbook — `rigacap-marketing-playbook.html/.pdf` (Jun 10-11)
- Sales playbook — `rigacap-sales-playbook.html/.pdf` (Jul 8)
- Signal intelligence — `rigacap-signal-intelligence.html/.pdf` (Jun 10) — this is the internal "dossier"; keep CONFIDENTIAL/internal-only (see [[project_secret_dossier]])
- Technical architecture — `rigacap-technical-architecture.html/.pdf` (Jun 10)

**Must reflect current posture (what changed since June):**
- 2-TIER product: Preserver (protect) + Maximizer (grow), not the old 3-tier / single "Ensemble" framing.
- Growth-forward-but-honest positioning; "Momentum, built around the drawdown."
- BRAND = claret/paper editorial (Fraunces + IBM Plex). DROP navy/gold (#172554/#fbbf24) entirely.
- "walk-forward" NOT "backtest." Numbers from OVERLAY SSOT (perf_numbers.js/.py): Maximizer 5yr 31.4/1.51/−14.9, 21yr 13.5/0.93/−20.8; Preserver 5yr 13.0/1.28/−12.9, 21yr 7.7/0.87/−13.7; SPY 5yr 14.2/−25.4, 21yr 9.8/−55.
- New machinery to document: social engine (reply deep-links, M/W/F autopost X+Threads+IG w/ editorial IG card, Kill/Edit), DR (env backup + terraform ignore_changes), tier serving.
- Kill stale taglines ("Your edge, quantified", "Beat the Market with AI-Powered Signals" — navy/gold era, in marketing-audience-strategy.md).

**Process:** HTML is the source; edit HTML then re-export PDF via headless Chrome (see CLAUDE.md "To regenerate PDFs"). Big effort across 6 docs — good candidate for a Workflow fan-out (one agent per doc) IF Erik opts into ultracode/workflows; otherwise sequential.
