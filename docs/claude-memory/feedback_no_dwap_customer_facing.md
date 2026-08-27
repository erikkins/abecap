---
name: feedback_no_dwap_customer_facing
description: "Never show 'DWAP' (or 'Wtd Avg') in customer-facing copy — brand-voice rule"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

Erik, emphatic (Aug 27 2026): **"NO, we are not using the term DWAP!"** — same class of rule as [[feedback_no_tape_brand_voice]].

DWAP (Daily Weighted Average Price) is an INTERNAL term. It must NOT appear in any customer-facing surface (charts, signal cards, emails, methodology, perf copy). Admin/internal tools (StrategyEditor, WalkForwardSimulator, FlexibleBacktest, AdminDashboard, StrategyGenerator) may keep DWAP.

**Approved customer-facing vocabulary (Aug 27):**
- The weighted-average-price line → **"Average price"** (was shown as "Wtd Avg" / "DWAP")
- The +5%-above-it entry threshold (`price ≥ DWAP × 1.05`, `DWAP_THRESHOLD_PCT=5.0`, still a LIVE Preserver/Core entry gate) → **"Entry trigger"** (was mislabeled "Breakout" — which wrongly collided with the Maximizer 50-day-high breakout)

**Why:** the concept is real and live (verified signals.py:1021), but the acronym is jargon and "breakout" conflated two different tier concepts.

**How to apply:** before shipping any customer copy, grep for `DWAP`/`dwap`/`Wtd Avg`; the known offenders are `frontend/src/App.jsx` and `frontend/src/perf_numbers.js`. Use "Average price" / "Entry trigger".