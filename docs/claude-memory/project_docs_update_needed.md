---
name: Design documents need number updates
description: 7 HTML design docs have 40+ references to old performance numbers (289%, 31% ann, etc.) — need updating to validated results
type: project
---

## Documents Needing Updates (as of Apr 4 2026)

| Document | Old References | Priority |
|----------|---------------|----------|
| rigacap-signal-intelligence.html | 14 | HIGH |
| rigacap-investor-report.html | 10 | HIGH |
| rigacap-messaging-frameworks.html | 6 | MEDIUM |
| rigacap-technical-architecture.html | 4 | MEDIUM |
| rigacap-marketing-playbook.html | 3 | LOW |
| rigacap-pricing-analysis.html | 2 | LOW |
| rigacap-beta-tester-guide.html | 1 | LOW |

## What to Replace
- 289% → +152% avg (range +93% to +267%) for 5yr, or +497% for 10yr
- 31% annualized → ~20% annualized
- -13.2% losing year → all years positive (+6% in 2022 bear market)
- 80% win rate → 100% years positive
- ~15 signals per month → 3-4 signals per month
- 6,500+ stocks → 4,000+ stocks
- 5% of 50-day high → 3% (if mentioned)
- Regime param adjustments → disabled (if mentioned)
- Any year-by-year table → new yearly data

## Also Regenerate PDFs After HTML Updates
```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer --print-to-pdf-no-header \
  --print-to-pdf="design/documents/FILENAME.pdf" \
  design/documents/FILENAME.html
```
