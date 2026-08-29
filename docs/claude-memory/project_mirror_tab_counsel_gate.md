---
name: project_mirror_tab_counsel_gate
description: Mirror tab stays ADMIN-ONLY until counsel signs off; then flip 2 isAdmin guards to open to all subs
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Mirror tab — gated on counsel before opening to all subscribers (Aug 29 2026)

The **Mirror tab in /app** (eclipse cockpit: alignment eclipse + book ledger + Today strip +
drift sparkline; shows how a user's own holdings line up with the model book) is shipped but
**ADMIN-ONLY**. Erik: **do NOT open it to all subscribers until he hears back from counsel** —
then he'll give the explicit go.

**Why gate it:** the Mirror compares a user's actual portfolio to the published model book — brushes
against investment-advice / suitability territory, so it needs a legal read first. (Design is
deliberately tool-safe: every line is a factual set-comparison, never an instruction — but counsel
still reviews before it's subscriber-facing.)

**How to open it when Erik says go (one-step flip):** in `frontend/src/App.jsx`, flip the two
`isAdmin` guards — (1) the Mirror tab **button** (`{isAdmin && (<button ...setActiveTab('mirror')...`)
and (2) the tab **content branch** (`activeTab === 'mirror' && isAdmin ?`). The EclipseGlyph in the
tab label + green "you hold this" bubbles on the Signals book are already live for everyone-safe
(empty held-set = no change), so no other changes needed. Rebuild + push.

**Do NOT flip preemptively.** Wait for Erik's word post-counsel. [[session-progress]]
