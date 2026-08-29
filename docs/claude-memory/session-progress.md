---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 29 2026 (Pascal rescued into tomorrow's issue; newsletter QUEUE harden = next build)

## ▶▶ GO SLOW / BE PRECISE. NO "DWAP"/"t30v"/"tape" customer-facing. Tool-safe=descriptive. Worker payloads TRUTHY. SPA HARD-RELOAD after deploy. NEVER bare `lambda --environment`.

## 📰 NEWSLETTER BUG FIXED (immediate) + HARDEN QUEUED (build next):
- ROOT CAUSE of "queued story keeps getting dropped" (2nd time): lead-story = a HIDDEN single S3 slot (newsletter/next_lead_story.json), set only via `set_newsletter_lead_story` Lambda event, consumed-on-use, NO admin UI. Used only twice ever (June). Erik's "queued" topics lived in convo/notes, never reached S3. Generator: newsletter_generator_service.generate_draft → lead_story or get_pending_lead_story() → §02 "One Idea"; else rotating _get_topic_for_week.
- FIXED FOR TOMORROW (2026-08-30 Sunday issue): queued Pascal via set_newsletter_lead_story, regenerated draft (force) → §02 now Pascal ("Pascal's Portfolio." — I hand-set the title in S3 draft since lead-story path hardcodes title "This week's lead story."). Slot consumed. Preserved the displaced "three signals" §02 VERBATIM into new queue file s3://.../newsletter/topic_queue.json (item: three-signals-confluence, title+concept+body_preserved).
- ▶▶ NEXT BUILD (Erik greenlit, awaiting "build it"): DURABLE QUEUE. Backend: list/add/remove/reorder over topic_queue.json; generate accepts picked topic_id → that story = §02 (real title from item), removed from queue on slot; migrate legacy slot in. Frontend NewsletterTab (AdminDashboard.jsx:2477): "Story queue" panel — see/add/remove/reorder + "Use in this issue →". Admin newsletter endpoints live in backend/app/api/admin.py (:3954 generate, :3976 draft, :4058 send, lock/unlock/test). DESIGN DEFAULT: slotting a topic w/ body_preserved → verbatim; concept-only → regenerate.

## 🌑 MIRROR COCKPIT /app/next (admin-gated, SHIPPED): alignment ECLIPSE (book=sun/portfolio=moon/100%=total eclipse+corona) React canvas driven by real mirror % (MirrorCheck onAlignment); forwards ?preview_tier. SLEEVE design locked: eclipse=FIDELITY (name-only, sleeve-agnostic, total eclipse reachable by all); SIZING=separate $ readout (SnapTrade yes/CSV parse market-value/free-entry declare). NEXT cockpit: dark cohesion, drop redundant gauge, DELTA feed + DRIFT chart, account-scoping picker, CSV market-value parse.

## ✅ Mirror+SnapTrade fully shipped (connect modal, multi-brokerage, disconnect DELETE /connection/{id}, one-shot paint, KMS col-encrypt INERT until prod-swap). SnapTrade TEST key RIGACAP-LLC-TEST-EKAKS (rotate at swap). Eclipse motif = original.

## ⏭️ Also: prod-key swap runbook; sector observatory DONE; regime _mas SOT; DST Scheduler; scrub DWAP perf_numbers.js; retire WhereStocksSit.
