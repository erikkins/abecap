---
name: session-progress
description: "Live session snapshot (auto-checkpointed ~15min) — what's done, in flight, and next"
metadata:
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

# Session snapshot — Aug 27 2026 (scan+email solid; double-signal job KILLED; digest-reframe question open)

## ▶▶ GO SLOW. NO "DWAP"/"t30v" customer-facing. NO live filters/guards w/o full WF re-run. SPA reload after deploy. Embed debug in RESPONSE when logs won't surface. BUY = INSTRUCTION (mirror the book), not self-manage.

## ✅ TONIGHT: scan clean (20 buys incl fresh CRCL/NOW/NVDA/FIG, rotating_bull, CRWD fixed, 0 dangerous). 6pm daily digest = 100% verified to send (freshness gate passes). Maximizer bought DT/OKTA/RBRK (valid breakouts, real data; OKTA=real earnings gap). Expo OTA published to preview channel (portfolio-check count etc → Erik's phone, needs app reopen ×2).

## 🔴 DOUBLE-SIGNAL ALERT — KILLED (Erik: "doesn't make sense anymore")
- Was silently failing all session on a `select` NameError → I fixed it (commit 266d9aa) → it RESUMED and fired "Consider adding CRCL". Erik: it's a self-managed-portfolio artifact (book is FULL, subscribers MIRROR — "consider adding" contradicts the model).
- KILLED: `aws events disable-rule rigacap-prod-double-signals` → State=DISABLED (was cron(0 21)=5pm ET). ⏭️ TODO: update terraform SSOT (disable/remove rule) + remove handler code (post-email cleanup) so a terraform apply won't re-enable.
- **Jacob (jacob@reider.us, TRIAL, joined Aug 25) GOT IT**: is_valid()=True for trial-in-window; double_signals pref defaults True (his email_preferences=null); CRCL fresh → sent. NOT a data error (real signal, just old framing) — Erik: don't send a correction.

## ⚠️ OPEN — same "Consider adding" framing is in the 6PM DAILY DIGEST (email_service.py:647 "New Today / Consider adding"), STILL ENABLED, sends to Jacob tonight. Killing the double-signal job does NOT fix the digest. REAL FIX = reframe digest to mirror-the-book ("the book entered X / we hold Y", BUY=instruction), not "consider adding". AWAITING Erik: draft the reframe now, or leave tonight's digest + reframe deliberately?

## 🧹 POST-EMAIL CLEANUP (Erik OK'd): strip debug scaffolding (/api/debug/mxholds, _mx_debug, [prevholds-mx] print, probe_maximizer_holds; read_perf_test/fetch_scope_test/history_source_probe). Then: ONE REGIME SOT (retire _mas 5-regime); DST-aware EventBridge Scheduler; double-signal terraform/handler removal; in-chart M badge; X-reply ticker fix.

## ✅ Earlier: SMCI=0 was `os` NameError in previous-holds (fixed 3da79ab, confirmed shows 3). Widget tier cols consistent (count · best%). t30v leak fixed.
