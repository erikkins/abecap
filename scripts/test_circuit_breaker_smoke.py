"""Local smoke test for circuit_breaker_state state machine.

Run from repo root with the backend venv:
  source backend/venv/bin/activate
  python3 scripts/test_circuit_breaker_smoke.py

Patches the S3 read/write functions to use an in-memory dict so we can
exercise the full state machine without AWS. Validates:

  1. Default disabled: every check is a no-op
  2. Enabled + manual pause: state writes, is_paused True
  3. Pause expiry: after due date, is_paused False
  4. Longer-pause-wins: long-then-short keeps long; short-then-long extends
  5. Threshold trigger: N >= threshold fires; N-1 doesn't
  6. No double-trigger while paused
  7. Event log retention (caps at MAX_EVENT_HISTORY)
  8. Multi-portfolio isolation (live vs signal_track_record)
"""

import os
import sys
from datetime import date, timedelta

# Force flag ON for testing
os.environ["CIRCUIT_BREAKER_ENABLED"] = "true"

# Make backend importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services import circuit_breaker_state as cb


# ─────────────────────────────────────────────────────────────
# In-memory S3 stub
# ─────────────────────────────────────────────────────────────

_FAKE_S3: dict = {}


def _fake_read(portfolio_type):
    import copy
    return copy.deepcopy(_FAKE_S3.get(portfolio_type, {}))


def _fake_write(portfolio_type, state):
    import copy
    _FAKE_S3[portfolio_type] = copy.deepcopy(state)


# Monkey-patch the module
cb._read_state_file = _fake_read
cb._write_state_file = _fake_write


# ─────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────

PASSES = 0
FAILS = 0


def assert_eq(actual, expected, msg):
    global PASSES, FAILS
    if actual == expected:
        print(f"  ✅ {msg}")
        PASSES += 1
    else:
        print(f"  ❌ {msg}")
        print(f"     expected: {expected!r}")
        print(f"     actual:   {actual!r}")
        FAILS += 1


def assert_truthy(actual, msg):
    assert_eq(bool(actual), True, msg)


def reset_state():
    _FAKE_S3.clear()


def section(name):
    print(f"\n=== {name} ===")


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

# Test 1: Default disabled
section("Test 1: flag disabled = all no-ops")
os.environ["CIRCUIT_BREAKER_ENABLED"] = "false"
reset_state()
assert_eq(cb.is_enabled(), False, "is_enabled() returns False")
assert_eq(cb.is_paused("live"), False, "is_paused() returns False")
result = cb.record_eod_trailing_stops("live", date(2026, 5, 3), ["A", "B", "C"])
assert_eq(result, None, "record_eod_trailing_stops() returns None")
assert_eq(_FAKE_S3, {}, "no state written to S3")

# Re-enable for remaining tests
os.environ["CIRCUIT_BREAKER_ENABLED"] = "true"

# Test 2: Manual pause
section("Test 2: manual pause writes state and is_paused returns True")
reset_state()
today = date(2026, 5, 3)
cb.request_pause("live", source="manual", days=10, today=today, context={"reason": "test"})
assert_truthy(cb.is_paused("live", today), "is_paused True on day of pause")
assert_truthy(cb.is_paused("live", today + timedelta(days=5)), "is_paused True mid-pause")
state = cb.get_state("live")
assert_eq(state.get("pause_until"), "2026-05-13", "pause_until = today+10")
assert_eq(state.get("pause_source"), "manual", "pause_source recorded")
assert_eq(len(state.get("events", [])), 1, "event logged")

# Test 3: Pause expiry
section("Test 3: pause expires correctly")
assert_eq(cb.is_paused("live", date(2026, 5, 13)), True, "is_paused True ON pause_until day")
assert_eq(cb.is_paused("live", date(2026, 5, 14)), False, "is_paused False day AFTER pause_until")
assert_eq(cb.is_paused("live", date(2026, 5, 30)), False, "is_paused False well past pause_until")

# Test 4: Longer-pause-wins
section("Test 4: longer-pause-wins semantics")
reset_state()
# Start with 5-day pause
cb.request_pause("live", "circuit_breaker", days=5, today=today)
assert_eq(_FAKE_S3["live"]["pause_until"], "2026-05-08", "initial 5-day pause")
# Try a SHORTER pause — should be shadowed
cb.request_pause("live", "regime_exit", days=2, today=today)
assert_eq(_FAKE_S3["live"]["pause_until"], "2026-05-08", "shorter pause shadowed (state unchanged)")
events = _FAKE_S3["live"]["events"]
assert_eq(len(events), 2, "shadowed event still logged")
assert_eq(events[-1]["shadowed_by_longer_pause"], True, "shadowed event flagged")
# Try a LONGER pause — should extend
cb.request_pause("live", "circuit_breaker", days=15, today=today)
assert_eq(_FAKE_S3["live"]["pause_until"], "2026-05-18", "longer pause extends (15 days from today)")
assert_eq(_FAKE_S3["live"]["pause_source"], "circuit_breaker", "source updated to extending source")

# Test 5: Threshold trigger
section("Test 5: record_eod_trailing_stops trigger threshold")
reset_state()
# threshold default = 3; below threshold = no trigger
result = cb.record_eod_trailing_stops("live", today, ["A", "B"])
assert_eq(result, None, "2 stops < 3 threshold → no trigger")
assert_eq(_FAKE_S3.get("live", {}).get("pause_until"), None, "no pause state written")
# Exactly at threshold = fires
result = cb.record_eod_trailing_stops("live", today, ["A", "B", "C"])
assert_truthy(result, "3 stops = 3 threshold → trigger fires")
assert_eq(result["source"], "circuit_breaker", "trigger source = circuit_breaker")
assert_eq(result["context"]["stops_today"], 3, "context records stops_today")
assert_eq(result["context"]["threshold"], 3, "context records threshold")
assert_eq(_FAKE_S3["live"]["pause_until"], "2026-05-13", "pause_until = today+10")
# Above threshold also fires (when not currently paused)
reset_state()
result = cb.record_eod_trailing_stops("live", today, ["A", "B", "C", "D", "E"])
assert_truthy(result, "5 stops > 3 threshold → trigger fires")

# Test 6: No double-trigger while paused
section("Test 6: no re-trigger while already paused")
reset_state()
# Trigger once
cb.record_eod_trailing_stops("live", today, ["A", "B", "C"])
events_before = len(_FAKE_S3["live"]["events"])
# Try again next day, while still paused — should NOT trigger
result = cb.record_eod_trailing_stops("live", today + timedelta(days=1), ["X", "Y", "Z"])
assert_eq(result, None, "already-paused state suppresses new trigger")
events_after = len(_FAKE_S3["live"]["events"])
assert_eq(events_after, events_before, "no new event logged while paused")

# Test 7: Event log retention
section("Test 7: event log capped at MAX_EVENT_HISTORY")
reset_state()
# Fire MAX+5 events — should retain only last MAX
for i in range(cb.MAX_EVENT_HISTORY + 5):
    cb.request_pause("live", "manual", days=1, today=today + timedelta(days=i))
events = _FAKE_S3["live"]["events"]
assert_eq(len(events), cb.MAX_EVENT_HISTORY, f"events capped at {cb.MAX_EVENT_HISTORY}")

# Test 8: Multi-portfolio isolation
section("Test 8: live and signal_track_record are independent")
reset_state()
cb.request_pause("live", "manual", days=10, today=today)
assert_truthy(cb.is_paused("live", today), "live is paused")
assert_eq(cb.is_paused("signal_track_record", today), False, "STR is NOT paused")
cb.request_pause("signal_track_record", "manual", days=3, today=today)
assert_truthy(cb.is_paused("signal_track_record", today), "STR now paused")
assert_truthy(cb.is_paused("live", today), "live still paused (independent)")
assert_eq(_FAKE_S3["live"]["pause_until"], "2026-05-13", "live state unchanged by STR pause")
assert_eq(_FAKE_S3["signal_track_record"]["pause_until"], "2026-05-06", "STR has its own state")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"RESULTS: {PASSES} passed, {FAILS} failed")
print(f"{'=' * 60}")
sys.exit(0 if FAILS == 0 else 1)
