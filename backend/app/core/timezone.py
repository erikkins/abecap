"""
Timezone utilities for market-aligned date boundaries.

All date *boundary* logic (is this signal fresh? how many days held?
new users today?) should use US/Eastern, since the app revolves around
US market hours.  Database *storage* stays UTC — only comparisons and
display use these helpers.
"""

from datetime import date, datetime, timedelta

import pytz

ET = pytz.timezone("US/Eastern")


def trading_now() -> datetime:
    """Current time in US/Eastern (timezone-aware)."""
    return datetime.now(ET)


def trading_today() -> date:
    """Today's date in US/Eastern."""
    return datetime.now(ET).date()


def trading_today_start() -> datetime:
    """Midnight ET today as a naive datetime (for comparing against UTC DB timestamps).

    We strip tzinfo so SQLAlchemy comparisons against naive UTC columns work.
    The resulting datetime is ET-midnight expressed as a wall-clock value.
    Since DB timestamps are UTC and ET is 4-5h behind, this gives a
    conservative boundary that matches user expectations.
    """
    now_et = datetime.now(ET)
    return now_et.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


def days_since_et(dt: datetime) -> int:
    """Number of calendar days between a (naive UTC) datetime and today in ET.

    Handles the common pattern: ``(today - created_at).days`` where
    created_at is stored as naive UTC in the database.
    """
    if dt is None:
        return 0
    d = dt.date() if isinstance(dt, datetime) else dt
    return (trading_today() - d).days
