"""Daily digest v3 — dashboard-sourced editorial render (both tiers).

Ported from the approved local renders (scratchpad digest_dash.py / build_max.py). Renders from
the exact data the scheduler passes into send_daily_summary — GENERATES NOTHING, invents nothing.

Layout: static per-regime ORB hero (cid:hero, no baked text) → live-HTML hook + regime·date +
S&P/VIX read → Today's Market Read → Today's Moves (only if entries today) → book sections →
Approaching. Every ticker links to its chart. No list truncation. Fluid max-width:600, mobile-safe.
"""
import os
from datetime import datetime

SERIF = "Georgia,'Times New Roman',serif"
SANS = "Helvetica,Arial,sans-serif"
MONO = "'Courier New',Courier,monospace"

HERO_REGIMES = {'strong_bull', 'rotating_bull', 'weak_bull', 'range_bound',
                'recovery', 'weak_bear', 'panic_crash'}
_HERO_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'email_heroes')


def digest_hero_path(market_regime):
    """Absolute path to the static per-regime ORB hero PNG, or None if missing (header still
    renders without the image). Falls back to range_bound for an unknown regime."""
    regime = (market_regime or {}).get('regime') or (market_regime or {}).get('current_regime') or 'range_bound'
    if regime not in HERO_REGIMES:
        regime = 'range_bound'
    p = os.path.abspath(os.path.join(_HERO_DIR, f'hero_{regime}.png'))
    return p if os.path.exists(p) else None


def _sector_of(symbol):
    """Best-effort real sector from the in-memory universe cache. Empty string if unavailable —
    never a fabricated/filler value."""
    try:
        from app.services.stock_universe import stock_universe_service
        info = (stock_universe_service.symbol_info or {}).get(symbol) or {}
        return info.get('sector') or ''
    except Exception:
        return ''


def _shell(inner):
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
            '<body style="margin:0;padding:0;background:#EDE7D8;-webkit-text-size-adjust:100%;-ms-text-size-adjust:100%">'
            '<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background:#EDE7D8">'
            '<tr><td align="center" style="padding:0 0 18px">'
            '<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="width:100%;max-width:600px;background:#F5F1E8">'
            + inner +
            '</table></td></tr></table></body></html>')


def _hero_text(hook, regime_name, date_str, spy, spy_chg, spy_up, fear, vix, tier_badge=''):
    """Live-HTML hero text that sits just under the static ORB image (on paper): the honest hook,
    regime·date, and the S&P/VIX read row. Nothing here is baked into the image."""
    arrow = '&#9650;' if spy_up else '&#9660;'
    acol = '#2D5F3F' if spy_up else '#8F2D3D'
    badge = ''
    if tier_badge:
        badge = ('<div style="font-family:%s;font-size:11px;letter-spacing:1.5px;text-transform:uppercase;'
                 'color:#7A2430;font-weight:bold;margin-bottom:8px">%s</div>' % (MONO, tier_badge))
    return (
        '<tr><td style="padding:18px 22px 0;text-align:center">'
        + badge +
        ('<div style="font-family:%s;font-size:38px;font-weight:bold;color:#141210;line-height:1.06">%s</div>' % (SERIF, hook)) +
        ('<div style="font-family:%s;font-style:italic;font-size:17px;color:#8A6A46;margin-top:6px">%s &middot; %s</div>' % (SERIF, regime_name, date_str)) +
        '<table role="presentation" cellpadding="0" cellspacing="0" align="center" style="margin:16px auto 0;border-top:1px solid #DDD5C7;border-bottom:1px solid #DDD5C7">'
        '<tr>'
        + ('<td style="padding:12px 22px;border-right:1px solid #DDD5C7;text-align:center">'
           '<div style="font-family:%s;font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:#8A8172">S&amp;P 500</div>'
           '<div style="font-family:%s;font-size:20px;color:#141210;margin-top:4px">%s <span style="color:%s;font-size:14px">%s%s</span></div></td>'
           % (MONO, SERIF, spy, acol, arrow, spy_chg))
        + ('<td style="padding:12px 22px;text-align:center">'
           '<div style="font-family:%s;font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:#8A8172">Volatility</div>'
           '<div style="font-family:%s;font-size:20px;color:#141210;margin-top:4px">%s <span style="color:#8A8172;font-size:14px">&middot; VIX %s</span></div></td>'
           % (MONO, SERIF, fear, vix))
        + '</tr></table></td></tr>')


def _sec(label, count, sub):
    cnt = (' &middot; ' + count) if count else ''
    return ('<div style="margin-top:32px;border-top:2px solid #141210;padding-top:14px">'
            '<div style="font-family:%s;font-size:12px;letter-spacing:1.2px;text-transform:uppercase;color:#7A2430;font-weight:bold">%s%s</div>'
            '<div style="font-family:%s;font-size:14px;color:#8A8172;margin-top:3px">%s</div></div>'
            % (MONO, label, cnt, SANS, sub))


def _row(sym, left2, rightbig, rightsmall, rc='#141210'):
    url = 'https://rigacap.com/app?chart=%s' % sym
    left2 = left2 or ''
    return ('<table role="presentation" width="100%%" cellpadding="0" cellspacing="0" style="border-bottom:1px solid #DDD5C7"><tr>'
            '<td valign="middle" style="padding:13px 0">'
            '<a href="%s" style="font-family:%s;font-size:19px;font-weight:bold;color:#7A2430;text-decoration:none">%s</a>'
            '<span style="font-family:%s;font-size:13px;color:#8A8172;margin-left:8px">%s</span></td>'
            '<td valign="middle" align="right" style="padding:13px 0">'
            '<span style="font-family:%s;font-size:16px;color:%s">%s</span>'
            '<span style="font-family:%s;font-size:11px;text-transform:uppercase;color:#8A8172;margin-left:8px">%s</span>'
            '</td></tr></table>'
            % (url, SERIF, sym, SANS, left2, MONO, rc, rightbig, MONO, rightsmall))


def _pnl_c(v):
    return '#2D5F3F' if (v or 0) >= 0 else '#8F2D3D'


def _footer(tier_line):
    return ('<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="margin-top:34px"><tr><td align="center">'
            '<table role="presentation" cellpadding="0" cellspacing="0"><tr><td style="background:#141210;border-radius:5px">'
            '<a href="https://rigacap.com/app" style="display:inline-block;font-family:' + MONO + ';font-size:15px;font-weight:bold;'
            'letter-spacing:1.5px;text-transform:uppercase;color:#F5F1E8;text-decoration:none;padding:16px 42px">Open Your Dashboard</a>'
            '</td></tr></table></td></tr></table>'
            '<div style="font-family:' + SERIF + ';font-style:italic;color:#8A8172;font-size:15px;text-align:center;margin-top:18px">'
            'Signals only &mdash; execute via your broker.</div>'
            '<div style="border-top:1px solid #DDD5C7;margin-top:24px;padding-top:16px;text-align:center;font-family:' + MONO + ';'
            'font-size:11px;color:#8A8172;line-height:1.7">' + tier_line + '<br>'
            '<a href="https://rigacap.com/app" style="color:#7A2430">Dashboard</a> &middot; '
            '<a href="https://rigacap.com/app" style="color:#7A2430">Manage Alerts</a> &middot; '
            '<a href="https://rigacap.com/auth/unsubscribe" style="color:#7A2430">Unsubscribe</a><br>'
            '<span style="color:#A79E8C">For information purposes only. Not a solicitation. '
            'Past performance does not guarantee future results.</span></div>')


def _market_read(text):
    if not text:
        return ''
    return ('<div style="margin-top:26px;font-family:%s;font-size:12px;letter-spacing:1.2px;text-transform:uppercase;color:#7A2430;font-weight:bold">Today\'s Market Read</div>'
            '<p style="font-family:%s;font-size:18px;line-height:1.62;color:#2A2620;margin:10px 0 0">%s</p>'
            % (MONO, SERIF, text))


def _regime_meta(market_regime, regime_forecast, date):
    mr = market_regime or {}
    rf = regime_forecast or {}
    regime = mr.get('regime') or rf.get('current_regime') or 'range_bound'
    name = mr.get('regime_name') or rf.get('current_regime_name') or regime.replace('_', ' ').title()
    spy = '%.2f' % (mr.get('spy_price') or 0)
    chg = mr.get('spy_change_pct')
    spy_up = (chg or 0) >= 0
    spy_chg = ('+' if spy_up else '') + ('%.1f%%' % (chg or 0))
    vl = mr.get('vix_level') or 0
    vix = '%.1f' % vl
    fear = 'Very calm' if vl < 14 else 'Calm' if vl < 20 else 'Elevated' if vl < 30 else 'Fearful'
    dt = date if isinstance(date, datetime) else datetime.utcnow()
    date_str = dt.strftime('%B %-d, %Y') if hasattr(dt, 'strftime') else str(date)
    return regime, name, spy, spy_chg, spy_up, vix, fear, date_str, dt


def _build_preserver(signals, market_regime, watchlist, market_context, regime_forecast, date):
    regime, name, spy, spy_chg, spy_up, vix, fear, date_str, dt = _regime_meta(market_regime, regime_forecast, date)
    sigs = signals or []
    today_key = dt.strftime('%Y-%m-%d')
    todays_moves = [s for s in sigs if s.get('ensemble_entry_date') == today_key]
    buyzone = [s for s in sigs if s.get('entry_status') in ('fresh', 'actionable')]
    holding = [s for s in sigs if s.get('entry_status') == 'extended']
    wl = watchlist or []

    hook = ('%d new move%s today' % (len(todays_moves), '' if len(todays_moves) == 1 else 's')) if todays_moves else 'No new moves today'

    body = ('<tr><td><img src="cid:hero" width="600" height="440" style="display:block;width:100%;max-width:600px;height:auto;border:0" alt="RigaCap Daily Signals"></td></tr>')
    body += _hero_text(hook, name, date_str, spy, spy_chg, spy_up, fear, vix)
    body += '<tr><td style="padding:0 22px 44px">'
    body += _market_read(market_context)

    if todays_moves:
        body += _sec("Today's Moves", "", "New positions entered or exited today &mdash; the sync list to mirror at your broker.")
        for s in todays_moves:
            body += _row(s.get('symbol', ''), s.get('sector', ''), "$%.2f" % (s.get('price') or 0), "new entry", '#2D5F3F')

    body += _sec("In the buy zone", str(len(buyzone)), "Held names still within entry range &mdash; where the model is currently positioned.")
    for s in sorted(buyzone, key=lambda x: -(x.get('ensemble_score') or 0)):
        mv = s.get('move_since_signal_pct') or 0
        body += _row(s.get('symbol', ''), s.get('sector', ''), ('+' if mv >= 0 else '') + '%.1f%%' % mv,
                     "score %d" % int(s.get('ensemble_score') or 0), _pnl_c(mv))

    if holding:
        body += _sec("Holding &middot; extended", str(len(holding)), "Positions past their entry range &mdash; held, not new buys.")
        for s in sorted(holding, key=lambda x: -(x.get('ensemble_score') or 0)):
            mv = s.get('move_since_signal_pct') or 0
            body += _row(s.get('symbol', ''), s.get('sector', ''), ('+' if mv >= 0 else '') + '%.1f%%' % mv,
                         "%d%% ext" % int(s.get('extended_pct') or 0), _pnl_c(mv))

    if wl:
        body += _sec("Approaching", str(len(wl)), "Names nearing their trigger &mdash; not yet in the book.")
        for w in wl:
            dtr = w.get('distance_to_trigger')
            dist = ('%.1f%%' % dtr) if isinstance(dtr, (int, float)) else '—'
            body += _row(w.get('symbol', ''), w.get('sector', ''), dist, "to trigger", '#7A2430')

    body += _footer("RigaCap &middot; Disciplined Momentum Strategy")
    body += '</td></tr>'
    return _shell(body)


def _build_maximizer(breakout_book, market_regime, breakout_radar, market_context, regime_forecast, date):
    regime, name, spy, spy_chg, spy_up, vix, fear, date_str, dt = _regime_meta(market_regime, regime_forecast, date)
    book = breakout_book or []
    radar = breakout_radar or []
    todays = [p for p in book if (p.get('status') == 'new' or p.get('is_fresh'))]

    hook = ('%d new breakout%s today' % (len(todays), '' if len(todays) == 1 else 's')) if todays else 'No new moves today'

    body = ('<tr><td><img src="cid:hero" width="600" height="440" style="display:block;width:100%;max-width:600px;height:auto;border:0" alt="RigaCap Maximizer"></td></tr>')
    body += _hero_text(hook, name, date_str, spy, spy_chg, spy_up, fear, vix, tier_badge='&#9670; Maximizer')
    body += '<tr><td style="padding:0 22px 44px">'
    body += _market_read(market_context)

    if todays:
        body += _sec("Today's Moves", "", "New breakout entries today &mdash; the sync list to mirror at your broker.")
        for p in todays:
            body += _row(p.get('symbol', ''), _sector_of(p.get('symbol', '')), "$%.2f" % (p.get('price') or 0), "new entry", '#2D5F3F')

    held = [p for p in book if not (p.get('status') == 'new' or p.get('is_fresh'))]
    body += _sec("The breakout book", str(len(book)), "What the Maximizer sleeve holds now &mdash; each rides a ~29-day time-stop.")
    for p in sorted(held, key=lambda x: -(x.get('days_held') or 0)):
        dh = p.get('days_held')
        hold = p.get('hold_days') or 29
        left = p.get('days_left')
        if left is None and dh is not None:
            left = max(0, hold - dh)
        pnl = p.get('pnl_pct')
        rightbig = ('%dd held' % dh) if dh is not None else 'holding'
        rightsmall = ('~%dd to stop' % left) if left is not None else ''
        rc = '#141210'
        if pnl is not None:
            rightbig = ('+' if pnl >= 0 else '') + '%.1f%%' % pnl
            rc = _pnl_c(pnl)
            rightsmall = (('%dd · ~%dd to stop' % (dh, left)) if (dh is not None and left is not None)
                          else (('~%dd to stop' % left) if left is not None else 'holding'))
        body += _row(p.get('symbol', ''), _sector_of(p.get('symbol', '')), rightbig, rightsmall, rc)

    if radar:
        body += _sec("Approaching", str(len(radar)), "Names nearing a breakout trigger &mdash; not yet in the book.")
        for r in radar:
            pct = r.get('pct_to_trigger')
            dist = ('%.1f%%' % pct) if isinstance(pct, (int, float)) else '—'
            body += _row(r.get('symbol', ''), _sector_of(r.get('symbol', '')), dist, "to trigger", '#7A2430')

    body += _footer("RigaCap &middot; Maximizer &middot; Disciplined Momentum")
    body += '</td></tr>'
    return _shell(body)


def build_digest_v3(tier='preserver', signals=None, market_regime=None, watchlist=None,
                    market_context=None, regime_forecast=None, date=None,
                    breakout_book=None, breakout_radar=None, secondary_market_context=None,
                    **_ignore):
    """Dispatch to the tier render. Maximizer uses breakout_book + its own market read;
    Preserver uses signals bucketed by entry_status. Extra kwargs are ignored so the caller can
    pass the full generate_daily_summary_html param set."""
    if tier == 'maximizer' and breakout_book is not None:
        return _build_maximizer(breakout_book, market_regime, breakout_radar,
                                market_context, regime_forecast, date)
    return _build_preserver(signals, market_regime, watchlist, market_context, regime_forecast, date)
