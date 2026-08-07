"""
Chart Card Generator - Create 1080x1350 Instagram chart cards from trade data.

Uses matplotlib with Agg backend (headless, works on Lambda).
Cards show price chart with entry/exit markers, return info, and branding.
4:5 aspect ratio (1080x1350) optimized for Instagram's 3:4 grid preview.
"""

import io
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Use Agg backend for headless rendering (must be set before importing pyplot)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch


# Brand colors — editorial paper/ink/claret palette (NOT navy/gold — retired)
BRAND_GREEN = '#2D5F3F'
BRAND_RED = '#8F2D3D'
BRAND_DARK = '#141210'
BRAND_LIGHT = '#F5F1E8'
BRAND_ACCENT = '#7A2430'   # claret
BRAND_GRAY = '#5A544E'
BRAND_HAIRLINE = '#DDD5C7'

# Bundled brand fonts (Fraunces display + IBM Plex Sans/Mono) so Lambda-rendered cards
# match the site + launch cards instead of generic matplotlib sans. OFL-licensed.
from matplotlib import font_manager as _fm
_FONT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "fonts"))
def _fp(fname):
    path = os.path.join(_FONT_DIR, fname)
    try:
        _fm.fontManager.addfont(path)
        return _fm.FontProperties(fname=path)
    except Exception:
        return None
FRAUNCES = _fp("Fraunces.ttf")
PLEX = _fp("IBMPlexSans.ttf")
PLEX_MONO = _fp("IBMPlexMono.ttf")


class ChartCardGenerator:
    """Generate Instagram-ready 1080x1350 chart card images from trade data."""

    def __init__(self):
        self._s3_client = None

    def _get_s3_client(self):
        if self._s3_client is None:
            import boto3
            self._s3_client = boto3.client('s3')
        return self._s3_client

    def generate_trade_card(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        entry_date: str,
        exit_date: str,
        pnl_pct: float,
        pnl_dollars: float = 0,
        exit_reason: str = "trailing_stop",
        strategy_name: str = "Ensemble",
        regime_name: str = "",
        company_name: str = "",
    ) -> bytes:
        """
        Generate a 1080x1350 chart card image (4:5 ratio for Instagram).

        Returns PNG bytes.
        """
        is_win = pnl_pct > 0
        accent_color = BRAND_GREEN if is_win else BRAND_RED

        # Try to get price data for the chart
        price_data = self._get_price_data(symbol, entry_date, exit_date)

        # 1080x1350 = 4:5 ratio, optimized for Instagram grid preview
        fig, ax = plt.subplots(1, 1, figsize=(10.8, 13.5), dpi=100)
        fig.patch.set_facecolor(BRAND_LIGHT)

        # Layout regions (in figure coordinates, 0=bottom, 1=top)
        # Header: 0.95-1.00 (top 5%)
        # Symbol/company: 0.88-0.95 (7%)
        # Chart: 0.38-0.86 (48%)
        # Return: 0.25-0.36 (11%)
        # Details: 0.15-0.24 (9%)
        # Badges: 0.08-0.14 (6%)
        # Footer: 0.02-0.07 (5%)

        # Clear axes for custom layout
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # --- Header ---
        ax.text(0.05, 0.97, 'RigaCap', fontsize=22, fontweight='bold',
                color=BRAND_DARK, va='top', ha='left',
                fontfamily='sans-serif')
        if regime_name:
            ax.text(0.95, 0.97, f'Market: {regime_name}', fontsize=15,
                    color=BRAND_GRAY, va='top', ha='right',
                    fontfamily='sans-serif')

        # --- Gold divider line (below header) ---
        ax.plot([0.08, 0.92], [0.935, 0.935], color=BRAND_ACCENT, lw=1.5, alpha=0.6)

        # --- Symbol ---
        ax.text(0.5, 0.91, f'${symbol}', fontsize=46, fontweight='bold',
                color=BRAND_DARK, va='top', ha='center',
                fontfamily='sans-serif')
        if company_name:
            ax.text(0.5, 0.87, company_name, fontsize=17,
                    color=BRAND_GRAY, va='top', ha='center',
                    fontfamily='sans-serif')

        # --- Price Chart ---
        if price_data is not None and len(price_data) > 5:
            chart_ax = fig.add_axes([0.08, 0.38, 0.84, 0.46])
            self._draw_price_chart(
                chart_ax, price_data, entry_date, exit_date,
                entry_price, exit_price, accent_color
            )
        else:
            # No chart data — show placeholder
            ax.text(0.5, 0.60, 'Price Chart', fontsize=18,
                    color=BRAND_GRAY, va='center', ha='center',
                    fontfamily='sans-serif', style='italic')

        # --- Return ---
        return_sign = '+' if pnl_pct > 0 else ''
        ax.text(0.5, 0.30, f'{return_sign}{pnl_pct:.1f}%', fontsize=68,
                fontweight='bold', color=accent_color,
                va='center', ha='center', fontfamily='sans-serif')

        if pnl_dollars:
            dollar_sign = '+' if pnl_dollars > 0 else ''
            ax.text(0.5, 0.24, f'{dollar_sign}${abs(pnl_dollars):,.0f}',
                    fontsize=24, color=accent_color,
                    va='center', ha='center', fontfamily='sans-serif')

        # --- Gold divider line (above details) ---
        ax.plot([0.08, 0.92], [0.20, 0.20], color=BRAND_ACCENT, lw=1.5, alpha=0.6)

        # --- Details ---
        entry_display = entry_date[:10] if entry_date else ''
        exit_display = exit_date[:10] if exit_date else ''
        if entry_display and exit_display:
            days_held = self._calc_days(entry_date, exit_date)
            detail_text = f'Entry: {entry_display}  →  Exit: {exit_display}  ({days_held}d)'
        else:
            detail_text = ''

        ax.text(0.5, 0.16, detail_text, fontsize=16,
                color=BRAND_GRAY, va='center', ha='center',
                fontfamily='sans-serif')

        # --- Badges ---
        badge_y = 0.10
        # Strategy badge
        self._draw_badge(ax, 0.35, badge_y, strategy_name, BRAND_ACCENT)
        # Exit reason badge
        _EXIT_MAP = {
            "simulation_end": "Portfolio Rebalance",
            "rebalance_exit": "Portfolio Rebalance",
            "trailing_stop": "Trailing Stop",
            "market_regime": "Regime Shift",
        }
        exit_display = _EXIT_MAP.get(exit_reason, exit_reason.replace('_', ' ').title()) if exit_reason else 'Exit'
        self._draw_badge(ax, 0.65, badge_y, exit_display, BRAND_GRAY)

        # --- Footer ---
        ax.text(0.05, 0.03, 'rigacap.com', fontsize=14,
                color=BRAND_ACCENT, va='bottom', ha='left',
                fontfamily='sans-serif', fontweight='bold')
        ax.text(0.95, 0.03, 'Walk-Forward Verified', fontsize=14,
                color=BRAND_GRAY, va='bottom', ha='right',
                fontfamily='sans-serif', style='italic')

        # Render to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                    facecolor=BRAND_LIGHT, edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _gaussian_smooth(prices, sigma=1.5):
        """Smooth prices with a Gaussian kernel (numpy only, no scipy needed)."""
        kernel_size = int(sigma * 4) | 1  # ensure odd
        x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        # Pad edges to avoid shrinkage
        pad = kernel_size // 2
        padded = np.concatenate([
            np.full(pad, prices[0]),
            prices,
            np.full(pad, prices[-1]),
        ])
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed

    def _draw_price_chart(self, ax, price_data, entry_date, exit_date,
                          entry_price, exit_price, accent_color):
        """Draw the price line chart with entry/exit markers."""
        import pandas as pd

        dates = price_data.index
        prices = price_data['close'].values

        # Background
        ax.set_facecolor(BRAND_LIGHT)

        # Smooth the price line with Gaussian kernel for polished look
        try:
            smooth_prices = self._gaussian_smooth(prices, sigma=1.5)
            ax.plot(dates, smooth_prices, color=BRAND_ACCENT, linewidth=2.0, alpha=0.95)
        except Exception:
            ax.plot(dates, prices, color=BRAND_ACCENT, linewidth=2.0, alpha=0.95)

        # Find entry/exit indices
        entry_dt = pd.Timestamp(entry_date[:10])
        exit_dt = pd.Timestamp(exit_date[:10])

        # Shade between entry and exit
        mask = (dates >= entry_dt) & (dates <= exit_dt)
        if mask.any():
            fill_dates = dates[mask]
            try:
                fill_prices = smooth_prices[mask]
            except Exception:
                fill_prices = prices[mask]
            ax.fill_between(fill_dates, fill_prices,
                           min(prices) * 0.98,
                           alpha=0.15, color=accent_color)

        # Pad y-axis so labels aren't clipped at top/bottom
        price_range = max(prices) - min(prices)
        y_pad = price_range * 0.18
        ax.set_ylim(min(prices) - y_pad, max(prices) + y_pad)

        # Label background box style
        label_bbox = dict(boxstyle='round,pad=0.3', facecolor=BRAND_LIGHT,
                          edgecolor='#DDD5C7', alpha=0.9)

        # Entry marker (green triangle up) — label ABOVE the line
        entry_idx = np.argmin(np.abs((dates - entry_dt).total_seconds()))
        ax.scatter([dates[entry_idx]], [prices[entry_idx]],
                  color=BRAND_GREEN, marker='^', s=120, zorder=5)
        ax.annotate(f'${entry_price:.0f}', (dates[entry_idx], prices[entry_idx]),
                   textcoords="offset points", xytext=(0, 22),
                   fontsize=11, color=BRAND_GREEN, ha='center',
                   fontweight='bold', zorder=6, bbox=label_bbox)

        # Exit marker (red triangle down) — label BELOW the line
        exit_idx = np.argmin(np.abs((dates - exit_dt).total_seconds()))
        ax.scatter([dates[exit_idx]], [prices[exit_idx]],
                  color=BRAND_RED, marker='v', s=120, zorder=5)
        ax.annotate(f'${exit_price:.0f}', (dates[exit_idx], prices[exit_idx]),
                   textcoords="offset points", xytext=(0, -26),
                   fontsize=11, color=BRAND_RED, ha='center',
                   fontweight='bold', zorder=6, bbox=label_bbox)

        # Formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(colors=BRAND_GRAY, labelsize=9)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis='y', alpha=0.5, color='#DDD5C7')

    def _draw_badge(self, ax, x, y, text, color):
        """Draw a rounded badge at the given position."""
        bbox = dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.3,
                    edgecolor=color, linewidth=1)
        ax.text(x, y, text, fontsize=12, color=BRAND_DARK,
                va='center', ha='center', fontfamily='sans-serif',
                bbox=bbox)

    def _get_price_data(self, symbol: str, entry_date: str, exit_date: str):
        """Get price data from scanner cache for chart rendering."""
        try:
            from app.services.scanner import scanner_service
            import pandas as pd

            df = scanner_service.data_cache.get(symbol)
            if df is None or df.empty:
                return None

            # Get 60-day window centered on the trade
            entry_dt = pd.Timestamp(entry_date[:10])
            buffer = timedelta(days=30)
            start = entry_dt - buffer
            end = pd.Timestamp(exit_date[:10]) + buffer

            mask = (df.index >= start) & (df.index <= end)
            subset = df.loc[mask]

            return subset if len(subset) > 5 else None
        except Exception:
            return None

    def _calc_days(self, entry_date: str, exit_date: str) -> int:
        """Calculate days between two date strings."""
        try:
            d1 = datetime.strptime(entry_date[:10], '%Y-%m-%d')
            d2 = datetime.strptime(exit_date[:10], '%Y-%m-%d')
            return (d2 - d1).days
        except Exception:
            return 0

    def upload_to_s3(self, png_bytes: bytes, post_id: int,
                     symbol: str, date_str: str) -> str:
        """Upload chart card PNG to S3. Returns the S3 key."""
        bucket = os.environ.get("PRICE_DATA_BUCKET", "rigacap-prod-price-data-149218244179")
        key = f"social/images/{post_id}_{symbol}_{date_str}.png"

        try:
            s3 = self._get_s3_client()
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=png_bytes,
                ContentType='image/png',
            )
            logger.info(f"Uploaded chart card to s3://{bucket}/{key}")
            return key
        except Exception as e:
            logger.error(f"Failed to upload chart card: {e}")
            return ""

    def get_presigned_url(self, s3_key: str, expires_in: int = 3600) -> str:
        """Get a presigned URL for an S3 chart card image."""
        if not s3_key:
            return ""

        bucket = os.environ.get("PRICE_DATA_BUCKET", "rigacap-prod-price-data-149218244179")
        try:
            s3 = self._get_s3_client()
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': s3_key},
                ExpiresIn=expires_in,
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return ""

    def generate_text_card(
        self,
        text: str,
        headline: str = "",
    ) -> bytes:
        """
        Generate a 1080x1350 editorial QUOTE card for Instagram text posts (insights,
        we_called_it, regime notes). Claret/paper brand, Fraunces pull-quote, hairline
        frame — matches the launch cards. Returns PNG bytes.
        """
        import textwrap

        import re as _re
        serif = FRAUNCES or _fm.FontProperties(family="serif")
        sans = PLEX or _fm.FontProperties(family="sans-serif")
        mono = PLEX_MONO or _fm.FontProperties(family="monospace")

        W_PX, H_PX = 1080, 1350
        fig, ax = plt.subplots(1, 1, figsize=(10.8, 13.5), dpi=100)
        fig.patch.set_facecolor(BRAND_LIGHT)
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        renderer = fig.canvas.get_renderer()

        def _sized(fp, size):
            f = fp.copy()
            f.set_size(size)
            return f

        def _wpx(s, fp):
            return renderer.get_text_width_height_descent(s, fp, False)[0]

        # Hairline print frame (editorial border, like the launch cards)
        ax.add_patch(plt.Rectangle((0.035, 0.028), 0.93, 0.944, fill=False,
                                   edgecolor=BRAND_HAIRLINE, lw=1.4))

        # Wordmark "RigaCap." — Fraunces, centered as ONE unit, with a claret period.
        serif_wm = _sized(serif, 30)
        wm_y = 0.905
        ax.text(0.5, wm_y, 'RigaCap.', color=BRAND_DARK,
                va='center', ha='center', fontproperties=serif_wm)
        right_x = 0.5 + (_wpx('RigaCap.', serif_wm) / 2.0) / W_PX  # right edge of centered string
        ax.text(right_x, wm_y, '.', color=BRAND_ACCENT,
                va='center', ha='right', fontproperties=serif_wm)  # claret period overlays the ink one

        # claret kicker under the wordmark — same length as the closing rule below
        KICK = (0.41, 0.59)
        ax.plot(list(KICK), [0.86, 0.86], color=BRAND_ACCENT, lw=2.4)

        # --- Body: break by SENTENCE, wrap by MEASURED pixel width (proportional font),
        #     and pull 1-2 word widows up into the prior line when they fit (guarded so a
        #     merged line never overruns the frame). IBM Plex Sans (site body font). ---
        clean = (text.split('http')[0]).strip()
        clean = clean.split('#')[0].strip() if '#' in clean else clean
        sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', clean) if s.strip()]

        COMFORTABLE = 0.76 * W_PX   # normal max line width
        HARD = 0.86 * W_PX          # absolute max when pulling a widow up (still inside frame)

        # Function words that shouldn't be stranded at the END of a line — an article/
        # preposition/conjunction belongs with the word that follows it ("...the weeks /
        # the system..." not "...the weeks the / system..."). Break BEFORE them instead.
        FUNCTION_WORDS = {
            "a", "an", "the", "and", "or", "but", "nor", "for", "yet", "so",
            "of", "to", "in", "on", "at", "by", "as", "with", "from", "into",
            "than", "then", "is", "are", "was", "were", "be", "your", "our",
            "its", "their", "his", "her", "my", "that", "this", "these", "those",
        }

        def _is_fn(w):
            return w.strip(".,;:!?—-\"'()").lower() in FUNCTION_WORDS

        def _wrap(sentence, fp):
            words = sentence.split()
            lines, cur = [], []
            for w in words:
                trial = " ".join(cur + [w])
                if not cur or _wpx(trial, fp) <= COMFORTABLE:
                    cur.append(w)
                else:
                    # Don't end a line on a function word — carry any trailing ones down
                    # with the new word so "the" stays attached to its noun.
                    carry = []
                    while len(cur) > 1 and _is_fn(cur[-1]):
                        carry.insert(0, cur.pop())
                    lines.append(" ".join(cur))
                    cur = carry + [w]
            if cur:
                lines.append(" ".join(cur))
            # widow pull-up: absorb a trailing 1-2 word line into the previous one if it fits
            while len(lines) >= 2 and len(lines[-1].split()) <= 2:
                merged = lines[-2] + " " + lines[-1]
                if _wpx(merged, fp) <= HARD:
                    lines[-2] = merged
                    lines.pop()
                else:
                    break
            return lines

        # Fixed "box" between the two claret rules; text is centered WITHIN it (so short
        # quotes sit in the middle of the box, not pinned to the bottom rule).
        TOP_RULE_Y, BOTTOM_RULE_Y = 0.86, 0.205
        band_top, band_bottom = 0.815, 0.25   # small insets from the rules, center = 0.5325
        band_h = band_top - band_bottom
        chosen = None
        for fs in (32, 30, 28, 26, 24, 22, 20, 18):
            fp = _sized(sans, fs)
            blocks = [_wrap(s, fp) for s in sentences]
            n_lines = sum(len(b) for b in blocks)
            line_step = (fs * (100.0 / 72.0) * 1.42) / H_PX   # line height in axes fraction
            gap = line_step * 0.62                            # space between sentences
            total = n_lines * line_step + max(0, len(blocks) - 1) * gap
            if total <= band_h:
                chosen = (fp, fs, blocks, line_step, gap, total)
                break
        if not chosen:
            fp = _sized(sans, 18)
            blocks = [_wrap(s, fp) for s in sentences]
            line_step = (18 * (100.0 / 72.0) * 1.42) / H_PX
            gap = line_step * 0.62
            total = sum(len(b) for b in blocks) * line_step + max(0, len(blocks) - 1) * gap
            chosen = (fp, 18, blocks, line_step, gap, total)
        fp, fs, blocks, line_step, gap, total = chosen

        cursor = band_bottom + (band_h + total) / 2  # center the block between the rules
        for block in blocks:
            for ln in block:
                ax.text(0.5, cursor, ln, fontsize=fs, color=BRAND_DARK,
                        va='top', ha='center', fontproperties=fp)
                cursor -= line_step
            cursor -= gap

        # closing claret kicker — FIXED position (defines the bottom of the box; text is
        # centered within it above, rather than the rule hugging the text)
        ax.plot(list(KICK), [BOTTOM_RULE_Y, BOTTOM_RULE_Y], color=BRAND_ACCENT, lw=2.4)

        # --- Footer: walk-forward note + url ---
        ax.plot([0.08, 0.92], [0.115, 0.115], color=BRAND_HAIRLINE, lw=1.2)
        ax.text(0.5, 0.075, 'Walk-forward tested — signals only, you execute.',
                fontsize=15, color=BRAND_GRAY, va='center', ha='center', fontproperties=sans)
        ax.text(0.5, 0.045, 'rigacap.com', fontsize=17, color=BRAND_ACCENT,
                va='center', ha='center', fontproperties=mono)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=BRAND_LIGHT, edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def generate_track_record_chart(
        self,
        equity_curve: list,
        regime_periods: list,
        total_return_pct: float = 289,
        benchmark_return_pct: float = 95,
    ) -> bytes:
        """
        Generate a landscape track record chart for PDF documents.
        Shows portfolio vs SPY equity curves with regime bands.
        Returns SVG bytes (vector — crisp at any resolution).
        """
        from matplotlib.ticker import FuncFormatter

        dates = [datetime.strptime(p["date"], "%Y-%m-%d") for p in equity_curve]
        equities = [p["equity"] for p in equity_curve]
        spy_equities = [p.get("spy_equity", 100000) for p in equity_curve]

        BG_COLOR = '#F5F1E8'
        TEXT_COLOR = '#141210'
        TEXT_MUTED = '#5A544E'
        GRID_COLOR = '#DDD5C7'
        SPINE_COLOR = '#C4BAA9'
        PORTFOLIO_COLOR = '#7A2430'

        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        # Regime bands — higher alpha for visibility on light background
        regime_colors = {
            'strong_bull': ((0.06, 0.73, 0.51, 0.18), '#059669'),
            'weak_bull': ((0.52, 0.80, 0.09, 0.15), '#65A30D'),
            'rotating_bull': ((0.55, 0.36, 0.96, 0.15), '#7C3AED'),
            'range_bound': ((0.96, 0.62, 0.04, 0.15), '#D97706'),
            'weak_bear': ((0.98, 0.45, 0.09, 0.18), '#EA580C'),
            'panic_crash': ((0.94, 0.27, 0.27, 0.20), '#DC2626'),
            'recovery': ((0.02, 0.71, 0.83, 0.15), '#0891B2'),
        }

        legend_regimes = {}
        for period in regime_periods:
            try:
                start = datetime.strptime(period["start_date"], "%Y-%m-%d")
                end = datetime.strptime(period["end_date"], "%Y-%m-%d")
                rtype = period.get("regime_type", "range_bound")
                bg_color, line_color = regime_colors.get(rtype, ((0.78, 0.78, 0.78, 0.1), '#6B7280'))
                ax.axvspan(start, end, facecolor=bg_color, edgecolor='none')
                if rtype not in legend_regimes:
                    legend_regimes[rtype] = {
                        'name': period.get("regime_name", rtype),
                        'color': line_color,
                        'bg': bg_color,
                    }
            except (ValueError, KeyError):
                continue

        # Plot lines
        ax.plot(dates, equities, color=PORTFOLIO_COLOR, linewidth=2.5, label='RigaCap Ensemble', zorder=5)
        ax.plot(dates, spy_equities, color='#94A3B8', linewidth=1.5, linestyle='--', label='S&P 500 (SPY)', zorder=4)

        # Formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))

        ax.tick_params(colors=TEXT_MUTED, labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(SPINE_COLOR)
        ax.spines['left'].set_color(SPINE_COLOR)
        ax.grid(axis='y', color=GRID_COLOR, linewidth=0.5)

        # Metrics overlay (top-left)
        ax.text(
            0.02, 0.95, f'+{total_return_pct:.0f}% Total Return',
            transform=ax.transAxes, fontsize=18, fontweight='bold',
            color=PORTFOLIO_COLOR, va='top', ha='left',
        )
        ax.text(
            0.02, 0.87, f'vs SPY +{benchmark_return_pct:.0f}%',
            transform=ax.transAxes, fontsize=11,
            color=TEXT_MUTED, va='top', ha='left',
        )

        # Line legend (top-right)
        ax.legend(
            loc='upper right', fontsize=9,
            facecolor=BG_COLOR, edgecolor=SPINE_COLOR,
            labelcolor=TEXT_COLOR,
        )

        # Watermark
        ax.text(
            0.98, 0.03, 'rigacap.com',
            transform=ax.transAxes, fontsize=8, color='#94A3B8',
            va='bottom', ha='right', style='italic',
        )

        # Regime legend at bottom
        if legend_regimes:
            regime_labels = []
            regime_handles = []
            for rtype, info in legend_regimes.items():
                patch = plt.Rectangle((0, 0), 1, 1, facecolor=info['bg'], edgecolor=info['color'], linewidth=1)
                regime_handles.append(patch)
                regime_labels.append(info['name'])

            fig.legend(
                regime_handles, regime_labels,
                loc='lower center', ncol=min(len(regime_labels), 7),
                fontsize=8, facecolor=BG_COLOR, edgecolor='none',
                labelcolor=TEXT_MUTED, framealpha=0,
                bbox_to_anchor=(0.5, 0.01),
            )

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        buf = io.BytesIO()
        fig.savefig(buf, format='svg', facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def upload_track_record_chart(self, svg_bytes: bytes) -> str:
        """Upload track record chart SVG to S3. Returns the S3 key."""
        bucket = os.environ.get("PRICE_DATA_BUCKET", "rigacap-prod-price-data-149218244179")
        key = "charts/track-record-5yr.svg"

        try:
            s3 = self._get_s3_client()
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=svg_bytes,
                ContentType='image/svg+xml',
            )
            logger.info(f"Uploaded track record chart to s3://{bucket}/{key}")
            return key
        except Exception as e:
            logger.error(f"Failed to upload track record chart: {e}")
            return ""


# Singleton instance
chart_card_generator = ChartCardGenerator()
