"""
Email Service - Daily summary emails for subscribers

Sends beautiful HTML emails with:
- Top signals of the day
- Market regime summary
- Open positions P&L
- Missed opportunities
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

logger = logging.getLogger(__name__)

# Email configuration from environment
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASS = os.getenv('SMTP_PASS', '')
FROM_EMAIL = os.getenv('FROM_EMAIL', 'signals@rigacap.com')
FROM_NAME = os.getenv('FROM_NAME', 'RigaCap Signals')


class EmailService:
    """
    Manages email sending for daily summaries and alerts
    """

    def __init__(self):
        self.enabled = bool(SMTP_USER and SMTP_PASS)
        if not self.enabled:
            logger.warning("Email service disabled - SMTP credentials not configured")

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """
        Send an email to a single recipient

        Args:
            to_email: Recipient email address
            subject: Email subject line
            html_content: HTML body of the email
            text_content: Plain text fallback (optional)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning(f"Email service disabled, would have sent to: {to_email}")
            return False

        try:
            import aiosmtplib

            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{FROM_NAME} <{FROM_EMAIL}>"
            msg['To'] = to_email

            # Add plain text part
            if text_content:
                text_part = MIMEText(text_content, 'plain', 'utf-8')
                msg.attach(text_part)

            # Add HTML part
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)

            # Send via SMTP
            await aiosmtplib.send(
                msg,
                hostname=SMTP_HOST,
                port=SMTP_PORT,
                username=SMTP_USER,
                password=SMTP_PASS,
                start_tls=True
            )

            logger.info(f"Email sent to {to_email}: {subject}")
            return True

        except ImportError:
            logger.error("aiosmtplib not installed. Run: pip install aiosmtplib")
            return False
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    def generate_daily_summary_html(
        self,
        signals: List[Dict],
        market_regime: Dict,
        positions: List[Dict],
        missed_opportunities: List[Dict],
        date: Optional[datetime] = None
    ) -> str:
        """
        Generate beautiful HTML for daily summary email

        Args:
            signals: List of today's signals
            market_regime: Market regime info (regime, spy_price, vix_level)
            positions: User's open positions with P&L
            missed_opportunities: Recent missed opportunities
            date: Date for the summary (default: today)

        Returns:
            HTML string for email body
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%A, %B %d, %Y")
        strong_signals = [s for s in signals if s.get('is_strong')]

        # Calculate totals
        total_positions_pnl = sum(
            (p.get('current_price', 0) - p.get('entry_price', 0)) * p.get('shares', 0)
            for p in positions
        )
        total_missed = sum(m.get('would_be_pnl', 0) for m in missed_opportunities[:5])

        # Regime styling
        regime = market_regime.get('regime', 'neutral') if market_regime else 'neutral'
        regime_colors = {
            'strong_bull': ('#059669', '#d1fae5', 'Strong Bull'),
            'bull': ('#10b981', '#ecfdf5', 'Bull Market'),
            'neutral': ('#6b7280', '#f3f4f6', 'Neutral'),
            'bear': ('#f59e0b', '#fef3c7', 'Bear Market'),
            'strong_bear': ('#dc2626', '#fee2e2', 'Strong Bear')
        }
        regime_color, regime_bg, regime_label = regime_colors.get(regime, regime_colors['neutral'])

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f3f4f6;">
    <table cellpadding="0" cellspacing="0" style="width: 100%; max-width: 600px; margin: 0 auto; background-color: #ffffff;">
        <!-- Header -->
        <tr>
            <td style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); padding: 32px 24px; text-align: center;">
                <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">
                    📈 RigaCap Daily
                </h1>
                <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.9); font-size: 14px;">
                    {date_str}
                </p>
            </td>
        </tr>

        <!-- Market Summary -->
        <tr>
            <td style="padding: 24px;">
                <table cellpadding="0" cellspacing="0" style="width: 100%;">
                    <tr>
                        <td style="background-color: {regime_bg}; border-radius: 12px; padding: 20px;">
                            <div style="font-size: 12px; text-transform: uppercase; color: #6b7280; font-weight: 600; margin-bottom: 8px;">
                                Market Regime
                            </div>
                            <div style="font-size: 24px; font-weight: 700; color: {regime_color};">
                                {regime_label}
                            </div>
                            <div style="margin-top: 12px; font-size: 14px; color: #374151;">
                                SPY: ${market_regime.get('spy_price', 'N/A') if market_regime else 'N/A'} &nbsp;•&nbsp;
                                VIX: {market_regime.get('vix_level', 'N/A') if market_regime else 'N/A'}
                            </div>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>

        <!-- Signals Section -->
        <tr>
            <td style="padding: 0 24px 24px;">
                <h2 style="margin: 0 0 16px 0; font-size: 18px; color: #111827;">
                    🎯 Today's Signals ({len(strong_signals)} Strong)
                </h2>
                {"".join(self._signal_row(s) for s in signals[:8]) if signals else '''
                <div style="background-color: #f9fafb; border-radius: 8px; padding: 24px; text-align: center; color: #6b7280;">
                    No signals found today. Check back tomorrow!
                </div>
                '''}
            </td>
        </tr>

        <!-- Open Positions -->
        {self._positions_section(positions, total_positions_pnl) if positions else ''}

        <!-- Missed Opportunities -->
        {self._missed_section(missed_opportunities[:5], total_missed) if missed_opportunities else ''}

        <!-- Footer -->
        <tr>
            <td style="background-color: #f9fafb; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
                <p style="margin: 0 0 8px 0; font-size: 14px; color: #6b7280;">
                    <a href="#" style="color: #4f46e5; text-decoration: none;">View Dashboard</a>
                    &nbsp;•&nbsp;
                    <a href="#" style="color: #4f46e5; text-decoration: none;">Manage Alerts</a>
                    &nbsp;•&nbsp;
                    <a href="#" style="color: #6b7280; text-decoration: none;">Unsubscribe</a>
                </p>
                <p style="margin: 0; font-size: 12px; color: #9ca3af;">
                    Trading involves risk. Past performance does not guarantee future results.
                </p>
                <p style="margin: 8px 0 0 0; font-size: 12px; color: #9ca3af;">
                    &copy; {date.year} RigaCap. All rights reserved.
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
"""
        return html

    def _signal_row(self, signal: Dict) -> str:
        """Generate HTML for a single signal row"""
        symbol = signal.get('symbol', 'N/A')
        price = signal.get('price', 0)
        pct_above = signal.get('pct_above_dwap', 0)
        strength = signal.get('signal_strength', 0)
        is_strong = signal.get('is_strong', False)
        sector = signal.get('sector', '')

        strength_color = '#059669' if strength >= 70 else '#f59e0b' if strength >= 50 else '#6b7280'
        badge = '🔥' if is_strong else '📊'

        return f"""
        <div style="background-color: #f9fafb; border-radius: 8px; padding: 16px; margin-bottom: 8px;">
            <table cellpadding="0" cellspacing="0" style="width: 100%;">
                <tr>
                    <td>
                        <div style="font-size: 16px; font-weight: 600; color: #111827;">
                            {badge} {symbol}
                            {f'<span style="font-size: 12px; font-weight: 400; color: #6b7280; margin-left: 8px;">{sector}</span>' if sector else ''}
                        </div>
                        <div style="font-size: 14px; color: #6b7280; margin-top: 4px;">
                            ${price:.2f} &nbsp;•&nbsp; +{pct_above:.1f}% above DWAP
                        </div>
                    </td>
                    <td style="text-align: right;">
                        <div style="display: inline-block; background-color: {strength_color}; color: #ffffff; font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 99px;">
                            {strength:.0f}
                        </div>
                    </td>
                </tr>
            </table>
        </div>
        """

    def _positions_section(self, positions: List[Dict], total_pnl: float) -> str:
        """Generate HTML for positions section"""
        pnl_color = '#059669' if total_pnl >= 0 else '#dc2626'
        pnl_sign = '+' if total_pnl >= 0 else ''

        rows = ""
        for p in positions[:5]:
            symbol = p.get('symbol', 'N/A')
            shares = p.get('shares', 0)
            entry = p.get('entry_price', 0)
            current = p.get('current_price', entry)
            pnl = (current - entry) * shares
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            color = '#059669' if pnl >= 0 else '#dc2626'
            sign = '+' if pnl >= 0 else ''

            rows += f"""
            <tr>
                <td style="padding: 8px 0; border-bottom: 1px solid #e5e7eb;">
                    <span style="font-weight: 600;">{symbol}</span>
                    <span style="color: #6b7280; font-size: 12px;"> ({shares} shares)</span>
                </td>
                <td style="padding: 8px 0; text-align: right; border-bottom: 1px solid #e5e7eb;">
                    <span style="color: {color}; font-weight: 600;">{sign}${abs(pnl):.0f}</span>
                    <span style="color: {color}; font-size: 12px;"> ({sign}{pnl_pct:.1f}%)</span>
                </td>
            </tr>
            """

        return f"""
        <tr>
            <td style="padding: 0 24px 24px;">
                <h2 style="margin: 0 0 16px 0; font-size: 18px; color: #111827;">
                    💼 Open Positions
                </h2>
                <table cellpadding="0" cellspacing="0" style="width: 100%;">
                    {rows}
                    <tr>
                        <td style="padding: 12px 0 0 0; font-weight: 600;">Total P&L</td>
                        <td style="padding: 12px 0 0 0; text-align: right; font-weight: 700; font-size: 18px; color: {pnl_color};">
                            {pnl_sign}${abs(total_pnl):,.0f}
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
        """

    def _missed_section(self, missed: List[Dict], total_missed: float) -> str:
        """Generate HTML for missed opportunities section"""
        rows = ""
        for m in missed:
            symbol = m.get('symbol', 'N/A')
            would_be = m.get('would_be_return', 0)
            would_be_pnl = m.get('would_be_pnl', 0)
            date = m.get('signal_date', '')

            rows += f"""
            <tr>
                <td style="padding: 8px 0; border-bottom: 1px solid #fef3c7;">
                    <span style="font-weight: 600;">{symbol}</span>
                    <span style="color: #92400e; font-size: 12px;"> ({date})</span>
                </td>
                <td style="padding: 8px 0; text-align: right; border-bottom: 1px solid #fef3c7; color: #b45309;">
                    +{would_be:.1f}% (+${would_be_pnl:.0f})
                </td>
            </tr>
            """

        return f"""
        <tr>
            <td style="padding: 0 24px 24px;">
                <div style="background-color: #fef3c7; border-radius: 12px; padding: 20px;">
                    <h2 style="margin: 0 0 12px 0; font-size: 16px; color: #92400e;">
                        😮 Missed Opportunities
                    </h2>
                    <table cellpadding="0" cellspacing="0" style="width: 100%;">
                        {rows}
                    </table>
                    <div style="margin-top: 12px; font-size: 14px; color: #b45309; font-weight: 600;">
                        Total missed: +${total_missed:,.0f}
                    </div>
                </div>
            </td>
        </tr>
        """

    def generate_plain_text(
        self,
        signals: List[Dict],
        market_regime: Dict,
        date: Optional[datetime] = None
    ) -> str:
        """Generate plain text fallback for email"""
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%A, %B %d, %Y")
        strong_signals = [s for s in signals if s.get('is_strong')]

        lines = [
            f"STOCKER DAILY - {date_str}",
            "=" * 40,
            "",
            f"Market Regime: {market_regime.get('regime', 'N/A') if market_regime else 'N/A'}",
            f"SPY: ${market_regime.get('spy_price', 'N/A') if market_regime else 'N/A'}",
            f"VIX: {market_regime.get('vix_level', 'N/A') if market_regime else 'N/A'}",
            "",
            f"TODAY'S SIGNALS ({len(strong_signals)} Strong)",
            "-" * 40,
        ]

        for s in signals[:10]:
            symbol = s.get('symbol', 'N/A')
            price = s.get('price', 0)
            pct = s.get('pct_above_dwap', 0)
            strength = s.get('signal_strength', 0)
            marker = "***" if s.get('is_strong') else ""
            lines.append(f"{marker}{symbol}: ${price:.2f} (+{pct:.1f}% DWAP) - Strength: {strength:.0f}")

        lines.extend([
            "",
            "View full details at: https://rigacap.com/dashboard",
            "",
            "---",
            "RigaCap - DWAP Trading Signals",
            "Trading involves risk. Past performance does not guarantee future results."
        ])

        return "\n".join(lines)

    async def send_daily_summary(
        self,
        to_email: str,
        signals: List[Dict],
        market_regime: Dict,
        positions: List[Dict] = None,
        missed_opportunities: List[Dict] = None
    ) -> bool:
        """
        Send daily summary email to a subscriber

        Args:
            to_email: Subscriber email
            signals: Today's signals
            market_regime: Current market regime info
            positions: User's open positions
            missed_opportunities: Recent missed opportunities

        Returns:
            True if sent successfully
        """
        positions = positions or []
        missed_opportunities = missed_opportunities or []

        strong_count = len([s for s in signals if s.get('is_strong')])
        subject = f"📊 RigaCap Daily: {strong_count} Strong Signals Found"

        html = self.generate_daily_summary_html(
            signals=signals,
            market_regime=market_regime,
            positions=positions,
            missed_opportunities=missed_opportunities
        )

        text = self.generate_plain_text(signals, market_regime)

        return await self.send_email(to_email, subject, html, text)

    async def send_bulk_daily_summary(
        self,
        subscribers: List[str],
        signals: List[Dict],
        market_regime: Dict
    ) -> Dict:
        """
        Send daily summary to all subscribers

        Args:
            subscribers: List of subscriber emails
            signals: Today's signals
            market_regime: Current market regime

        Returns:
            Summary of sent/failed emails
        """
        sent = 0
        failed = 0

        for email in subscribers:
            try:
                # TODO: Fetch user-specific positions and missed opportunities
                success = await self.send_daily_summary(
                    to_email=email,
                    signals=signals,
                    market_regime=market_regime
                )
                if success:
                    sent += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to send to {email}: {e}")
                failed += 1

            # Rate limiting - don't spam SMTP server
            await asyncio.sleep(0.5)

        logger.info(f"Bulk email complete: {sent} sent, {failed} failed")
        return {"sent": sent, "failed": failed, "total": len(subscribers)}


    async def send_welcome_email(self, to_email: str, name: str) -> bool:
        """
        Send a beautiful welcome email when a user signs up.
        """
        first_name = name.split()[0] if name else "there"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f3f4f6;">
    <table cellpadding="0" cellspacing="0" style="width: 100%; max-width: 600px; margin: 0 auto; background-color: #ffffff;">
        <!-- Header -->
        <tr>
            <td style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); padding: 48px 24px; text-align: center;">
                <div style="font-size: 48px; margin-bottom: 16px;">🚀</div>
                <h1 style="margin: 0; color: #ffffff; font-size: 32px; font-weight: 700;">
                    Welcome to RigaCap!
                </h1>
                <p style="margin: 12px 0 0 0; color: rgba(255,255,255,0.9); font-size: 18px;">
                    Your journey to smarter trading starts now
                </p>
            </td>
        </tr>

        <!-- Welcome Message -->
        <tr>
            <td style="padding: 40px 32px;">
                <p style="font-size: 18px; color: #374151; margin: 0 0 24px 0; line-height: 1.6;">
                    Hey {first_name}! 👋
                </p>
                <p style="font-size: 16px; color: #374151; margin: 0 0 24px 0; line-height: 1.6;">
                    Thank you for joining RigaCap! We're thrilled to have you on board.
                    You've just unlocked access to our powerful DWAP trading signals that
                    have helped traders achieve <strong>216% returns</strong> in backtesting.
                </p>

                <!-- What You Get Box -->
                <div style="background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); border-radius: 16px; padding: 24px; margin: 24px 0;">
                    <h2 style="margin: 0 0 16px 0; font-size: 18px; color: #059669;">
                        ✨ Here's what you get:
                    </h2>
                    <ul style="margin: 0; padding: 0 0 0 20px; color: #374151; line-height: 2;">
                        <li><strong>Daily AI-powered signals</strong> — Know exactly when to buy</li>
                        <li><strong>4,500+ stocks scanned</strong> — Never miss an opportunity</li>
                        <li><strong>Stop-loss & profit targets</strong> — Manage risk automatically</li>
                        <li><strong>Market regime analysis</strong> — Bull or bear, we've got you</li>
                        <li><strong>Daily email digest</strong> — Signals delivered to your inbox</li>
                        <li><strong>Portfolio tracking</strong> — See your P&L in real-time</li>
                    </ul>
                </div>

                <p style="font-size: 16px; color: #374151; margin: 24px 0; line-height: 1.6;">
                    Your <strong>7-day free trial</strong> starts now. Explore the dashboard,
                    check out today's signals, and see the DWAP algorithm in action!
                </p>

                <!-- CTA Button -->
                <div style="text-align: center; margin: 32px 0;">
                    <a href="https://rigacap.com/app"
                       style="display: inline-block; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #ffffff; font-size: 16px; font-weight: 600; padding: 16px 40px; border-radius: 12px; text-decoration: none;">
                        View Today's Signals →
                    </a>
                </div>

                <!-- Pro Tip -->
                <div style="background-color: #fef3c7; border-radius: 12px; padding: 20px; margin: 24px 0;">
                    <p style="margin: 0; font-size: 14px; color: #92400e;">
                        <strong>💡 Pro Tip:</strong> For the best results, focus on signals with
                        a strength score above 70. These "Strong Signals" have the highest
                        probability of success!
                    </p>
                </div>

                <p style="font-size: 16px; color: #374151; margin: 24px 0 0 0; line-height: 1.6;">
                    If you have any questions, just reply to this email — we're always here to help.
                </p>

                <p style="font-size: 16px; color: #374151; margin: 24px 0 0 0; line-height: 1.6;">
                    Happy trading! 📈<br>
                    <strong>The RigaCap Team</strong>
                </p>
            </td>
        </tr>

        <!-- Footer -->
        <tr>
            <td style="background-color: #f9fafb; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
                <p style="margin: 0; font-size: 12px; color: #9ca3af;">
                    Trading involves risk. Past performance does not guarantee future results.
                </p>
                <p style="margin: 8px 0 0 0; font-size: 12px; color: #9ca3af;">
                    &copy; {datetime.now().year} RigaCap. All rights reserved.
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
"""

        text = f"""
Welcome to RigaCap, {first_name}!

Your journey to smarter trading starts now.

Here's what you get:
- Daily AI-powered signals
- 4,500+ stocks scanned daily
- Stop-loss & profit targets
- Market regime analysis
- Daily email digest
- Portfolio tracking

Your 7-day free trial starts now. Visit https://rigacap.com/app to see today's signals!

Pro Tip: Focus on signals with strength above 70 for the best results.

Happy trading!
The RigaCap Team

---
Trading involves risk. Past performance does not guarantee future results.
"""

        return await self.send_email(
            to_email=to_email,
            subject="🚀 Welcome to RigaCap — Your Trading Edge Starts Now!",
            html_content=html,
            text_content=text
        )

    async def send_goodbye_email(self, to_email: str, name: str) -> bool:
        """
        Send a 'sorry to see you go' email when a user cancels or trial expires.
        """
        first_name = name.split()[0] if name else "there"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f3f4f6;">
    <table cellpadding="0" cellspacing="0" style="width: 100%; max-width: 600px; margin: 0 auto; background-color: #ffffff;">
        <!-- Header -->
        <tr>
            <td style="background: linear-gradient(135deg, #1f2937 0%, #374151 100%); padding: 48px 24px; text-align: center;">
                <div style="font-size: 48px; margin-bottom: 16px;">💔</div>
                <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">
                    We're Sad to See You Go
                </h1>
            </td>
        </tr>

        <!-- Message -->
        <tr>
            <td style="padding: 40px 32px;">
                <p style="font-size: 18px; color: #374151; margin: 0 0 24px 0; line-height: 1.6;">
                    Hey {first_name},
                </p>
                <p style="font-size: 16px; color: #374151; margin: 0 0 24px 0; line-height: 1.6;">
                    We noticed your RigaCap subscription has ended. We're truly sorry to see you go!
                    Before you leave completely, we wanted to share what you might be missing...
                </p>

                <!-- What You're Missing -->
                <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-radius: 16px; padding: 24px; margin: 24px 0;">
                    <h2 style="margin: 0 0 16px 0; font-size: 18px; color: #dc2626;">
                        📉 What You're Missing Today:
                    </h2>
                    <ul style="margin: 0; padding: 0 0 0 20px; color: #374151; line-height: 2;">
                        <li>Fresh daily signals from 4,500+ stocks</li>
                        <li>Real-time market regime updates</li>
                        <li>Buy signals before they surge</li>
                        <li>Stop-loss alerts to protect your capital</li>
                    </ul>
                </div>

                <!-- Stats -->
                <div style="background-color: #f0fdf4; border-radius: 16px; padding: 24px; margin: 24px 0; text-align: center;">
                    <p style="margin: 0; font-size: 14px; color: #059669; text-transform: uppercase; font-weight: 600;">
                        Our Backtest Performance
                    </p>
                    <p style="margin: 8px 0 0 0; font-size: 48px; font-weight: 700; color: #059669;">
                        216%
                    </p>
                    <p style="margin: 4px 0 0 0; font-size: 14px; color: #374151;">
                        Total return over the test period
                    </p>
                </div>

                <p style="font-size: 16px; color: #374151; margin: 24px 0; line-height: 1.6;">
                    We'd love to have you back. If you left because something wasn't working,
                    please reply to this email and let us know — we're always improving based on feedback.
                </p>

                <!-- Special Offer -->
                <div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); border-radius: 16px; padding: 24px; margin: 24px 0; text-align: center;">
                    <p style="margin: 0 0 8px 0; font-size: 14px; color: rgba(255,255,255,0.9); text-transform: uppercase; font-weight: 600;">
                        Come Back Offer
                    </p>
                    <p style="margin: 0 0 16px 0; font-size: 24px; font-weight: 700; color: #ffffff;">
                        Get 20% Off Your First Month
                    </p>
                    <a href="https://rigacap.com/app?promo=COMEBACK20"
                       style="display: inline-block; background-color: #ffffff; color: #4f46e5; font-size: 16px; font-weight: 600; padding: 14px 32px; border-radius: 10px; text-decoration: none;">
                        Reactivate Now →
                    </a>
                </div>

                <p style="font-size: 16px; color: #374151; margin: 24px 0; line-height: 1.6;">
                    Whatever you decide, we wish you the best with your trading journey.
                    The markets will always be here, and so will we if you ever want to come back.
                </p>

                <p style="font-size: 16px; color: #374151; margin: 24px 0 0 0; line-height: 1.6;">
                    All the best,<br>
                    <strong>The RigaCap Team</strong>
                </p>
            </td>
        </tr>

        <!-- Footer -->
        <tr>
            <td style="background-color: #f9fafb; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
                <p style="margin: 0 0 8px 0; font-size: 14px; color: #6b7280;">
                    <a href="#" style="color: #6b7280; text-decoration: none;">Unsubscribe</a>
                </p>
                <p style="margin: 0; font-size: 12px; color: #9ca3af;">
                    &copy; {datetime.now().year} RigaCap. All rights reserved.
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
"""

        text = f"""
Hey {first_name},

We noticed your RigaCap subscription has ended. We're sad to see you go!

What you're missing:
- Fresh daily signals from 4,500+ stocks
- Real-time market regime updates
- Buy signals before they surge
- Stop-loss alerts to protect your capital

Our backtest showed 216% returns. We'd love to have you back.

SPECIAL OFFER: Get 20% off your first month when you reactivate.
Visit: https://rigacap.com/app?promo=COMEBACK20

If something wasn't working for you, please reply and let us know — we're always improving.

All the best,
The RigaCap Team

---
Unsubscribe: https://rigacap.com/unsubscribe
"""

        return await self.send_email(
            to_email=to_email,
            subject="💔 We Miss You at RigaCap — Here's a Special Offer",
            html_content=html,
            text_content=text
        )


    async def send_ticker_alert(
        self,
        to_email: str,
        issues: list,
        check_type: str = "position"
    ) -> bool:
        """
        Send alert email when ticker issues are detected.

        Args:
            to_email: Admin email to alert
            issues: List of dicts with 'symbol', 'issue', 'last_price', 'last_date'
            check_type: 'position' or 'universe'

        Returns:
            True if sent successfully
        """
        if not issues:
            return True

        issue_rows = ""
        for issue in issues:
            symbol = issue.get('symbol', 'N/A')
            problem = issue.get('issue', 'Unknown issue')
            last_price = issue.get('last_price', 'N/A')
            last_date = issue.get('last_date', 'N/A')
            suggestion = issue.get('suggestion', 'Research ticker change or delisting')

            issue_rows += f"""
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #fee2e2; font-weight: 600; color: #dc2626;">
                    {symbol}
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #fee2e2; color: #374151;">
                    {problem}
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #fee2e2; color: #6b7280; font-size: 13px;">
                    Last: ${last_price} on {last_date}
                </td>
            </tr>
            <tr>
                <td colspan="3" style="padding: 8px 12px 16px; color: #92400e; font-size: 13px; background-color: #fef3c7;">
                    💡 {suggestion}
                </td>
            </tr>
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f3f4f6;">
    <table cellpadding="0" cellspacing="0" style="width: 100%; max-width: 600px; margin: 0 auto; background-color: #ffffff;">
        <!-- Header -->
        <tr>
            <td style="background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); padding: 32px 24px; text-align: center;">
                <div style="font-size: 40px; margin-bottom: 12px;">⚠️</div>
                <h1 style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 700;">
                    Ticker Health Alert
                </h1>
                <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.9); font-size: 14px;">
                    {len(issues)} issue(s) detected in {"open positions" if check_type == "position" else "stock universe"}
                </p>
            </td>
        </tr>

        <!-- Issues Table -->
        <tr>
            <td style="padding: 24px;">
                <p style="margin: 0 0 16px 0; font-size: 14px; color: #374151;">
                    The following tickers failed to return data during the daily health check.
                    This may indicate a ticker change, delisting, or merger.
                </p>

                <table cellpadding="0" cellspacing="0" style="width: 100%; border: 1px solid #fecaca; border-radius: 8px; overflow: hidden;">
                    <tr style="background-color: #fef2f2;">
                        <th style="padding: 12px; text-align: left; font-size: 12px; text-transform: uppercase; color: #991b1b;">Symbol</th>
                        <th style="padding: 12px; text-align: left; font-size: 12px; text-transform: uppercase; color: #991b1b;">Issue</th>
                        <th style="padding: 12px; text-align: left; font-size: 12px; text-transform: uppercase; color: #991b1b;">Last Known</th>
                    </tr>
                    {issue_rows}
                </table>
            </td>
        </tr>

        <!-- Action Items -->
        <tr>
            <td style="padding: 0 24px 24px;">
                <div style="background-color: #f0f9ff; border-radius: 12px; padding: 20px;">
                    <h3 style="margin: 0 0 12px 0; font-size: 16px; color: #0369a1;">
                        🔧 Recommended Actions
                    </h3>
                    <ol style="margin: 0; padding: 0 0 0 20px; color: #374151; line-height: 1.8;">
                        <li>Search for recent news about the affected ticker(s)</li>
                        <li>Check if ticker changed (e.g., SQ → XYZ for Block Inc)</li>
                        <li>If delisted/acquired, close any open positions manually</li>
                        <li>Update MUST_INCLUDE list in stock_universe.py if needed</li>
                    </ol>
                </div>
            </td>
        </tr>

        <!-- Footer -->
        <tr>
            <td style="background-color: #f9fafb; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
                <p style="margin: 0; font-size: 12px; color: #9ca3af;">
                    This is an automated alert from RigaCap Health Monitor
                </p>
                <p style="margin: 8px 0 0 0; font-size: 12px; color: #9ca3af;">
                    Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
"""

        text_lines = [
            "⚠️ TICKER HEALTH ALERT",
            "=" * 40,
            f"{len(issues)} issue(s) detected in {'open positions' if check_type == 'position' else 'stock universe'}",
            "",
        ]

        for issue in issues:
            text_lines.append(f"• {issue.get('symbol')}: {issue.get('issue')}")
            text_lines.append(f"  Last: ${issue.get('last_price', 'N/A')} on {issue.get('last_date', 'N/A')}")
            text_lines.append(f"  → {issue.get('suggestion', 'Research ticker change')}")
            text_lines.append("")

        text_lines.extend([
            "RECOMMENDED ACTIONS:",
            "1. Search for recent news about the ticker(s)",
            "2. Check if ticker changed (e.g., SQ → XYZ)",
            "3. Close positions manually if delisted",
            "4. Update MUST_INCLUDE list if needed",
        ])

        return await self.send_email(
            to_email=to_email,
            subject=f"⚠️ RigaCap Alert: {len(issues)} Ticker Issue(s) Detected",
            html_content=html,
            text_content="\n".join(text_lines)
        )


    async def send_strategy_analysis_email(
        self,
        to_email: str,
        analysis_results: dict,
        recommendation: str,
        switch_executed: bool = False,
        switch_reason: str = ""
    ) -> bool:
        """
        Send biweekly strategy analysis summary email.

        Args:
            to_email: Admin email
            analysis_results: Dict with evaluations and recommendation
            recommendation: Recommendation text
            switch_executed: Whether a switch was executed
            switch_reason: Reason for switch or why blocked
        """
        evaluations = analysis_results.get("evaluations", [])
        analysis_date = analysis_results.get("analysis_date", datetime.now().isoformat())
        lookback_days = analysis_results.get("lookback_days", 90)

        # Sort by score
        sorted_evals = sorted(evaluations, key=lambda x: x.get("recommendation_score", 0), reverse=True)

        eval_rows = ""
        for i, e in enumerate(sorted_evals[:5]):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            score_color = "#059669" if e.get("recommendation_score", 0) >= 60 else "#f59e0b" if e.get("recommendation_score", 0) >= 40 else "#6b7280"
            return_color = "#059669" if e.get("total_return_pct", 0) >= 0 else "#dc2626"

            eval_rows += f"""
            <tr>
                <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                    {rank_emoji} <strong>{e.get('name', 'Unknown')}</strong>
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: right;">
                    <span style="color: {score_color}; font-weight: 600;">{e.get('recommendation_score', 0):.0f}</span>
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: right;">
                    {e.get('sharpe_ratio', 0):.2f}
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; text-align: right; color: {return_color};">
                    {'+' if e.get('total_return_pct', 0) >= 0 else ''}{e.get('total_return_pct', 0):.1f}%
                </td>
            </tr>
            """

        status_color = "#059669" if switch_executed else "#f59e0b"
        status_text = "Switch Executed" if switch_executed else "No Switch"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f3f4f6;">
    <table cellpadding="0" cellspacing="0" style="width: 100%; max-width: 600px; margin: 0 auto; background-color: #ffffff;">
        <tr>
            <td style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); padding: 32px 24px; text-align: center;">
                <h1 style="margin: 0; color: #ffffff; font-size: 24px;">🔬 Strategy Analysis Report</h1>
                <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.9); font-size: 14px;">
                    {lookback_days}-day rolling backtest
                </p>
            </td>
        </tr>

        <!-- Status Banner -->
        <tr>
            <td style="padding: 24px;">
                <div style="background-color: {'#ecfdf5' if switch_executed else '#fef3c7'}; border-radius: 12px; padding: 16px; border-left: 4px solid {status_color};">
                    <div style="font-weight: 600; color: {status_color};">
                        {status_text}
                    </div>
                    <div style="color: #374151; margin-top: 4px;">
                        {switch_reason}
                    </div>
                </div>
            </td>
        </tr>

        <!-- Strategy Rankings -->
        <tr>
            <td style="padding: 0 24px 24px;">
                <h2 style="margin: 0 0 16px; font-size: 18px; color: #111827;">📊 Strategy Rankings</h2>
                <table cellpadding="0" cellspacing="0" style="width: 100%; border: 1px solid #e5e7eb; border-radius: 8px;">
                    <tr style="background-color: #f9fafb;">
                        <th style="padding: 12px; text-align: left; font-size: 12px; text-transform: uppercase; color: #6b7280;">Strategy</th>
                        <th style="padding: 12px; text-align: right; font-size: 12px; text-transform: uppercase; color: #6b7280;">Score</th>
                        <th style="padding: 12px; text-align: right; font-size: 12px; text-transform: uppercase; color: #6b7280;">Sharpe</th>
                        <th style="padding: 12px; text-align: right; font-size: 12px; text-transform: uppercase; color: #6b7280;">Return</th>
                    </tr>
                    {eval_rows}
                </table>
            </td>
        </tr>

        <!-- Recommendation -->
        <tr>
            <td style="padding: 0 24px 24px;">
                <div style="background-color: #f0f9ff; border-radius: 12px; padding: 16px;">
                    <h3 style="margin: 0 0 8px; font-size: 14px; color: #0369a1;">💡 Recommendation</h3>
                    <p style="margin: 0; color: #374151; white-space: pre-line;">{recommendation}</p>
                </div>
            </td>
        </tr>

        <tr>
            <td style="background-color: #f9fafb; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
                <p style="margin: 0; font-size: 12px; color: #9ca3af;">
                    Analysis completed at {analysis_date}<br>
                    RigaCap Strategy Management
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
"""

        return await self.send_email(
            to_email=to_email,
            subject=f"📊 Strategy Analysis: {status_text}",
            html_content=html
        )

    async def send_switch_notification_email(
        self,
        to_email: str,
        from_strategy: str,
        to_strategy: str,
        reason: str,
        metrics: dict
    ) -> bool:
        """
        Send notification when an automatic strategy switch occurs.

        Args:
            to_email: Admin email
            from_strategy: Previous strategy name
            to_strategy: New strategy name
            reason: Reason for the switch
            metrics: Dict with score_before, score_after, score_diff
        """
        score_before = metrics.get("score_before", 0)
        score_after = metrics.get("score_after", 0)
        score_diff = metrics.get("score_diff", 0)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f3f4f6;">
    <table cellpadding="0" cellspacing="0" style="width: 100%; max-width: 600px; margin: 0 auto; background-color: #ffffff;">
        <tr>
            <td style="background: linear-gradient(135deg, #059669 0%, #10b981 100%); padding: 32px 24px; text-align: center;">
                <div style="font-size: 48px; margin-bottom: 12px;">🔄</div>
                <h1 style="margin: 0; color: #ffffff; font-size: 24px;">Strategy Switch Executed</h1>
            </td>
        </tr>

        <tr>
            <td style="padding: 32px 24px;">
                <!-- Switch Visual -->
                <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 24px;">
                    <div style="background-color: #fee2e2; padding: 16px 24px; border-radius: 12px; text-align: center;">
                        <div style="font-size: 12px; color: #991b1b; text-transform: uppercase; margin-bottom: 4px;">From</div>
                        <div style="font-size: 18px; font-weight: 600; color: #dc2626;">{from_strategy or 'None'}</div>
                        <div style="font-size: 14px; color: #6b7280; margin-top: 4px;">Score: {score_before:.0f}</div>
                    </div>
                    <div style="font-size: 24px;">→</div>
                    <div style="background-color: #d1fae5; padding: 16px 24px; border-radius: 12px; text-align: center;">
                        <div style="font-size: 12px; color: #065f46; text-transform: uppercase; margin-bottom: 4px;">To</div>
                        <div style="font-size: 18px; font-weight: 600; color: #059669;">{to_strategy}</div>
                        <div style="font-size: 14px; color: #6b7280; margin-top: 4px;">Score: {score_after:.0f}</div>
                    </div>
                </div>

                <!-- Score Improvement -->
                <div style="background-color: #f0fdf4; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 24px;">
                    <div style="font-size: 14px; color: #065f46; margin-bottom: 8px;">Score Improvement</div>
                    <div style="font-size: 36px; font-weight: 700; color: #059669;">+{score_diff:.1f}</div>
                </div>

                <!-- Reason -->
                <div style="background-color: #f3f4f6; border-radius: 8px; padding: 16px;">
                    <div style="font-size: 14px; font-weight: 600; color: #374151; margin-bottom: 8px;">Reason</div>
                    <div style="color: #6b7280;">{reason}</div>
                </div>
            </td>
        </tr>

        <tr>
            <td style="padding: 0 24px 24px;">
                <p style="margin: 0; font-size: 14px; color: #6b7280;">
                    The new strategy is now active and will be used for all trading signals.
                    You can review and override this in the Admin Dashboard.
                </p>
            </td>
        </tr>

        <tr>
            <td style="background-color: #f9fafb; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
                <p style="margin: 0; font-size: 12px; color: #9ca3af;">
                    Switch executed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC<br>
                    RigaCap Strategy Management
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
"""

        return await self.send_email(
            to_email=to_email,
            subject=f"🔄 Strategy Switch: {from_strategy or 'None'} → {to_strategy}",
            html_content=html
        )

    async def send_generation_complete_email(
        self,
        to_email: str,
        best_params: dict,
        expected_metrics: dict,
        market_regime: str,
        created_strategy_name: str = None
    ) -> bool:
        """
        Send notification when AI strategy generation completes.

        Args:
            to_email: Admin email
            best_params: Best parameters found
            expected_metrics: Expected sharpe, return, drawdown
            market_regime: Detected market regime
            created_strategy_name: Name of created strategy (if auto_create was True)
        """
        params_html = ""
        for key, value in best_params.items():
            params_html += f"""
            <tr>
                <td style="padding: 8px 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">{key.replace('_', ' ').title()}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #e5e7eb; font-weight: 600; color: #111827;">{value}</td>
            </tr>
            """

        regime_colors = {
            "bull": ("#059669", "#d1fae5"),
            "bear": ("#dc2626", "#fee2e2"),
            "neutral": ("#6b7280", "#f3f4f6")
        }
        regime_color, regime_bg = regime_colors.get(market_regime, regime_colors["neutral"])

        created_section = ""
        if created_strategy_name:
            created_section = f"""
            <tr>
                <td style="padding: 0 24px 24px;">
                    <div style="background-color: #d1fae5; border-radius: 12px; padding: 16px; border-left: 4px solid #059669;">
                        <div style="font-weight: 600; color: #065f46;">✅ Strategy Created</div>
                        <div style="color: #374151; margin-top: 4px;">
                            "{created_strategy_name}" has been added to your strategy library.
                        </div>
                    </div>
                </td>
            </tr>
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f3f4f6;">
    <table cellpadding="0" cellspacing="0" style="width: 100%; max-width: 600px; margin: 0 auto; background-color: #ffffff;">
        <tr>
            <td style="background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%); padding: 32px 24px; text-align: center;">
                <div style="font-size: 48px; margin-bottom: 12px;">🤖</div>
                <h1 style="margin: 0; color: #ffffff; font-size: 24px;">AI Strategy Generation Complete</h1>
            </td>
        </tr>

        <!-- Market Regime -->
        <tr>
            <td style="padding: 24px;">
                <div style="background-color: {regime_bg}; border-radius: 12px; padding: 16px; text-align: center;">
                    <div style="font-size: 12px; color: #6b7280; text-transform: uppercase; margin-bottom: 4px;">Market Regime Detected</div>
                    <div style="font-size: 24px; font-weight: 700; color: {regime_color};">{market_regime.upper()}</div>
                </div>
            </td>
        </tr>

        <!-- Expected Metrics -->
        <tr>
            <td style="padding: 0 24px 24px;">
                <h2 style="margin: 0 0 16px; font-size: 18px; color: #111827;">📈 Expected Performance</h2>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                    <div style="background-color: #f0fdf4; border-radius: 8px; padding: 16px; text-align: center;">
                        <div style="font-size: 12px; color: #065f46; margin-bottom: 4px;">Sharpe Ratio</div>
                        <div style="font-size: 24px; font-weight: 700; color: #059669;">{expected_metrics.get('sharpe', 0):.2f}</div>
                    </div>
                    <div style="background-color: #f0fdf4; border-radius: 8px; padding: 16px; text-align: center;">
                        <div style="font-size: 12px; color: #065f46; margin-bottom: 4px;">Expected Return</div>
                        <div style="font-size: 24px; font-weight: 700; color: #059669;">+{expected_metrics.get('return', 0):.1f}%</div>
                    </div>
                    <div style="background-color: #fef2f2; border-radius: 8px; padding: 16px; text-align: center;">
                        <div style="font-size: 12px; color: #991b1b; margin-bottom: 4px;">Max Drawdown</div>
                        <div style="font-size: 24px; font-weight: 700; color: #dc2626;">-{expected_metrics.get('drawdown', 0):.1f}%</div>
                    </div>
                </div>
            </td>
        </tr>

        <!-- Best Parameters -->
        <tr>
            <td style="padding: 0 24px 24px;">
                <h2 style="margin: 0 0 16px; font-size: 18px; color: #111827;">⚙️ Optimal Parameters</h2>
                <table cellpadding="0" cellspacing="0" style="width: 100%; border: 1px solid #e5e7eb; border-radius: 8px;">
                    {params_html}
                </table>
            </td>
        </tr>

        {created_section}

        <tr>
            <td style="background-color: #f9fafb; padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
                <p style="margin: 0; font-size: 12px; color: #9ca3af;">
                    Generation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC<br>
                    RigaCap AI Strategy Generator
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
"""

        return await self.send_email(
            to_email=to_email,
            subject="🤖 AI Strategy Generation Complete",
            html_content=html
        )


# Singleton instance
email_service = EmailService()
