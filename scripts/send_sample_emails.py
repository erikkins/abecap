"""
Send sample emails of all configured email types to a test address.

Usage:
    python scripts/send_sample_emails.py [email@example.com]
"""

import asyncio
import os
import sys

# Set SMTP credentials before importing email service
os.environ['SMTP_HOST'] = 'smtp.gmail.com'
os.environ['SMTP_PORT'] = '587'
os.environ['SMTP_USER'] = os.getenv('SMTP_USER', 'erik@rigacap.com')
os.environ['SMTP_PASS'] = os.getenv('SMTP_PASS', '')  # Set via env var
os.environ['FROM_EMAIL'] = 'signals@rigacap.com'
os.environ['FROM_NAME'] = 'RigaCap Signals'

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.email_service import EmailService, AdminEmailService, ADMIN_EMAILS

ADMIN_EMAIL = "erik@rigacap.com"


async def send_all_samples(to_email: str):
    svc = EmailService()
    admin_svc = AdminEmailService()

    if not svc.enabled:
        print("ERROR: Email service not enabled. Check SMTP credentials.")
        return

    results = {}

    # 1. Daily Summary (ensemble format with freshness + watchlist)
    print("1/9 Sending Daily Summary...")
    results['daily_summary'] = await svc.send_daily_summary(
        to_email=to_email,
        signals=[
            {"symbol": "NVDA", "price": 892.50, "pct_above_dwap": 8.2, "momentum_rank": 1, "is_strong": True, "is_fresh": True, "days_since_crossover": 0},
            {"symbol": "META", "price": 512.30, "pct_above_dwap": 6.7, "momentum_rank": 3, "is_strong": True, "is_fresh": True, "days_since_crossover": 2},
            {"symbol": "AMZN", "price": 185.40, "pct_above_dwap": 5.9, "momentum_rank": 5, "is_strong": False, "is_fresh": True, "days_since_crossover": 4},
            {"symbol": "AVGO", "price": 1345.00, "pct_above_dwap": 5.3, "momentum_rank": 8, "is_strong": False, "is_fresh": False, "days_since_crossover": 12},
            {"symbol": "CRM", "price": 298.75, "pct_above_dwap": 5.1, "momentum_rank": 14, "is_strong": False, "is_fresh": False, "days_since_crossover": 30},
        ],
        market_regime={"regime": "strong_bull", "spy_price": 523.45, "vix_level": 14.2},
        positions=[
            {"symbol": "AAPL", "shares": 50, "entry_price": 185.00, "current_price": 198.50},
            {"symbol": "MSFT", "shares": 30, "entry_price": 375.00, "current_price": 412.00},
            {"symbol": "GOOGL", "shares": 25, "entry_price": 142.00, "current_price": 155.80},
        ],
        missed_opportunities=[
            {"symbol": "SMCI", "signal_date": "2024-01-10", "would_be_return": 45.2, "would_be_pnl": 4520},
            {"symbol": "ARM", "signal_date": "2024-01-15", "would_be_return": 22.8, "would_be_pnl": 2280},
            {"symbol": "PLTR", "signal_date": "2024-01-22", "would_be_return": 15.3, "would_be_pnl": 1530},
        ],
        watchlist=[
            {"symbol": "TSLA", "price": 245.80, "pct_above_dwap": 3.8, "distance_to_trigger": 1.2},
            {"symbol": "AMD", "price": 178.50, "pct_above_dwap": 4.1, "distance_to_trigger": 0.9},
            {"symbol": "NFLX", "price": 620.30, "pct_above_dwap": 3.5, "distance_to_trigger": 1.5},
        ]
    )

    # 2. Welcome Email
    print("2/9 Sending Welcome Email...")
    results['welcome'] = await svc.send_welcome_email(
        to_email=to_email,
        name="Erik Kinsman"
    )

    # 3. Trial Ending Email
    print("3/9 Sending Trial Ending Email...")
    results['trial_ending'] = await svc.send_trial_ending_email(
        to_email=to_email,
        name="Erik Kinsman",
        days_remaining=2,
        signals_generated=47,
        strong_signals_seen=12
    )

    # 4. Goodbye Email
    print("4/9 Sending Goodbye Email...")
    results['goodbye'] = await svc.send_goodbye_email(
        to_email=to_email,
        name="Erik Kinsman"
    )

    # --- ADMIN EMAILS (sent to admin address only) ---

    # 5. Ticker Alert (ADMIN)
    print("5/9 Sending Ticker Alert (admin)...")
    results['ticker_alert'] = await admin_svc.send_ticker_alert(
        to_email=ADMIN_EMAIL,
        issues=[
            {
                "symbol": "ATVI",
                "issue": "No data returned - possible delisting after MSFT acquisition",
                "last_price": 95.10,
                "last_date": "2023-10-13",
                "suggestion": "Ticker delisted after Microsoft acquisition completed. Close position and remove from universe."
            },
            {
                "symbol": "TWTR",
                "issue": "Ticker changed after acquisition",
                "last_price": 53.70,
                "last_date": "2022-10-27",
                "suggestion": "Twitter was taken private by Elon Musk. Remove from stock universe."
            },
        ],
        check_type="position"
    )

    # 6. Strategy Analysis (ADMIN)
    print("6/9 Sending Strategy Analysis (admin)...")
    results['strategy_analysis'] = await admin_svc.send_strategy_analysis_email(
        to_email=ADMIN_EMAIL,
        analysis_results={
            "analysis_date": "2026-02-10T16:00:00",
            "lookback_days": 90,
            "evaluations": [
                {"name": "Momentum v2", "recommendation_score": 78, "sharpe_ratio": 1.48, "total_return_pct": 14.2},
                {"name": "DWAP Classic", "recommendation_score": 52, "sharpe_ratio": 0.85, "total_return_pct": 8.1},
                {"name": "Mean Reversion", "recommendation_score": 45, "sharpe_ratio": 0.62, "total_return_pct": 5.3},
                {"name": "Breakout v1", "recommendation_score": 38, "sharpe_ratio": 0.41, "total_return_pct": 3.7},
            ]
        },
        recommendation="Momentum v2 continues to outperform all alternatives with a recommendation score of 78. No strategy switch recommended at this time. The current bull market regime favors momentum-based approaches.",
        switch_executed=False,
        switch_reason="Current strategy (Momentum v2) remains the top performer. Score difference of 26 pts vs. next best (DWAP Classic) exceeds the minimum threshold of 10 pts."
    )

    # 7. Strategy Switch Notification (ADMIN)
    print("7/9 Sending Strategy Switch (admin)...")
    results['switch_notification'] = await admin_svc.send_switch_notification_email(
        to_email=ADMIN_EMAIL,
        from_strategy="DWAP Classic",
        to_strategy="Momentum v2",
        reason="Momentum v2 has consistently outperformed DWAP Classic over the last 90 days with a significantly higher Sharpe ratio (1.48 vs 0.85) and better risk-adjusted returns.",
        metrics={"score_before": 52, "score_after": 78, "score_diff": 26}
    )

    # 8. AI Generation Complete (ADMIN)
    print("8/9 Sending AI Generation Complete (admin)...")
    results['generation_complete'] = await admin_svc.send_generation_complete_email(
        to_email=ADMIN_EMAIL,
        best_params={
            "short_momentum_days": 10,
            "long_momentum_days": 60,
            "trailing_stop_pct": 15,
            "max_positions": 5,
            "position_size_pct": 18,
            "rebalance_frequency": "weekly",
            "regime_filter": "SPY > 200MA",
        },
        expected_metrics={"sharpe": 1.52, "return": 16.3, "drawdown": 12.8},
        market_regime="bull",
        created_strategy_name="AI-Bull-2026-02"
    )

    # 9. Double Signal Alert (with regime context + freshness)
    print("9/9 Sending Double Signal Alert...")
    results['double_signal'] = await svc.send_double_signal_alert(
        to_email=to_email,
        new_signals=[
            {"symbol": "NVDA", "price": 892.50, "pct_above_dwap": 8.2, "momentum_rank": 1, "short_momentum": 12.5, "dwap_crossover_date": "Today", "days_since_crossover": 0},
            {"symbol": "META", "price": 512.30, "pct_above_dwap": 6.7, "momentum_rank": 3, "short_momentum": 8.3, "dwap_crossover_date": "Today", "days_since_crossover": 0},
            {"symbol": "AVGO", "price": 1345.00, "pct_above_dwap": 5.3, "momentum_rank": 7, "short_momentum": 6.1, "dwap_crossover_date": "2026-02-10", "days_since_crossover": 1},
        ],
        approaching=[
            {"symbol": "AMZN", "price": 185.40, "pct_above_dwap": 4.2, "distance_to_trigger": 0.8},
            {"symbol": "CRM", "price": 298.75, "pct_above_dwap": 3.8, "distance_to_trigger": 1.2},
        ],
        market_regime={"regime": "strong_bull", "spy_price": 523.45}
    )

    # Print results
    print("\n" + "=" * 50)
    print(f"Results for {to_email}:")
    print("=" * 50)
    for name, success in results.items():
        status = "SENT" if success else "FAILED"
        print(f"  {name:.<35} {status}")

    sent = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {sent}/{total} emails sent successfully")


if __name__ == "__main__":
    to_email = sys.argv[1] if len(sys.argv) > 1 else "erikkins@gmail.com"
    print(f"Sending 5 subscriber emails to: {to_email}")
    print(f"Sending 4 admin emails to: {ADMIN_EMAIL}")
    print(f"Admin allowlist: {ADMIN_EMAILS}\n")
    asyncio.run(send_all_samples(to_email))
