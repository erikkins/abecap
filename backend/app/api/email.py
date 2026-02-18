"""
Email API - Endpoints for email management and testing
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

from app.core.database import User
from app.core.security import get_admin_user
from app.services.email_service import email_service
from app.services.scanner import scanner_service
from app.services.market_analysis import market_analysis_service

router = APIRouter()


class TestEmailRequest(BaseModel):
    email: EmailStr


class TimeTravelEmailRequest(BaseModel):
    email: EmailStr
    as_of_date: str  # YYYY-MM-DD
    buy_signals: list = []
    regime_forecast: Optional[dict] = None
    watchlist: list = []


class EmailPreviewResponse(BaseModel):
    subject: str
    html: str
    text: str


@router.post("/test")
async def send_test_email(request: TestEmailRequest, admin: User = Depends(get_admin_user)):
    """
    Send a test daily summary email to a specific address

    Useful for testing email rendering and delivery.
    """
    try:
        # Get current signals
        signals = await scanner_service.scan(refresh_data=False)
        signal_dicts = [s.to_dict() for s in signals]

        # Get market regime
        regime = market_analysis_service.get_market_regime()

        # Mock some missed opportunities for the test email
        missed = [
            {
                "symbol": "NVDA",
                "signal_date": "2024-01-15",
                "would_be_return": 12.5,
                "would_be_pnl": 1250.00
            },
            {
                "symbol": "META",
                "signal_date": "2024-01-18",
                "would_be_return": 8.3,
                "would_be_pnl": 830.00
            }
        ]

        # Mock some positions
        positions = [
            {
                "symbol": "AAPL",
                "shares": 50,
                "entry_price": 185.00,
                "current_price": 192.50
            },
            {
                "symbol": "MSFT",
                "shares": 30,
                "entry_price": 375.00,
                "current_price": 388.00
            }
        ]

        # Send test email
        success = await email_service.send_daily_summary(
            to_email=request.email,
            signals=signal_dicts,
            market_regime=regime,
            positions=positions,
            missed_opportunities=missed
        )

        if success:
            return {
                "success": True,
                "message": f"Test email sent to {request.email}",
                "signals_count": len(signals),
                "strong_signals": len([s for s in signals if s.is_strong])
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to send email. Check SMTP configuration."
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/preview", response_model=EmailPreviewResponse)
async def preview_daily_email(admin: User = Depends(get_admin_user)):
    """
    Preview the daily summary email without sending

    Returns the HTML and text content that would be sent.
    """
    try:
        # Get current signals
        signals = await scanner_service.scan(refresh_data=False)
        signal_dicts = [s.to_dict() for s in signals]

        # Get market regime
        regime = market_analysis_service.get_market_regime()

        # Generate email content
        strong_count = len([s for s in signals if s.is_strong])
        subject = f"📊 RigaCap Daily: {strong_count} Strong Signals Found"

        html = email_service.generate_daily_summary_html(
            signals=signal_dicts,
            market_regime=regime,
            positions=[],
            missed_opportunities=[]
        )

        text = email_service.generate_plain_text(signal_dicts, regime)

        return EmailPreviewResponse(
            subject=subject,
            html=html,
            text=text
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status")
async def get_email_status(admin: User = Depends(get_admin_user)):
    """
    Get email service status and configuration
    """
    return {
        "enabled": email_service.enabled,
        "smtp_host": email_service.enabled and "configured" or "not configured",
        "next_scheduled": "6:00 PM ET (Mon-Fri)"
    }


@router.post("/trigger-daily")
async def trigger_daily_emails(background_tasks: BackgroundTasks, admin: User = Depends(get_admin_user)):
    """
    Manually trigger the daily email job

    Runs in background and sends to all configured subscribers.
    """
    from app.services.scheduler import scheduler_service

    background_tasks.add_task(scheduler_service.send_daily_emails)

    return {
        "success": True,
        "message": "Daily email job triggered in background"
    }


@router.post("/time-travel")
async def send_time_travel_email(request: TimeTravelEmailRequest, admin: User = Depends(get_admin_user)):
    """
    Send the daily summary email as it would have appeared on a historical date.

    Accepts pre-computed dashboard data from the frontend (admin time-travel mode)
    and sends the email with the historical date in the header/subject.
    """
    try:
        email_date = datetime.strptime(request.as_of_date, "%Y-%m-%d")

        # Map regime_forecast to the market_regime format the email template expects
        regime = {}
        if request.regime_forecast:
            regime = {
                'regime': request.regime_forecast.get('current_regime', 'unknown'),
                'spy_price': request.regime_forecast.get('spy_price'),
                'vix_level': request.regime_forecast.get('vix_level'),
            }

        success = await email_service.send_daily_summary(
            to_email=request.email,
            signals=request.buy_signals,
            market_regime=regime,
            watchlist=request.watchlist,
            regime_forecast=request.regime_forecast,
            date=email_date
        )

        if success:
            fresh = len([s for s in request.buy_signals if s.get('is_fresh')])
            return {
                "success": True,
                "message": f"Time-travel email sent for {request.as_of_date}",
                "signals_count": len(request.buy_signals),
                "fresh_signals": fresh
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to send email. Check SMTP configuration."
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid date format")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
