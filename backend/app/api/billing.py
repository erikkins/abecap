"""Billing API endpoints for Stripe integration."""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db, User, Subscription
from app.core.config import settings
from app.core.security import get_current_user

router = APIRouter()


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


class PortalResponse(BaseModel):
    portal_url: str


class SubscriptionResponse(BaseModel):
    status: str
    is_valid: bool
    days_remaining: int
    trial_end: Optional[str] = None
    current_period_end: Optional[str] = None
    cancel_at_period_end: bool = False


class CheckoutRequest(BaseModel):
    plan: str = "monthly"  # "monthly" or "annual"


@router.post("/create-checkout", response_model=CheckoutResponse)
async def create_checkout_session(
    request: CheckoutRequest = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a Stripe Checkout session for subscription.

    Args:
        plan: "monthly" ($10/month) or "annual" ($110/year, one month free)
    """
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe is not configured"
        )

    # Determine which price ID to use
    plan = request.plan if request else "monthly"
    if plan == "annual":
        price_id = settings.STRIPE_PRICE_ID_ANNUAL
        if not price_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Annual plan not configured"
            )
    else:
        price_id = settings.STRIPE_PRICE_ID
        if not price_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Monthly plan not configured"
            )

    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY

    try:
        # Create or get Stripe customer
        if not user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=user.email,
                name=user.name,
                metadata={"user_id": str(user.id)}
            )
            user.stripe_customer_id = customer.id
            await db.commit()

        # Create checkout session
        session = stripe.checkout.Session.create(
            customer=user.stripe_customer_id,
            mode="subscription",
            line_items=[{
                "price": price_id,
                "quantity": 1
            }],
            success_url=f"{settings.FRONTEND_URL}/dashboard?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{settings.FRONTEND_URL}/pricing?checkout=canceled",
            allow_promotion_codes=True,
            automatic_payment_methods={"enabled": True},
        )

        return CheckoutResponse(
            checkout_url=session.url,
            session_id=session.id
        )

    except stripe.StripeError as e:
        print(f"❌ Stripe error in create-checkout: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Payment service error: {str(e)}"
        )
    except Exception as e:
        print(f"❌ Unexpected error in create-checkout: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Checkout error: {type(e).__name__}"
        )


@router.post("/portal", response_model=PortalResponse)
async def create_portal_session(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a Stripe Customer Portal session for managing subscription."""
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe is not configured"
        )

    if not user.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No billing account found"
        )

    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY

    try:
        session = stripe.billing_portal.Session.create(
            customer=user.stripe_customer_id,
            return_url=f"{settings.FRONTEND_URL}/dashboard"
        )
        return PortalResponse(portal_url=session.url)
    except stripe.StripeError as e:
        print(f"❌ Stripe error in portal: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Payment service error: {str(e)}"
        )


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's subscription status."""
    result = await db.execute(
        select(Subscription).where(Subscription.user_id == user.id)
    )
    subscription = result.scalar_one_or_none()

    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No subscription found"
        )

    return SubscriptionResponse(
        status=subscription.status,
        is_valid=subscription.is_valid(),
        days_remaining=subscription.days_remaining(),
        trial_end=subscription.trial_end.isoformat() if subscription.trial_end else None,
        current_period_end=subscription.current_period_end.isoformat() if subscription.current_period_end else None,
        cancel_at_period_end=subscription.cancel_at_period_end
    )


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
    db: AsyncSession = Depends(get_db)
):
    """Handle Stripe webhook events."""
    if not settings.STRIPE_SECRET_KEY or not settings.STRIPE_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe webhooks not configured"
        )

    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY

    payload = await request.body()

    try:
        event = stripe.Webhook.construct_event(
            payload,
            stripe_signature,
            settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the event
    if event.type == "customer.subscription.created":
        await handle_subscription_created(event.data.object, db)
    elif event.type == "customer.subscription.updated":
        await handle_subscription_updated(event.data.object, db)
    elif event.type == "customer.subscription.deleted":
        await handle_subscription_deleted(event.data.object, db)
    elif event.type == "invoice.payment_failed":
        await handle_payment_failed(event.data.object, db)

    return {"received": True}


async def handle_subscription_created(stripe_sub, db: AsyncSession):
    """Handle new Stripe subscription created."""
    customer_id = stripe_sub.customer

    # Find user by Stripe customer ID
    result = await db.execute(
        select(User).where(User.stripe_customer_id == customer_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        print(f"No user found for Stripe customer {customer_id}")
        return

    # Get or create subscription
    result = await db.execute(
        select(Subscription).where(Subscription.user_id == user.id)
    )
    subscription = result.scalar_one_or_none()

    if not subscription:
        subscription = Subscription(user_id=user.id)
        db.add(subscription)

    # Update subscription details
    subscription.status = "active"
    subscription.stripe_subscription_id = stripe_sub.id
    subscription.stripe_price_id = stripe_sub.items.data[0].price.id if stripe_sub.items.data else None
    subscription.current_period_start = datetime.fromtimestamp(stripe_sub.current_period_start)
    subscription.current_period_end = datetime.fromtimestamp(stripe_sub.current_period_end)
    subscription.cancel_at_period_end = stripe_sub.cancel_at_period_end

    await db.commit()


async def handle_subscription_updated(stripe_sub, db: AsyncSession):
    """Handle Stripe subscription updated."""
    # Find subscription by Stripe ID
    result = await db.execute(
        select(Subscription).where(Subscription.stripe_subscription_id == stripe_sub.id)
    )
    subscription = result.scalar_one_or_none()

    if not subscription:
        # Try to find by customer ID
        customer_id = stripe_sub.customer
        result = await db.execute(
            select(User).where(User.stripe_customer_id == customer_id)
        )
        user = result.scalar_one_or_none()

        if user:
            result = await db.execute(
                select(Subscription).where(Subscription.user_id == user.id)
            )
            subscription = result.scalar_one_or_none()

    if not subscription:
        print(f"No subscription found for Stripe sub {stripe_sub.id}")
        return

    # Map Stripe status to our status
    status_map = {
        "active": "active",
        "past_due": "past_due",
        "canceled": "canceled",
        "unpaid": "past_due",
        "trialing": "trial",
    }

    subscription.status = status_map.get(stripe_sub.status, stripe_sub.status)
    subscription.stripe_subscription_id = stripe_sub.id
    subscription.current_period_start = datetime.fromtimestamp(stripe_sub.current_period_start)
    subscription.current_period_end = datetime.fromtimestamp(stripe_sub.current_period_end)
    subscription.cancel_at_period_end = stripe_sub.cancel_at_period_end

    await db.commit()


async def handle_subscription_deleted(stripe_sub, db: AsyncSession):
    """Handle Stripe subscription canceled/deleted."""
    result = await db.execute(
        select(Subscription).where(Subscription.stripe_subscription_id == stripe_sub.id)
    )
    subscription = result.scalar_one_or_none()

    if subscription:
        subscription.status = "canceled"
        subscription.cancel_at_period_end = False
        await db.commit()


async def handle_payment_failed(invoice, db: AsyncSession):
    """Handle failed payment."""
    subscription_id = invoice.subscription

    if subscription_id:
        result = await db.execute(
            select(Subscription).where(Subscription.stripe_subscription_id == subscription_id)
        )
        subscription = result.scalar_one_or_none()

        if subscription:
            subscription.status = "past_due"
            await db.commit()
