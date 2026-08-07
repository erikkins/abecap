"""
Social Media Admin API - Post queue management for admin review/approval.

All endpoints require admin authentication.
"""

import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, delete

from app.core.config import settings
from app.core.database import get_db, SocialPost
from app.core.security import get_admin_user


class ComposeRequest(BaseModel):
    platform: str  # "twitter" or "instagram"
    text_content: str
    hashtags: Optional[str] = None
    post_type: str = "manual"
    status: str = "draft"  # "draft" or "approved"
    image_s3_key: Optional[str] = None


class EditRequest(BaseModel):
    text_content: Optional[str] = None
    hashtags: Optional[str] = None


class ScheduleRequest(BaseModel):
    publish_at: str  # ISO datetime string

router = APIRouter()


@router.get("/posts")
async def list_posts(
    status: Optional[str] = None,
    post_type: Optional[str] = None,
    platform: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """List social posts with optional filters."""
    if status == "scheduled":
        # Scheduled posts: next-to-publish first
        query = select(SocialPost).order_by(SocialPost.scheduled_for.asc())
    else:
        query = select(SocialPost).order_by(desc(SocialPost.created_at))

    if status:
        query = query.where(SocialPost.status == status)
    if post_type:
        query = query.where(SocialPost.post_type == post_type)
    if platform:
        query = query.where(SocialPost.platform == platform)

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    posts = result.scalars().all()

    return {
        "posts": [_post_to_dict(p) for p in posts],
        "count": len(posts),
        "offset": offset,
        "limit": limit,
    }


@router.get("/posts/{post_id}")
async def get_post(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Get a single post with full content."""
    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    return _post_to_dict(post, include_source=True)


@router.post("/posts/{post_id}/approve")
async def approve_post(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_admin_user),
):
    """Mark a post as approved for publishing."""
    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.status not in ("draft", "rejected", "scheduled", "posted"):
        raise HTTPException(status_code=400, detail=f"Cannot approve post with status '{post.status}'")

    # Clear image when re-approving a posted post so text card regenerates on next publish
    if post.status == "posted" and post.platform == "instagram":
        post.image_s3_key = None

    post.status = "approved"
    post.reviewed_by = admin.email
    post.reviewed_at = datetime.utcnow()
    post.rejection_reason = None
    post.scheduled_for = None  # Clear schedule when approving
    await db.commit()

    return {"status": "approved", "post_id": post_id}


@router.post("/posts/{post_id}/reject")
async def reject_post(
    post_id: int,
    reason: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_admin_user),
):
    """Reject a post with optional reason."""
    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.status not in ("draft", "approved"):
        raise HTTPException(status_code=400, detail=f"Cannot reject post with status '{post.status}'")

    post.status = "rejected"
    post.reviewed_by = admin.email
    post.reviewed_at = datetime.utcnow()
    post.rejection_reason = reason
    await db.commit()

    return {"status": "rejected", "post_id": post_id, "reason": reason}


@router.get("/posts/{post_id}/preview")
async def preview_post(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Preview a post with text content and presigned image URL."""
    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    preview = {
        "post_id": post.id,
        "platform": post.platform,
        "post_type": post.post_type,
        "text_content": post.text_content,
        "hashtags": post.hashtags,
        "image_url": None,
    }

    # Generate presigned URL for image if available
    if post.image_s3_key:
        from app.services.chart_card_generator import chart_card_generator
        preview["image_url"] = chart_card_generator.get_presigned_url(post.image_s3_key)

    # Full post text with hashtags
    full_text = post.text_content or ""
    if post.hashtags:
        full_text += f"\n\n{post.hashtags}"
    preview["full_text"] = full_text

    # Character count (relevant for Twitter)
    preview["char_count"] = len(full_text)
    if post.platform == "twitter":
        preview["over_limit"] = len(full_text) > 280

    return preview


@router.post("/posts/{post_id}/regenerate")
async def regenerate_post(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Re-generate content from the post's source data."""
    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if not post.source_trade_json and not post.source_data_json:
        raise HTTPException(status_code=400, detail="No source data available for regeneration")

    from app.services.social_content_service import social_content_service

    # Re-generate based on post type and source data
    trade = json.loads(post.source_trade_json) if post.source_trade_json else None

    if trade and post.post_type == "trade_result":
        if post.platform == "twitter":
            new_post = social_content_service._make_trade_result_twitter(trade)
        else:
            new_post = social_content_service._make_trade_result_instagram(trade)
        post.text_content = new_post.text_content
        post.hashtags = new_post.hashtags
    elif trade and post.post_type == "missed_opportunity":
        if post.platform == "twitter":
            new_post = social_content_service._make_missed_opportunity_twitter(trade)
        else:
            new_post = social_content_service._make_missed_opportunity_instagram(trade)
        post.text_content = new_post.text_content
        post.hashtags = new_post.hashtags

    post.status = "draft"
    post.reviewed_at = None
    post.reviewed_by = None
    await db.commit()

    return {"status": "regenerated", "post_id": post_id}


@router.delete("/posts/{post_id}")
async def delete_post(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Delete a draft or rejected post."""
    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.status not in ("draft", "rejected"):
        raise HTTPException(status_code=400, detail=f"Cannot delete post with status '{post.status}'")

    await db.delete(post)
    await db.commit()

    return {"status": "deleted", "post_id": post_id}


@router.get("/stats")
async def get_stats(
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Get post counts by status, type, and platform."""
    # Count by status
    status_result = await db.execute(
        select(SocialPost.status, func.count(SocialPost.id))
        .group_by(SocialPost.status)
    )
    by_status = {row[0]: row[1] for row in status_result.fetchall()}

    # Count by type
    type_result = await db.execute(
        select(SocialPost.post_type, func.count(SocialPost.id))
        .group_by(SocialPost.post_type)
    )
    by_type = {row[0]: row[1] for row in type_result.fetchall()}

    # Count by platform
    platform_result = await db.execute(
        select(SocialPost.platform, func.count(SocialPost.id))
        .group_by(SocialPost.platform)
    )
    by_platform = {row[0]: row[1] for row in platform_result.fetchall()}

    # Total
    total_result = await db.execute(select(func.count(SocialPost.id)))
    total = total_result.scalar() or 0

    return {
        "total": total,
        "by_status": by_status,
        "by_type": by_type,
        "by_platform": by_platform,
    }


@router.get("/platform-toggles")
async def get_platform_toggles_endpoint(_admin=Depends(get_admin_user)):
    """Per-platform posting on/off state (the Social tab pause switches)."""
    from app.services.social_platform_toggles import get_platform_toggles
    return {"toggles": get_platform_toggles(force=True)}


@router.post("/platform-toggles")
async def set_platform_toggles_endpoint(payload: dict, _admin=Depends(get_admin_user)):
    """Flip one or more platforms on/off. Body e.g. {"instagram": false}."""
    from app.services.social_platform_toggles import set_platform_toggles, PLATFORMS
    updates = {k: v for k, v in (payload or {}).items() if k in PLATFORMS}
    if not updates:
        raise HTTPException(status_code=400, detail="No valid platform in payload")
    return {"toggles": set_platform_toggles(updates)}


@router.post("/generate-chart/{post_id}")
async def generate_chart_card(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Generate and upload a chart card image for an Instagram post."""
    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.platform != "instagram":
        raise HTTPException(status_code=400, detail="Chart cards are only for Instagram posts")

    # Parse image metadata
    if not post.image_metadata_json:
        raise HTTPException(status_code=400, detail="No image metadata available")

    meta = json.loads(post.image_metadata_json)

    from app.services.chart_card_generator import chart_card_generator

    # Fetch price data for this symbol (API Lambda doesn't load the pickle)
    symbol = meta.get("symbol", "???")
    from app.services.scanner import scanner_service
    if symbol not in scanner_service.data_cache:
        import pandas as pd
        from app.services.market_data_provider import market_data_provider
        entry_dt = pd.Timestamp(meta.get("entry_date", "")[:10])
        fetch_start = (entry_dt - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        bars = await market_data_provider.fetch_bars([symbol], fetch_start)
        if symbol in bars:
            scanner_service.data_cache[symbol] = bars[symbol]

    # Generate the image
    png_bytes = chart_card_generator.generate_trade_card(
        symbol=meta.get("symbol", "???"),
        entry_price=meta.get("entry_price", 0),
        exit_price=meta.get("exit_price", 0),
        entry_date=meta.get("entry_date", ""),
        exit_date=meta.get("exit_date", ""),
        pnl_pct=meta.get("pnl_pct", 0),
        pnl_dollars=meta.get("pnl_dollars", 0),
        exit_reason=meta.get("exit_reason", "trailing_stop"),
        strategy_name=meta.get("strategy_name", "Ensemble"),
        regime_name=meta.get("regime_name", ""),
        company_name=meta.get("company_name", ""),
    )

    # Upload to S3
    date_str = meta.get("exit_date", "")[:10].replace("-", "")
    s3_key = chart_card_generator.upload_to_s3(
        png_bytes, post.id, meta.get("symbol", "UNK"), date_str
    )

    if s3_key:
        post.image_s3_key = s3_key
        await db.commit()
        image_url = chart_card_generator.get_presigned_url(s3_key)
    else:
        image_url = None

    return {
        "status": "generated",
        "post_id": post_id,
        "s3_key": s3_key,
        "image_url": image_url,
    }


@router.post("/posts/{post_id}/publish")
async def publish_post(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Publish an approved post to its target platform (Twitter/Instagram)."""
    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.status != "approved":
        raise HTTPException(
            status_code=400,
            detail=f"Only approved posts can be published (current: '{post.status}')",
        )

    from app.services.social_posting_service import social_posting_service

    pub_result = await social_posting_service.publish_post(post)

    if "error" in pub_result:
        raise HTTPException(status_code=502, detail=pub_result["error"])

    await db.commit()

    return {
        "status": "posted",
        "post_id": post_id,
        "platform": post.platform,
        **pub_result,
    }


@router.post("/posts/compose")
async def compose_post(
    body: ComposeRequest,
    db: AsyncSession = Depends(get_db),
    admin=Depends(get_admin_user),
):
    """Create a new manual social post."""
    if body.platform not in ("twitter", "instagram", "threads"):
        raise HTTPException(status_code=400, detail="Platform must be 'twitter', 'instagram', or 'threads'")
    if body.status not in ("draft", "approved"):
        raise HTTPException(status_code=400, detail="Status must be 'draft' or 'approved'")

    post = SocialPost(
        platform=body.platform,
        text_content=body.text_content,
        hashtags=body.hashtags,
        post_type=body.post_type,
        status=body.status,
        image_s3_key=body.image_s3_key,
    )

    if body.status == "approved":
        post.reviewed_by = admin.email
        post.reviewed_at = datetime.utcnow()

    db.add(post)
    await db.commit()
    await db.refresh(post)

    return {"status": post.status, "post": _post_to_dict(post)}


@router.post("/posts/{post_id}/edit")
async def edit_post(
    post_id: int,
    body: EditRequest,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Edit text content and/or hashtags of a draft or approved post."""
    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.status not in ("draft", "approved"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot edit post with status '{post.status}'",
        )

    if body.text_content is not None:
        post.text_content = body.text_content
    if body.hashtags is not None:
        post.hashtags = body.hashtags

    await db.commit()

    return {"status": "updated", "post": _post_to_dict(post)}


@router.post("/posts/{post_id}/schedule")
async def schedule_post(
    post_id: int,
    body: ScheduleRequest,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Schedule an approved post for auto-publishing at a specific time."""
    from app.services.post_scheduler_service import post_scheduler_service

    try:
        publish_at = datetime.fromisoformat(body.publish_at.replace("Z", "+00:00"))
        # Strip timezone info for naive UTC comparison (consistent with rest of codebase)
        publish_at = publish_at.replace(tzinfo=None)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO 8601.")

    if publish_at <= datetime.utcnow():
        raise HTTPException(status_code=400, detail="Scheduled time must be in the future")

    success = await post_scheduler_service.schedule_post(post_id, publish_at, db)
    if not success:
        raise HTTPException(status_code=400, detail="Could not schedule post. Must be draft or approved.")

    return {"status": "scheduled", "post_id": post_id, "publish_at": publish_at.isoformat()}


@router.post("/posts/{post_id}/cancel")
async def cancel_post(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Cancel a scheduled post (sets status='cancelled')."""
    from app.services.post_scheduler_service import post_scheduler_service

    success = await post_scheduler_service.cancel_post(post_id, db)
    if not success:
        raise HTTPException(status_code=400, detail="Could not cancel post. Already posted or not found.")

    return {"status": "approved", "post_id": post_id, "message": "Schedule cleared. Post is ready to reschedule."}


@router.get("/posts/{post_id}/cancel-email")
async def cancel_post_via_email(
    post_id: int,
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """One-click cancel from email link (JWT-authenticated, no login needed)."""
    from fastapi.responses import HTMLResponse
    from app.services.post_scheduler_service import post_scheduler_service

    verified_post_id = post_scheduler_service.verify_cancel_token(token)
    if verified_post_id is None or verified_post_id != post_id:
        return HTMLResponse(
            content="<html><body><h2>Invalid or expired cancel link.</h2>"
            "<p>Please log in to the admin dashboard to manage posts.</p></body></html>",
            status_code=400,
        )

    success = await post_scheduler_service.cancel_post(post_id, db)

    # Autopost fans one insight out to X + Threads + Instagram as sibling rows sharing a
    # scheduled_for — cancel them all so one kill click stops the whole insight everywhere.
    siblings_cancelled = 0
    try:
        target = (await db.execute(select(SocialPost).where(SocialPost.id == post_id))).scalar_one_or_none()
        if target and target.post_type == "research_insight" and target.scheduled_for:
            sibs = (await db.execute(
                select(SocialPost).where(
                    SocialPost.post_type == "research_insight",
                    SocialPost.scheduled_for == target.scheduled_for,
                    SocialPost.id != post_id,
                    SocialPost.status.in_(["scheduled", "approved", "draft"]),
                )
            )).scalars().all()
            for s in sibs:
                s.status = "cancelled"
                siblings_cancelled += 1
            if siblings_cancelled:
                await db.commit()
    except Exception:
        pass

    if success:
        extra = f" (plus {siblings_cancelled} on the other platform{'s' if siblings_cancelled != 1 else ''})" if siblings_cancelled else ""
        return HTMLResponse(
            content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            "<h2 style='color:#059669;'>Post Cancelled</h2>"
            f"<p>The scheduled post has been cancelled and will not be published{extra}.</p>"
            f"<p><a href='{settings.FRONTEND_URL}/app'>Return to Dashboard</a></p></body></html>",
        )
    else:
        return HTMLResponse(
            content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            "<h2>Could not cancel post</h2>"
            "<p>The post may have already been published or cancelled.</p>"
            f"<p><a href='{settings.FRONTEND_URL}/app'>Return to Dashboard</a></p></body></html>",
            status_code=400,
        )


@router.get("/posts/{post_id}/approve-email")
async def approve_and_publish_via_email(
    post_id: int,
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """One-click approve & publish from email link (JWT-authenticated, no login needed)."""
    from fastapi.responses import HTMLResponse
    from app.services.post_scheduler_service import post_scheduler_service
    from app.services.social_posting_service import social_posting_service

    verified_post_id = post_scheduler_service.verify_approve_token(token)
    if verified_post_id is None or verified_post_id != post_id:
        return HTMLResponse(
            content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            "<h2 style='color:#dc2626;'>Invalid or expired approval link.</h2>"
            "<p>Please log in to the admin dashboard to manage posts.</p></body></html>",
            status_code=400,
        )

    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        return HTMLResponse(
            content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            "<h2>Post not found.</h2></body></html>",
            status_code=404,
        )

    if post.status == "published":
        return HTMLResponse(
            content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            "<h2 style='color:#0ea5e9;'>Already Published</h2>"
            "<p>This reply has already been posted.</p>"
            f"<p><a href='{settings.FRONTEND_URL}/app'>Return to Dashboard</a></p></body></html>",
        )

    if post.status not in ("draft", "approved"):
        return HTMLResponse(
            content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            f"<h2>Cannot publish post with status '{post.status}'.</h2>"
            f"<p><a href='{settings.FRONTEND_URL}/app'>Return to Dashboard</a></p></body></html>",
            status_code=400,
        )

    # Approve + publish immediately
    post.status = "approved"
    post.reviewed_at = datetime.utcnow()
    post.reviewed_by = "email_approval"
    await db.commit()

    try:
        pub_result = await social_posting_service.publish_post(post)
        await db.commit()

        if "error" in pub_result:
            return HTMLResponse(
                content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
                f"<h2 style='color:#dc2626;'>Publish Failed</h2>"
                f"<p>{pub_result['error']}</p>"
                f"<p><a href='{settings.FRONTEND_URL}/app'>Return to Dashboard</a></p></body></html>",
                status_code=500,
            )

        username = getattr(post, "reply_to_username", "") or ""
        return HTMLResponse(
            content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            "<h2 style='color:#059669;'>Reply Posted!</h2>"
            f"<p>Your reply to @{username} has been published to Twitter/X.</p>"
            f"<p><a href='{settings.FRONTEND_URL}/app'>Return to Dashboard</a></p></body></html>",
        )
    except Exception as e:
        return HTMLResponse(
            content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            f"<h2 style='color:#dc2626;'>Error</h2>"
            f"<p>Something went wrong. Please try from the dashboard.</p>"
            f"<p><a href='{settings.FRONTEND_URL}/app'>Return to Dashboard</a></p></body></html>",
            status_code=500,
        )


def _edit_page_html(post_id: int, token: str, text: str, platforms_note: str, when_note: str, err: str = "") -> str:
    """The tokenized edit form (no login) — tweak an autopost's copy before it publishes."""
    import html as _h
    action = f"https://api.rigacap.com/api/admin/social/posts/{post_id}/edit-email?token={token}"
    err_html = (f"<p style='color:#8F2D3D;font-size:14px;margin:0 0 10px;'>{_h.escape(err)}</p>" if err else "")
    return (
        "<html><body style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "background:#F5F1E8;color:#141210;padding:40px 20px;max-width:640px;margin:0 auto;\">"
        "<h2 style='color:#7A2430;margin:0 0 6px;'>Edit this post</h2>"
        f"<p style='color:#5A544E;font-size:14px;margin:0 0 4px;'>Posts to {platforms_note} {when_note}.</p>"
        "<p style='color:#5A544E;font-size:14px;margin:0 0 16px;'>Tweak the wording and save — the copy "
        "(and the Instagram card) update on every platform. X caps at 280 characters.</p>"
        f"{err_html}"
        f"<form method='post' action='{action}'>"
        f"<textarea name='text' rows='9' style='width:100%;font-size:16px;line-height:1.5;padding:14px;"
        "border:1px solid #DDD5C7;border-radius:10px;background:#FAF7F0;color:#141210;box-sizing:border-box;'>"
        f"{_h.escape(text)}</textarea>"
        "<button type='submit' style='background:#7A2430;color:#fff;padding:11px 24px;border:none;"
        "border-radius:8px;font-size:15px;font-weight:600;margin-top:14px;cursor:pointer;'>Save changes</button>"
        "</form></body></html>"
    )


async def _edit_load(post_id, token, db):
    """Shared: verify token + load the target post + its same-scheduled_for siblings."""
    from app.services.post_scheduler_service import post_scheduler_service
    if post_scheduler_service.verify_cancel_token(token) != post_id:
        return None, None
    post = (await db.execute(select(SocialPost).where(SocialPost.id == post_id))).scalar_one_or_none()
    if not post:
        return None, None
    sibs = []
    if post.post_type == "research_insight" and post.scheduled_for:
        sibs = (await db.execute(
            select(SocialPost).where(
                SocialPost.post_type == "research_insight",
                SocialPost.scheduled_for == post.scheduled_for,
                SocialPost.status.in_(["scheduled", "approved", "draft"]),
            )
        )).scalars().all()
    if post not in sibs:
        sibs = list(sibs) + [post]
    return post, sibs


def _when_platforms(post, sibs):
    names = {"twitter": "X", "threads": "Threads", "instagram": "Instagram"}
    plats = sorted({s.platform for s in sibs}, key=lambda p: ["twitter", "threads", "instagram"].index(p) if p in ("twitter", "threads", "instagram") else 9)
    labs = [names.get(p, p.title()) for p in plats]
    plat_note = " & ".join([", ".join(labs[:-1]), labs[-1]]) if len(labs) > 1 else (labs[0] if labs else "your feed")
    when_note = ""
    if post.scheduled_for:
        try:
            from pytz import timezone as _tz
            et = post.scheduled_for.replace(tzinfo=_tz("UTC")).astimezone(_tz("US/Eastern"))
            when_note = et.strftime("at %-I:%M %p ET on %b %-d")
        except Exception:
            when_note = "as scheduled"
    return plat_note, when_note


@router.get("/posts/{post_id}/edit-email")
async def edit_post_form(post_id: int, token: str, db: AsyncSession = Depends(get_db)):
    """One-tap from the heads-up email: show a form to tweak the autopost copy before it posts."""
    from fastapi.responses import HTMLResponse
    post, sibs = await _edit_load(post_id, token, db)
    if not post:
        return HTMLResponse("<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
                            "<h2 style='color:#dc2626;'>Invalid or expired link.</h2></body></html>", status_code=400)
    if post.status not in ("scheduled", "approved", "draft"):
        return HTMLResponse("<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
                            "<h2>This post already went out (or was cancelled) — nothing to edit.</h2></body></html>")
    plat_note, when_note = _when_platforms(post, sibs)
    return HTMLResponse(_edit_page_html(post_id, token, post.text_content or "", plat_note, when_note))


@router.post("/posts/{post_id}/edit-email")
async def edit_post_submit(post_id: int, token: str, request: Request, db: AsyncSession = Depends(get_db)):
    """Save the tweaked copy to the post + all its platform siblings (IG card regenerates
    from this text at publish time, so no manual card rebuild needed).

    Parses the urlencoded form body manually (avoids the python-multipart dependency that
    fastapi.Form requires — not in the Lambda image)."""
    from fastapi.responses import HTMLResponse
    from urllib.parse import parse_qs
    raw = (await request.body()).decode("utf-8", "ignore")
    text = (parse_qs(raw).get("text", [""]) or [""])[0]
    post, sibs = await _edit_load(post_id, token, db)
    if not post:
        return HTMLResponse("<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
                            "<h2 style='color:#dc2626;'>Invalid or expired link.</h2></body></html>", status_code=400)
    plat_note, when_note = _when_platforms(post, sibs)
    new_text = (text or "").strip()
    if len(new_text) < 15:
        return HTMLResponse(_edit_page_html(post_id, token, new_text, plat_note, when_note,
                                            err="Too short — give it at least a sentence."), status_code=400)
    if len(new_text) > 280:
        return HTMLResponse(_edit_page_html(post_id, token, new_text, plat_note, when_note,
                                            err=f"That's {len(new_text)} characters — X caps at 280. Trim a little."), status_code=400)
    for s in sibs:
        if s.status in ("scheduled", "approved", "draft"):
            s.text_content = new_text
            s.image_s3_key = None  # force the IG card to regenerate from the new text at publish
    await db.commit()
    return HTMLResponse(
        "<html><body style=\"font-family:-apple-system,sans-serif;text-align:center;padding:60px 24px;"
        "background:#F5F1E8;color:#141210;\"><h2 style='color:#059669;'>Saved</h2>"
        f"<p style='color:#5A544E;'>Updated on {plat_note}. It’ll post {when_note} with your new wording.</p>"
        "</body></html>"
    )


@router.get("/posts/{post_id}/compose-email")
async def compose_reply_via_email(
    post_id: int,
    token: str,
    force: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """One-tap from the approval email: mark the draft handled, then redirect to the X
    composer PRE-FILLED as a reply to the target tweet with our drafted text. No API write
    (avoids the Free-tier 'can only reply where mentioned/author' 403) — Erik just hits Post.

    If the reply was already opened once (reviewed_by='deeplink_compose') or is already
    published, a repeat click shows an "already opened" interstitial (with an explicit
    force link) instead of silently reopening the composer — prevents accidental double-posts.
    """
    import html as _html
    from fastapi.responses import RedirectResponse, HTMLResponse
    from urllib.parse import quote
    from app.services.post_scheduler_service import post_scheduler_service

    verified_post_id = post_scheduler_service.verify_approve_token(token)
    if verified_post_id is None or verified_post_id != post_id:
        return HTMLResponse(
            content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            "<h2 style='color:#dc2626;'>Invalid or expired link.</h2></body></html>",
            status_code=400,
        )

    result = await db.execute(select(SocialPost).where(SocialPost.id == post_id))
    post = result.scalar_one_or_none()
    if not post:
        return HTMLResponse(
            content="<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
            "<h2>Post not found.</h2></body></html>",
            status_code=404,
        )

    text = post.text_content or ""
    target_id = post.reply_to_tweet_id or post.reply_to_thread_id
    platform = (post.platform or "twitter").lower()

    # Already handled once? Don't silently reopen a fresh composer (double-post risk).
    already = (post.reviewed_by == "deeplink_compose") or (post.status in ("published", "posted"))
    if already and not force:
        when = ""
        if post.reviewed_at:
            try:
                from pytz import timezone as _tz
                et = post.reviewed_at.replace(tzinfo=_tz("UTC")).astimezone(_tz("US/Eastern"))
                when = et.strftime(" on %b %-d at %-I:%M %p ET")
            except Exception:
                when = ""
        uname = getattr(post, "reply_to_username", "") or ""
        again_url = f"https://api.rigacap.com/api/admin/social/posts/{post_id}/compose-email?token={token}&force=1"
        return HTMLResponse(
            content=(
                "<html><body style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
                "text-align:center;padding:60px 24px;color:#141210;background:#F5F1E8;\">"
                "<h2 style='color:#7A2430;margin:0 0 10px;'>You already opened this reply</h2>"
                f"<p style='color:#5A544E;font-size:15px;line-height:1.5;'>Reply to @{_html.escape(uname)}{when}. "
                "If you already posted it on X, you're all set &mdash; just close this tab.</p>"
                f"<p style='margin-top:28px;'><a href='{again_url}' style='display:inline-block;background:#7A2430;"
                "color:#fff;padding:11px 24px;border-radius:8px;text-decoration:none;font-weight:600;'>"
                "Open the composer again</a></p>"
                "<p style='color:#8A8279;font-size:12px;margin-top:14px;'>Only do this if it didn't post the first time.</p>"
                "</body></html>"
            ),
        )

    if platform == "twitter" and target_id:
        # X Web Intent: opens the composer as a reply to that tweet, text pre-filled.
        intent_url = f"https://twitter.com/intent/tweet?in_reply_to={target_id}&text={quote(text)}"
    elif platform == "threads":
        # Threads has no reliable pre-filled-reply intent; open the composer with our text
        # so Erik can attach it to the thread manually.
        intent_url = f"https://www.threads.net/intent/post?text={quote(text)}"
    else:
        intent_url = f"https://twitter.com/intent/tweet?text={quote(text)}"

    # Record that Erik acted on it (not "published" — the actual Post happens in the X app).
    if post.status in ("draft", "approved"):
        post.status = "approved"
        post.reviewed_at = datetime.utcnow()
        post.reviewed_by = "deeplink_compose"
        await db.commit()

    return RedirectResponse(url=intent_url, status_code=302)


@router.post("/posts/{post_id}/regenerate-ai")
async def regenerate_post_ai(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    _admin=Depends(get_admin_user),
):
    """Re-generate content via Claude API (instead of template re-roll)."""
    result = await db.execute(
        select(SocialPost).where(SocialPost.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if not post.source_trade_json:
        raise HTTPException(status_code=400, detail="No source trade data for AI regeneration")

    from app.services.ai_content_service import ai_content_service

    new_text = await ai_content_service.regenerate_post(post)
    if new_text:
        post.text_content = new_text
        post.ai_generated = True
        post.ai_model = "claude-sonnet-4-6"
        post.status = "draft"
        post.reviewed_at = None
        post.reviewed_by = None

        # Also regenerate chart card for Instagram posts
        image_url = None
        if post.platform == "instagram" and post.image_metadata_json:
            try:
                from app.services.chart_card_generator import chart_card_generator
                meta = json.loads(post.image_metadata_json)
                png_bytes = chart_card_generator.generate_trade_card(
                    symbol=meta.get("symbol", "???"),
                    entry_price=meta.get("entry_price", 0),
                    exit_price=meta.get("exit_price", 0),
                    entry_date=meta.get("entry_date", ""),
                    exit_date=meta.get("exit_date", ""),
                    pnl_pct=meta.get("pnl_pct", 0),
                    pnl_dollars=meta.get("pnl_dollars", 0),
                    exit_reason=meta.get("exit_reason", "trailing_stop"),
                    strategy_name=meta.get("strategy_name", "Ensemble"),
                    regime_name=meta.get("regime_name", ""),
                    company_name=meta.get("company_name", ""),
                )
                date_str = meta.get("exit_date", "")[:10].replace("-", "")
                s3_key = chart_card_generator.upload_to_s3(
                    png_bytes, post.id, meta.get("symbol", "UNK"), date_str
                )
                if s3_key:
                    post.image_s3_key = s3_key
                    image_url = chart_card_generator.get_presigned_url(s3_key)
            except Exception as e:
                logger.warning(f"Chart card regeneration failed for post {post_id}: {e}")

        await db.commit()
        resp = {"status": "regenerated_ai", "post_id": post_id, "text_content": new_text}
        if image_url:
            resp["image_url"] = image_url
        return resp

    # Fall back to template regeneration
    raise HTTPException(status_code=502, detail="AI regeneration failed. Use /regenerate for template-based fallback.")


def _post_to_dict(post: SocialPost, include_source: bool = False) -> dict:
    """Convert a SocialPost to a dict for API response."""
    d = {
        "id": post.id,
        "post_type": post.post_type,
        "platform": post.platform,
        "status": post.status,
        "text_content": post.text_content,
        "hashtags": post.hashtags,
        "image_s3_key": post.image_s3_key,
        "scheduled_for": post.scheduled_for.isoformat() if post.scheduled_for else None,
        "posted_at": post.posted_at.isoformat() if post.posted_at else None,
        "reviewed_by": post.reviewed_by,
        "reviewed_at": post.reviewed_at.isoformat() if post.reviewed_at else None,
        "rejection_reason": post.rejection_reason,
        "ai_generated": getattr(post, "ai_generated", False) or False,
        "ai_model": getattr(post, "ai_model", None),
        "news_context_json": getattr(post, "news_context_json", None),
        "reply_to_tweet_id": getattr(post, "reply_to_tweet_id", None),
        "reply_to_username": getattr(post, "reply_to_username", None),
        "source_tweet_text": getattr(post, "source_tweet_text", None),
        "created_at": post.created_at.isoformat() if post.created_at else None,
        "updated_at": post.updated_at.isoformat() if post.updated_at else None,
    }

    if include_source:
        d["source_simulation_id"] = post.source_simulation_id
        d["source_trade_json"] = post.source_trade_json
        d["source_data_json"] = post.source_data_json
        d["image_metadata_json"] = post.image_metadata_json

    return d
