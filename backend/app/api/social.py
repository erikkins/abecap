"""
Social Media Admin API - Post queue management for admin review/approval.

All endpoints require admin authentication.
"""

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, delete

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
    if post.status not in ("draft", "rejected"):
        raise HTTPException(status_code=400, detail=f"Cannot approve post with status '{post.status}'")

    post.status = "approved"
    post.reviewed_by = admin.email
    post.reviewed_at = datetime.utcnow()
    post.rejection_reason = None
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
    if body.platform not in ("twitter", "instagram"):
        raise HTTPException(status_code=400, detail="Platform must be 'twitter' or 'instagram'")
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
        "created_at": post.created_at.isoformat() if post.created_at else None,
        "updated_at": post.updated_at.isoformat() if post.updated_at else None,
    }

    if include_source:
        d["source_simulation_id"] = post.source_simulation_id
        d["source_trade_json"] = post.source_trade_json
        d["source_data_json"] = post.source_data_json
        d["image_metadata_json"] = post.image_metadata_json

    return d
