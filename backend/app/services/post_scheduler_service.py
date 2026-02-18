"""
Post Scheduler Service - Automated scheduling, admin notifications, and auto-publishing.

Handles the full lifecycle:
1. Auto-schedule draft posts across optimal posting windows
2. Send T-24h and T-1h admin approval notifications with cancel links
3. Auto-publish approved posts when scheduled_for <= now
4. Cancel mechanism via JWT-signed one-click email links
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from jose import jwt, JWTError, ExpiredSignatureError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.core.config import settings
from app.core.database import SocialPost

logger = logging.getLogger(__name__)

# Admin email
ADMIN_EMAIL = "erik@rigacap.com"

# Optimal posting hours (ET) by day type
OPTIMAL_HOURS = {
    "weekday": [9, 12, 17],   # Morning, lunch, after-work
    "weekend": [10, 14],       # Mid-morning, afternoon
}


class PostSchedulerService:
    """Schedule posts and send admin approval notifications."""

    async def schedule_post(
        self, post_id: int, publish_at: datetime, db: AsyncSession
    ) -> bool:
        """
        Set scheduled_for on a SocialPost. Status must be 'approved'.

        Returns True if successfully scheduled.
        """
        result = await db.execute(
            select(SocialPost).where(SocialPost.id == post_id)
        )
        post = result.scalar_one_or_none()

        if not post:
            logger.error(f"Post {post_id} not found")
            return False

        if post.status not in ("approved", "draft"):
            logger.error(f"Cannot schedule post {post_id} with status '{post.status}'")
            return False

        post.scheduled_for = publish_at
        if post.status == "draft":
            post.status = "approved"
        await db.commit()

        logger.info(f"Post {post_id} scheduled for {publish_at.isoformat()}")
        return True

    async def auto_schedule_drafts(self, db: AsyncSession) -> int:
        """
        Called nightly after content generation.
        Takes all new draft posts (without scheduled_for), assigns scheduled_for times
        spread across the next 1-3 days at optimal posting hours.

        Returns count of posts scheduled.
        """
        result = await db.execute(
            select(SocialPost).where(
                and_(
                    SocialPost.status == "draft",
                    SocialPost.scheduled_for.is_(None),
                )
            ).order_by(SocialPost.created_at)
        )
        posts = result.scalars().all()

        if not posts:
            return 0

        now = datetime.utcnow()
        scheduled_count = 0

        for i, post in enumerate(posts):
            # Spread posts across 1-3 days
            day_offset = 1 + (i // 4)  # 4 posts per day max
            target_date = now + timedelta(days=day_offset)

            # Pick optimal hour based on weekday/weekend
            is_weekend = target_date.weekday() >= 5
            hours = OPTIMAL_HOURS["weekend" if is_weekend else "weekday"]
            hour = hours[i % len(hours)]

            publish_at = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)

            post.scheduled_for = publish_at
            scheduled_count += 1

        await db.commit()

        logger.info(f"Auto-scheduled {scheduled_count} draft posts")

        # Send T-24h notification for posts scheduled within next 48h
        upcoming = [p for p in posts if p.scheduled_for and p.scheduled_for <= now + timedelta(hours=48)]
        if upcoming:
            await self._send_batch_notification(upcoming, hours_before=24)

        return scheduled_count

    async def check_and_publish(self, db: AsyncSession) -> int:
        """
        Called every 15 minutes by scheduler.
        Finds posts where status='approved' AND scheduled_for <= now.
        Publishes them via SocialPostingService.

        Returns count of posts published.
        """
        now = datetime.utcnow()

        result = await db.execute(
            select(SocialPost).where(
                and_(
                    SocialPost.status == "approved",
                    SocialPost.scheduled_for.isnot(None),
                    SocialPost.scheduled_for <= now,
                )
            )
        )
        posts = result.scalars().all()

        if not posts:
            return 0

        from app.services.social_posting_service import social_posting_service

        published = 0
        for post in posts:
            try:
                pub_result = await social_posting_service.publish_post(post)
                if "error" not in pub_result:
                    published += 1
                    logger.info(f"Auto-published post {post.id} to {post.platform}")
                else:
                    logger.error(f"Auto-publish failed for post {post.id}: {pub_result['error']}")
            except Exception as e:
                logger.error(f"Auto-publish error for post {post.id}: {e}")

        await db.commit()

        if published:
            logger.info(f"Auto-published {published}/{len(posts)} scheduled posts")

        return published

    async def send_notifications(self, db: AsyncSession) -> int:
        """
        Called every hour by scheduler.
        Sends T-24h and T-1h admin notifications for upcoming scheduled posts.

        Returns count of notifications sent.
        """
        now = datetime.utcnow()
        sent = 0

        # T-24h notifications: posts scheduled in 23-25 hours
        result_24h = await db.execute(
            select(SocialPost).where(
                and_(
                    SocialPost.status == "approved",
                    SocialPost.scheduled_for.isnot(None),
                    SocialPost.scheduled_for > now + timedelta(hours=23),
                    SocialPost.scheduled_for <= now + timedelta(hours=25),
                    SocialPost.notification_24h_sent == False,
                )
            )
        )
        posts_24h = result_24h.scalars().all()

        for post in posts_24h:
            success = await self._send_notification(post, hours_before=24)
            if success:
                post.notification_24h_sent = True
                sent += 1

        # T-1h notifications: posts scheduled in 30min-90min
        result_1h = await db.execute(
            select(SocialPost).where(
                and_(
                    SocialPost.status == "approved",
                    SocialPost.scheduled_for.isnot(None),
                    SocialPost.scheduled_for > now + timedelta(minutes=30),
                    SocialPost.scheduled_for <= now + timedelta(minutes=90),
                    SocialPost.notification_1h_sent == False,
                )
            )
        )
        posts_1h = result_1h.scalars().all()

        for post in posts_1h:
            success = await self._send_notification(post, hours_before=1)
            if success:
                post.notification_1h_sent = True
                sent += 1

        if sent:
            await db.commit()
            logger.info(f"Sent {sent} post notifications")

        return sent

    async def cancel_post(self, post_id: int, db: AsyncSession) -> bool:
        """Admin cancels a scheduled post. Sets status='cancelled'."""
        result = await db.execute(
            select(SocialPost).where(SocialPost.id == post_id)
        )
        post = result.scalar_one_or_none()

        if not post:
            return False

        if post.status in ("posted", "cancelled"):
            return False

        post.status = "cancelled"
        await db.commit()

        logger.info(f"Post {post_id} cancelled")
        return True

    def generate_cancel_token(self, post_id: int, expires_hours: int = 48) -> str:
        """Generate a JWT token for one-click cancel from email."""
        payload = {
            "post_id": post_id,
            "action": "cancel",
            "exp": datetime.utcnow() + timedelta(hours=expires_hours),
        }
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")

    def verify_cancel_token(self, token: str) -> Optional[int]:
        """
        Verify a cancel JWT token and return the post_id.
        Returns None if invalid or expired.
        """
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
            if payload.get("action") != "cancel":
                return None
            return payload.get("post_id")
        except ExpiredSignatureError:
            logger.warning("Cancel token expired")
            return None
        except JWTError:
            logger.warning("Invalid cancel token")
            return None

    async def _send_notification(self, post: SocialPost, hours_before: int) -> bool:
        """Send email notification for a single post."""
        try:
            from app.services.email_service import admin_email_service

            cancel_token = self.generate_cancel_token(post.id)
            cancel_url = f"{settings.FRONTEND_URL}/api/admin/social/posts/{post.id}/cancel-email?token={cancel_token}"

            return await admin_email_service.send_post_approval_notification(
                to_email=ADMIN_EMAIL,
                post=post,
                hours_before=hours_before,
                cancel_url=cancel_url,
            )
        except Exception as e:
            logger.error(f"Failed to send notification for post {post.id}: {e}")
            return False

    async def _send_batch_notification(
        self, posts: list, hours_before: int
    ) -> int:
        """Send notification for a batch of posts."""
        sent = 0
        for post in posts:
            if await self._send_notification(post, hours_before):
                sent += 1
        return sent


# Singleton instance
post_scheduler_service = PostSchedulerService()
