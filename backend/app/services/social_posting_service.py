"""
Social media publishing service — posts to Twitter API v2 and Instagram Graph API.

Uses httpx (already in requirements.txt) for all HTTP requests.
OAuth 1.0a signing for Twitter is done manually (no extra dependency).
"""

import hashlib
import hmac
import logging
import os
import time
import urllib.parse
import uuid
from base64 import b64encode
from datetime import datetime
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class SocialPostingService:
    """Publish posts to Twitter and Instagram."""

    # Twitter API v2 endpoints
    TWITTER_TWEET_URL = "https://api.twitter.com/2/tweets"
    TWITTER_MEDIA_UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"

    # Instagram Graph API
    INSTAGRAM_API_BASE = "https://graph.facebook.com/v24.0"

    # ── Twitter ──────────────────────────────────────────────────────

    def _oauth1_signature(
        self, method: str, url: str, params: dict, body_params: dict = None
    ) -> str:
        """Generate OAuth 1.0a Authorization header for Twitter."""
        consumer_key = settings.TWITTER_API_KEY
        consumer_secret = settings.TWITTER_API_SECRET
        token = settings.TWITTER_ACCESS_TOKEN
        token_secret = settings.TWITTER_ACCESS_TOKEN_SECRET

        oauth_params = {
            "oauth_consumer_key": consumer_key,
            "oauth_nonce": uuid.uuid4().hex,
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_token": token,
            "oauth_version": "1.0",
        }

        # Combine all params for signature base string
        all_params = {**oauth_params, **params}
        if body_params:
            all_params.update(body_params)

        sorted_params = "&".join(
            f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(str(v), safe='')}"
            for k, v in sorted(all_params.items())
        )

        base_string = (
            f"{method.upper()}&"
            f"{urllib.parse.quote(url, safe='')}&"
            f"{urllib.parse.quote(sorted_params, safe='')}"
        )

        signing_key = (
            f"{urllib.parse.quote(consumer_secret, safe='')}&"
            f"{urllib.parse.quote(token_secret, safe='')}"
        )

        signature = b64encode(
            hmac.new(
                signing_key.encode(), base_string.encode(), hashlib.sha1
            ).digest()
        ).decode()

        oauth_params["oauth_signature"] = signature

        auth_header = "OAuth " + ", ".join(
            f'{urllib.parse.quote(k, safe="")}="{urllib.parse.quote(v, safe="")}"'
            for k, v in sorted(oauth_params.items())
        )
        return auth_header

    async def _upload_media_to_twitter(self, image_bytes: bytes) -> Optional[str]:
        """Upload an image to Twitter and return the media_id string."""
        auth_header = self._oauth1_signature(
            "POST", self.TWITTER_MEDIA_UPLOAD_URL, {}
        )

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.TWITTER_MEDIA_UPLOAD_URL,
                headers={"Authorization": auth_header},
                files={"media_data": ("image.png", image_bytes, "image/png")},
            )

        if resp.status_code not in (200, 201):
            logger.error("Twitter media upload failed: %s %s", resp.status_code, resp.text)
            return None

        media_id = resp.json().get("media_id_string")
        logger.info("Twitter media uploaded: %s", media_id)
        return media_id

    async def post_to_twitter(
        self, text: str, image_url: Optional[str] = None,
        reply_to_tweet_id: Optional[str] = None,
    ) -> dict:
        """Post a tweet. Optionally attach an image (downloaded from image_url).

        Returns {"tweet_id": "...", "tweet_url": "..."} on success,
        or {"error": "..."} on failure.
        """
        if not settings.TWITTER_API_KEY:
            return {"error": "Twitter API credentials not configured"}

        media_id = None
        if image_url:
            async with httpx.AsyncClient(timeout=30) as client:
                img_resp = await client.get(image_url)
            if img_resp.status_code == 200:
                media_id = await self._upload_media_to_twitter(img_resp.content)

        payload = {"text": text}
        if media_id:
            payload["media"] = {"media_ids": [media_id]}
        if reply_to_tweet_id:
            payload["reply"] = {"in_reply_to_tweet_id": reply_to_tweet_id}

        # Twitter API v2 uses JSON body — OAuth 1.0a signature only covers URL params
        auth_header = self._oauth1_signature("POST", self.TWITTER_TWEET_URL, {})

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.TWITTER_TWEET_URL,
                headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if resp.status_code not in (200, 201):
            logger.error("Twitter post failed: %s %s", resp.status_code, resp.text)
            return {"error": f"Twitter API error {resp.status_code}: {resp.text}"}

        data = resp.json().get("data", {})
        tweet_id = data.get("id", "")
        return {
            "tweet_id": tweet_id,
            "tweet_url": f"https://x.com/rigacap/status/{tweet_id}",
        }

    # ── Instagram ────────────────────────────────────────────────────

    async def post_to_instagram(
        self, caption: str, image_url: str
    ) -> dict:
        """Publish a photo to Instagram via the Graph API.

        image_url must be publicly accessible (use S3 presigned URL with long expiry).

        Returns {"media_id": "...", "permalink": "..."} on success,
        or {"error": "..."} on failure.
        """
        if not settings.INSTAGRAM_ACCESS_TOKEN or not settings.INSTAGRAM_BUSINESS_ACCOUNT_ID:
            return {"error": "Instagram API credentials not configured"}

        ig_user_id = settings.INSTAGRAM_BUSINESS_ACCOUNT_ID
        access_token = settings.INSTAGRAM_ACCESS_TOKEN

        async with httpx.AsyncClient(timeout=60) as client:
            # Step 1: Create media container
            container_resp = await client.post(
                f"{self.INSTAGRAM_API_BASE}/{ig_user_id}/media",
                data={
                    "image_url": image_url,
                    "caption": caption,
                    "access_token": access_token,
                },
            )

            if container_resp.status_code != 200:
                logger.error("IG container creation failed: %s %s", container_resp.status_code, container_resp.text)
                return {"error": f"Instagram container error: {container_resp.text}"}

            container_id = container_resp.json().get("id")
            if not container_id:
                return {"error": "No container ID returned"}

            # Step 2: Poll until container is ready (up to 30s)
            for _ in range(6):
                status_resp = await client.get(
                    f"{self.INSTAGRAM_API_BASE}/{container_id}",
                    params={
                        "fields": "status_code",
                        "access_token": access_token,
                    },
                )
                status_code = status_resp.json().get("status_code")
                if status_code == "FINISHED":
                    break
                if status_code == "ERROR":
                    return {"error": "Instagram container processing failed"}
                await _async_sleep(5)

            # Step 3: Publish
            publish_resp = await client.post(
                f"{self.INSTAGRAM_API_BASE}/{ig_user_id}/media_publish",
                data={
                    "creation_id": container_id,
                    "access_token": access_token,
                },
            )

            if publish_resp.status_code != 200:
                logger.error("IG publish failed: %s %s", publish_resp.status_code, publish_resp.text)
                return {"error": f"Instagram publish error: {publish_resp.text}"}

            media_id = publish_resp.json().get("id", "")

            # Get permalink
            permalink = ""
            try:
                perm_resp = await client.get(
                    f"{self.INSTAGRAM_API_BASE}/{media_id}",
                    params={
                        "fields": "permalink",
                        "access_token": access_token,
                    },
                )
                permalink = perm_resp.json().get("permalink", "")
            except Exception:
                pass

            return {"media_id": media_id, "permalink": permalink}

    # ── Twitter Follow ──────────────────────────────────────────────

    TWITTER_USERS_ME_URL = "https://api.twitter.com/2/users/me"
    TWITTER_USERS_BY_USERNAME_URL = "https://api.twitter.com/2/users/by/username"

    async def get_authenticated_user_id(self) -> Optional[str]:
        """Get the Twitter user ID for the authenticated account."""
        auth_header = self._oauth1_signature("GET", self.TWITTER_USERS_ME_URL, {})
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                self.TWITTER_USERS_ME_URL,
                headers={"Authorization": auth_header},
            )
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("id")
        logger.error("Failed to get authenticated user: %s %s", resp.status_code, resp.text)
        return None

    async def lookup_twitter_user_id(self, username: str) -> Optional[str]:
        """Look up a Twitter user ID by username (without @)."""
        url = f"{self.TWITTER_USERS_BY_USERNAME_URL}/{username}"
        auth_header = self._oauth1_signature("GET", url, {})
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                url,
                headers={"Authorization": auth_header},
            )
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("id")
        logger.warning("User lookup failed for @%s: %s %s", username, resp.status_code, resp.text)
        return None

    async def follow_twitter_user(self, target_user_id: str, my_user_id: str = None) -> dict:
        """Follow a Twitter user by their user ID.

        Returns {"following": True, "pending": False} on success,
        or {"error": "..."} on failure.
        """
        if not settings.TWITTER_API_KEY:
            return {"error": "Twitter API credentials not configured"}

        if not my_user_id:
            my_user_id = await self.get_authenticated_user_id()
            if not my_user_id:
                return {"error": "Could not determine authenticated user ID"}

        url = f"https://api.twitter.com/2/users/{my_user_id}/following"
        auth_header = self._oauth1_signature("POST", url, {})

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/json",
                },
                json={"target_user_id": target_user_id},
            )

        if resp.status_code in (200, 201):
            return resp.json().get("data", {})
        logger.error("Twitter follow failed: %s %s", resp.status_code, resp.text)
        return {"error": f"Twitter API error {resp.status_code}: {resp.text}"}

    async def batch_follow_twitter(self, usernames: list[str]) -> dict:
        """Follow a list of Twitter usernames. Returns results per username."""
        import asyncio

        my_user_id = await self.get_authenticated_user_id()
        if not my_user_id:
            return {"error": "Could not determine authenticated user ID"}

        results = {}
        for username in usernames:
            clean = username.lstrip("@").strip()
            if not clean:
                continue
            try:
                target_id = await self.lookup_twitter_user_id(clean)
                if not target_id:
                    results[clean] = {"error": "User not found"}
                    continue
                result = await self.follow_twitter_user(target_id, my_user_id)
                results[clean] = result
                logger.info("Followed @%s: %s", clean, result)
            except Exception as e:
                results[clean] = {"error": str(e)}
                logger.error("Error following @%s: %s", clean, e)
            # Rate limit: 5 requests per 15 min window for follows
            await asyncio.sleep(3)

        return {
            "my_user_id": my_user_id,
            "followed": sum(1 for r in results.values() if r.get("following")),
            "failed": sum(1 for r in results.values() if "error" in r),
            "results": results,
        }

    # ── Unified publish ──────────────────────────────────────────────

    async def publish_post(self, post) -> dict:
        """Publish a SocialPost to its target platform.

        Updates post.status and post.posted_at in-place (caller must commit).
        Returns the platform-specific result dict.
        """
        text = post.text_content or ""
        if post.hashtags:
            text += f"\n\n{post.hashtags}"

        image_url = None
        if post.image_s3_key:
            from app.services.chart_card_generator import chart_card_generator
            # Long expiry for Instagram (needs to download the image)
            image_url = chart_card_generator.get_presigned_url(
                post.image_s3_key, expires_in=3600
            )

        if post.platform == "twitter":
            reply_to = getattr(post, 'reply_to_tweet_id', None)
            result = await self.post_to_twitter(text, image_url, reply_to_tweet_id=reply_to)
        elif post.platform == "instagram":
            if not image_url:
                return {"error": "Instagram posts require an image"}
            result = await self.post_to_instagram(text, image_url)
        else:
            return {"error": f"Unknown platform: {post.platform}"}

        if "error" not in result:
            post.status = "posted"
            post.posted_at = datetime.utcnow()

        return result


async def _async_sleep(seconds: float):
    """Async sleep without importing asyncio at module level."""
    import asyncio
    await asyncio.sleep(seconds)


# Singleton
social_posting_service = SocialPostingService()
