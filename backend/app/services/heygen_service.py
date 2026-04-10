"""
HeyGen Video Service - Generate AI avatar videos from trade results.

Uses HeyGen API v2 to create short-form video content featuring an AI avatar
narrating trade results, market commentary, and "we called it" moments.
Integrates with the existing social content pipeline.

API docs: https://docs.heygen.com/reference/create-an-avatar-video-v2
"""

import json
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# HeyGen API v2 endpoints
HEYGEN_BASE_URL = "https://api.heygen.com"
VIDEO_GENERATE_URL = f"{HEYGEN_BASE_URL}/v2/video/generate"
VIDEO_STATUS_URL = f"{HEYGEN_BASE_URL}/v1/video_status.get"
LIST_AVATARS_URL = f"{HEYGEN_BASE_URL}/v2/avatars"
LIST_VOICES_URL = f"{HEYGEN_BASE_URL}/v2/voices"

# Video dimension presets
DIMENSIONS = {
    "landscape": {"width": 1920, "height": 1080},  # 16:9
    "portrait": {"width": 1080, "height": 1920},    # 9:16 (Reels/TikTok/Shorts)
    "square": {"width": 1080, "height": 1080},      # 1:1 (Instagram feed)
}

# Default avatar/voice (override via method params or env vars)
DEFAULT_AVATAR_ID = os.getenv("HEYGEN_DEFAULT_AVATAR_ID", "")
DEFAULT_VOICE_ID = os.getenv("HEYGEN_DEFAULT_VOICE_ID", "")


class HeyGenService:
    """Generate AI avatar videos using HeyGen API v2."""

    def __init__(self):
        self.api_key = os.getenv("HEYGEN_API_KEY", "")
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logger.warning("HeyGen service disabled - HEYGEN_API_KEY not configured")

    def _headers(self) -> dict:
        return {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

    async def list_avatars(self) -> Optional[list]:
        """
        List all available avatars from HeyGen.

        Returns list of avatar dicts with avatar_id, avatar_name, gender, preview_url, etc.
        """
        if not self.enabled:
            return None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(LIST_AVATARS_URL, headers=self._headers())

            if resp.status_code != 200:
                logger.error(f"HeyGen list avatars error {resp.status_code}: {resp.text}")
                return None

            data = resp.json()
            avatars = data.get("data", {}).get("avatars", [])
            logger.info(f"HeyGen: found {len(avatars)} avatars")
            return avatars

        except Exception as e:
            logger.error(f"HeyGen list avatars failed: {e}")
            return None

    async def list_voices(self) -> Optional[list]:
        """
        List all available voices from HeyGen.

        Returns list of voice dicts with voice_id, name, language, gender, preview_url, etc.
        """
        if not self.enabled:
            return None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(LIST_VOICES_URL, headers=self._headers())

            if resp.status_code != 200:
                logger.error(f"HeyGen list voices error {resp.status_code}: {resp.text}")
                return None

            data = resp.json()
            voices = data.get("data", {}).get("voices", [])
            logger.info(f"HeyGen: found {len(voices)} voices")
            return voices

        except Exception as e:
            logger.error(f"HeyGen list voices failed: {e}")
            return None

    async def create_video(
        self,
        script: str,
        avatar_id: Optional[str] = None,
        voice_id: Optional[str] = None,
        format: str = "portrait",
        test: bool = False,
        caption: bool = True,
        background_color: str = "#172554",
    ) -> Optional[str]:
        """
        Create a video with an AI avatar speaking the given script.

        Args:
            script: Text for the avatar to speak.
            avatar_id: HeyGen avatar ID. Falls back to HEYGEN_DEFAULT_AVATAR_ID env var.
            voice_id: HeyGen voice ID. Falls back to HEYGEN_DEFAULT_VOICE_ID env var.
            format: "portrait" (9:16), "landscape" (16:9), or "square" (1:1).
            test: If True, generates a lower-quality test video (free/cheaper).
            caption: Whether to add captions/subtitles.
            background_color: Hex color for the background. Default is RigaCap navy.

        Returns:
            video_id string if successfully queued, or None on failure.
        """
        if not self.enabled:
            logger.error("HeyGen service not enabled")
            return None

        avatar_id = avatar_id or DEFAULT_AVATAR_ID
        voice_id = voice_id or DEFAULT_VOICE_ID

        if not avatar_id or not voice_id:
            logger.error("HeyGen: avatar_id and voice_id are required. "
                         "Set HEYGEN_DEFAULT_AVATAR_ID and HEYGEN_DEFAULT_VOICE_ID env vars "
                         "or pass them explicitly.")
            return None

        dimension = DIMENSIONS.get(format, DIMENSIONS["portrait"])

        payload = {
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        "avatar_id": avatar_id,
                        "avatar_style": "normal",
                    },
                    "voice": {
                        "type": "text",
                        "input_text": script,
                        "voice_id": voice_id,
                    },
                    "background": {
                        "type": "color",
                        "value": background_color,
                    },
                }
            ],
            "dimension": dimension,
            "test": test,
            "caption": caption,
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    VIDEO_GENERATE_URL,
                    headers=self._headers(),
                    json=payload,
                )

            if resp.status_code != 200:
                logger.error(f"HeyGen create video error {resp.status_code}: {resp.text}")
                return None

            data = resp.json()
            video_id = data.get("data", {}).get("video_id")
            if video_id:
                logger.info(f"HeyGen: video queued, video_id={video_id}")
            else:
                logger.error(f"HeyGen: no video_id in response: {data}")
            return video_id

        except Exception as e:
            logger.error(f"HeyGen create video failed: {e}")
            return None

    async def get_video_status(self, video_id: str) -> Optional[dict]:
        """
        Check the status of a video generation job.

        Returns dict with keys:
            - status: "processing", "completed", "failed", "pending"
            - video_url: Download URL (only when status == "completed")
            - error: Error message (only when status == "failed")

        The video_url contains temporary signed parameters and refreshes
        each time you call this endpoint.
        """
        if not self.enabled:
            return None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    VIDEO_STATUS_URL,
                    headers=self._headers(),
                    params={"video_id": video_id},
                )

            if resp.status_code != 200:
                logger.error(f"HeyGen video status error {resp.status_code}: {resp.text}")
                return None

            data = resp.json()
            inner = data.get("data", {})
            status = inner.get("status", "unknown")
            result = {"status": status, "video_id": video_id}

            if status == "completed":
                result["video_url"] = inner.get("video_url")
                result["duration"] = inner.get("duration")
                result["thumbnail_url"] = inner.get("thumbnail_url")
            elif status == "failed":
                result["error"] = inner.get("error", "Unknown error")

            return result

        except Exception as e:
            logger.error(f"HeyGen video status check failed: {e}")
            return None

    async def generate_trade_video(
        self,
        trade_data: dict,
        format: str = "portrait",
        avatar_id: Optional[str] = None,
        voice_id: Optional[str] = None,
        test: bool = False,
    ) -> Optional[dict]:
        """
        Generate a video from trade result data.

        Takes the same trade_data dict used by ai_content_service (symbol, entry_price,
        exit_price, pnl_pct, entry_date, exit_date, exit_reason) and builds a narration
        script, then queues a HeyGen video.

        Args:
            trade_data: Dict with trade fields (symbol, pnl_pct, entry_price, exit_price, etc.)
            format: Video format - "portrait", "landscape", or "square".
            avatar_id: Override default avatar.
            voice_id: Override default voice.
            test: Generate test-quality video.

        Returns:
            Dict with video_id and script, or None on failure.
        """
        script = self._build_trade_script(trade_data)
        if not script:
            return None

        video_id = await self.create_video(
            script=script,
            avatar_id=avatar_id,
            voice_id=voice_id,
            format=format,
            test=test,
        )

        if not video_id:
            return None

        return {
            "video_id": video_id,
            "script": script,
            "trade_data": trade_data,
            "format": format,
        }

    @staticmethod
    def _build_trade_script(trade_data: dict) -> Optional[str]:
        """
        Build a narration script from trade result data.

        Generates a concise, confident script matching RigaCap's voice:
        witty, data-driven, never financial advice.
        """
        symbol = trade_data.get("symbol")
        pnl_pct = trade_data.get("pnl_pct")
        entry_price = trade_data.get("entry_price")
        exit_price = trade_data.get("exit_price")
        entry_date = str(trade_data.get("entry_date", ""))[:10]
        exit_date = str(trade_data.get("exit_date", ""))[:10]
        exit_reason = trade_data.get("exit_reason", "trailing_stop")

        if not symbol or pnl_pct is None:
            logger.error("Trade data missing required fields (symbol, pnl_pct)")
            return None

        # Build script based on outcome
        if pnl_pct > 0:
            script = (
                f"Our system flagged {symbol} on {entry_date} at {entry_price:.2f} dollars. "
                f"We exited on {exit_date} at {exit_price:.2f} for a {pnl_pct:+.1f}% gain. "
            )
            if exit_reason == "trailing_stop":
                script += "The trailing stop locked in profits on the way up. "
            elif exit_reason == "market_regime":
                script += "Our market regime filter signaled caution, so we took profits early. "

            script += (
                "No luck involved. This is algorithmic momentum, walk-forward validated "
                "across five years of live market data. "
                "See our full track record at rigacap.com."
            )
        else:
            script = (
                f"Transparency matters. Our system entered {symbol} on {entry_date} "
                f"at {entry_price:.2f} and exited on {exit_date} at {exit_price:.2f} "
                f"for a {pnl_pct:+.1f}% loss. "
            )
            if exit_reason == "trailing_stop":
                script += "The trailing stop did its job, limiting the downside. "
            elif exit_reason == "market_regime":
                script += "Our regime detector caught the shift and cut the position. "

            script += (
                "Not every trade wins. But our system manages risk so one bad trade "
                "doesn't derail the portfolio. Check our verified track record at rigacap.com."
            )

        return script


# Singleton instance
heygen_service = HeyGenService()
