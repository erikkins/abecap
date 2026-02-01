"""Cloudflare Turnstile verification service."""

import httpx
from app.core.config import settings


async def verify_turnstile(token: str, remote_ip: str = None) -> bool:
    """
    Verify a Cloudflare Turnstile token.

    Args:
        token: The turnstile response token from the client
        remote_ip: Optional client IP address for additional validation

    Returns:
        True if verification succeeds, False otherwise
    """
    if not settings.TURNSTILE_SECRET_KEY:
        # Skip verification in development if not configured
        return True

    try:
        async with httpx.AsyncClient() as client:
            data = {
                "secret": settings.TURNSTILE_SECRET_KEY,
                "response": token,
            }

            if remote_ip:
                data["remoteip"] = remote_ip

            response = await client.post(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data=data,
                timeout=10.0
            )

            result = response.json()
            return result.get("success", False)

    except Exception as e:
        print(f"Turnstile verification error: {e}")
        return False
