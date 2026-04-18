"""
Social Engagement Opportunities Service

Scans Twitter feeds from curated finance accounts, filters for
topics RigaCap has a strong take on, and generates Claude-drafted
comment suggestions for the founder to post manually.

Daily 9 AM ET delivery via admin email.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Curated finance accounts to monitor. Add/remove as needed.
# Format: (twitter_handle, display_name, why_we_follow)
MONITORED_ACCOUNTS = [
    ("unusual_whales", "Unusual Whales", "options flow + market data"),
    ("TrendSpider", "TrendSpider", "technical analysis platform"),
    ("OptionsPlay", "OptionsPlay", "options strategy education"),
    ("stockabortiond", "The Stock Sniper", "momentum trading"),
    ("FinanceLancelot", "Finance Lancelot", "market commentary"),
    ("CiovaccoCapital", "Ciovacco Capital", "regime/trend analysis"),
    ("AndrewThrasher", "Andrew Thrasher", "market breadth + regimes"),
    ("sentabortiond", "Sentiment Trader", "sentiment/fear analysis"),
    ("LizAnnSonders", "Liz Ann Sonders", "macro market commentary"),
    ("RyanDetrick", "Ryan Detrick", "seasonality + market stats"),
    ("jabortiondmoor7", "Jason Moore", "quant trading"),
    ("Investopedia", "Investopedia", "finance education"),
    ("tastylive", "tastylive", "trading education"),
    ("TheChartGuys", "The Chart Guys", "technical analysis"),
    ("MacroCharts", "Macro Charts", "macro + regime analysis"),
]

# Topics we have strong takes on — used for keyword filtering
TOPIC_KEYWORDS = [
    # Market regime
    "bull market", "bear market", "market regime", "regime change",
    "market crash", "correction", "recovery", "breadth",
    "rotating", "rotation", "sector rotation",
    # Momentum / signals
    "momentum", "breakout", "signal", "buy signal", "sell signal",
    "trailing stop", "stop loss", "risk management",
    # Strategy / backtesting
    "backtest", "walk forward", "walk-forward", "systematic",
    "algorithmic", "quant", "rules-based",
    # Market conditions
    "VIX", "volatility", "fear", "market fear",
    "SPY", "S&P 500", "200-day", "moving average",
    # General themes we can riff on
    "win rate", "drawdown", "max drawdown", "sharpe",
    "position sizing", "portfolio", "concentration",
    "let winners run", "cut losses",
]

# Minimum keyword matches to qualify a tweet as relevant
MIN_KEYWORD_MATCHES = 1


class EngagementService:

    def _get_twitter_headers(self) -> dict:
        """Build OAuth 1.0a headers for Twitter API v2."""
        import hashlib
        import hmac
        import time
        import urllib.parse
        import uuid
        from app.core.config import settings

        # For read-only endpoints, we can use Bearer token (app-only auth)
        # But user timeline requires user context. Use OAuth 1.0a.
        # Actually for reading OTHER users' tweets, app-only (Bearer) works.
        # Let's try Bearer first — simpler.
        return {
            "Authorization": f"Bearer {self._get_bearer_token()}",
        }

    def _get_bearer_token(self) -> str:
        """Get Twitter app-only Bearer token using API key + secret."""
        import base64
        import httpx
        from app.core.config import settings

        key = settings.TWITTER_API_KEY
        secret = settings.TWITTER_API_SECRET
        credentials = base64.b64encode(f"{key}:{secret}".encode()).decode()

        resp = httpx.post(
            "https://api.twitter.com/oauth2/token",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data="grant_type=client_credentials",
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()["access_token"]
        logger.error(f"Bearer token fetch failed: {resp.status_code} {resp.text[:200]}")
        return ""

    def _resolve_user_id(self, handle: str, headers: dict) -> Optional[str]:
        """Resolve a Twitter handle to a user ID."""
        import httpx
        try:
            resp = httpx.get(
                f"https://api.twitter.com/2/users/by/username/{handle}",
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("data", {}).get("id")
        except Exception as e:
            logger.warning(f"Failed to resolve @{handle}: {e}")
        return None

    def _get_user_tweets(self, user_id: str, headers: dict, since_hours: int = 24) -> List[dict]:
        """Get recent tweets from a specific user."""
        import httpx
        since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            resp = httpx.get(
                f"https://api.twitter.com/2/users/{user_id}/tweets",
                headers=headers,
                params={
                    "max_results": 10,
                    "start_time": since,
                    "tweet.fields": "created_at,public_metrics,text",
                    "exclude": "retweets,replies",
                },
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
            elif resp.status_code == 429:
                logger.warning("Twitter rate limit hit")
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch tweets for user {user_id}: {e}")
        return []

    def _score_relevance(self, text: str) -> tuple:
        """Score a tweet's relevance to RigaCap's topics. Returns (score, matched_keywords)."""
        text_lower = text.lower()
        matches = [kw for kw in TOPIC_KEYWORDS if kw.lower() in text_lower]
        return len(matches), matches

    def _generate_comment(self, tweet_text: str, author: str, matched_topics: List[str],
                          market_context: str = "") -> str:
        """Use Claude to draft a suggested comment."""
        import httpx
        from app.core.config import settings

        if not settings.ANTHROPIC_API_KEY:
            return "(Claude API key not available — draft manually)"

        system = (
            "You write short, insightful Twitter replies for RigaCap, an AI-powered stock signal service. "
            "Your tone: knowledgeable, conversational, confident but not salesy. You're adding value to "
            "the conversation, not pitching. Never say 'our algorithm' — you can reference 'our system' "
            "or 'our signals' if relevant, but only if it adds to the point naturally. "
            "NEVER use the word 'tape'. "
            "Most replies should NOT mention RigaCap at all — just share a smart take. "
            "1-2 sentences max. No hashtags. No emojis. Sound like a sharp analyst friend."
        )

        market_note = f"\n\nToday's market context for reference: {market_context}" if market_context else ""

        prompt = (
            f"Write a reply to this tweet by @{author}:\n\n"
            f'"{tweet_text}"\n\n'
            f"Relevant topics: {', '.join(matched_topics)}.{market_note}\n\n"
            f"Draft a 1-2 sentence reply that adds insight. Don't pitch anything."
        )

        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 150,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=15,
            )
            if resp.status_code == 200:
                content = resp.json().get("content", [])
                if content and content[0].get("type") == "text":
                    return content[0]["text"].strip().strip('"')
        except Exception as e:
            logger.warning(f"Claude comment generation failed: {e}")

        return "(Draft generation failed — reply manually)"

    async def scan_engagement_opportunities(
        self, max_opportunities: int = 5, since_hours: int = 24
    ) -> List[Dict]:
        """
        Scan monitored Twitter accounts for engagement-worthy posts.

        Returns a list of opportunities, each with:
        - author, handle, tweet_text, tweet_url
        - relevance_score, matched_topics
        - suggested_comment (Claude-drafted)
        """
        import asyncio

        headers = self._get_twitter_headers()
        if not headers.get("Authorization") or headers["Authorization"] == "Bearer ":
            return [{"error": "Failed to get Twitter Bearer token"}]

        # Resolve handles → user IDs (cache these eventually)
        all_tweets = []
        for handle, display_name, _ in MONITORED_ACCOUNTS:
            user_id = self._resolve_user_id(handle, headers)
            if not user_id:
                continue
            tweets = self._get_user_tweets(user_id, headers, since_hours)
            for t in tweets:
                score, keywords = self._score_relevance(t.get("text", ""))
                if score >= MIN_KEYWORD_MATCHES:
                    metrics = t.get("public_metrics", {})
                    all_tweets.append({
                        "author": display_name,
                        "handle": handle,
                        "tweet_id": t["id"],
                        "tweet_text": t["text"],
                        "tweet_url": f"https://twitter.com/{handle}/status/{t['id']}",
                        "created_at": t.get("created_at"),
                        "likes": metrics.get("like_count", 0),
                        "retweets": metrics.get("retweet_count", 0),
                        "replies": metrics.get("reply_count", 0),
                        "relevance_score": score,
                        "matched_topics": keywords,
                    })

        # Sort by relevance × engagement
        all_tweets.sort(
            key=lambda t: t["relevance_score"] * (1 + t["likes"] + t["retweets"] * 3),
            reverse=True,
        )

        # Take top N and generate comments
        opportunities = all_tweets[:max_opportunities]

        # Get today's market context for Claude (if available)
        market_context = ""
        try:
            import boto3
            import json
            s3 = boto3.client("s3", region_name="us-east-1")
            obj = s3.get_object(
                Bucket="rigacap-prod-price-data-149218244179",
                Key="signals/dashboard.json",
            )
            dash = json.loads(obj["Body"].read())
            market_context = dash.get("market_context", "")
        except Exception:
            pass

        for opp in opportunities:
            opp["suggested_comment"] = self._generate_comment(
                opp["tweet_text"],
                opp["handle"],
                opp["matched_topics"],
                market_context,
            )

        return opportunities


engagement_service = EngagementService()
