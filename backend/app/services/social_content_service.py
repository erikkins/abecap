"""
Social Content Service - Generate social media posts from walk-forward trade results.

Creates draft posts for admin review/approval before posting.
Supports Twitter (280 chars) and Instagram (longer captions) formats.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from app.core.database import SocialPost, WalkForwardSimulation

logger = logging.getLogger(__name__)


class SocialContentService:
    """Generate social media content from walk-forward simulation results."""

    # Hashtag sets by post type
    HASHTAGS = {
        "trade_result": "#StockTrading #AlgoTrading #WalkForward #TradingSignals #RigaCap",
        "missed_opportunity": "#StockTrading #MissedTrade #AlgoTrading #TradingSignals #RigaCap",
        "weekly_recap": "#WeeklyRecap #StockMarket #AlgoTrading #TradingResults #RigaCap",
        "regime_commentary": "#MarketRegime #StockMarket #TradingStrategy #MarketAnalysis #RigaCap",
    }

    async def generate_from_nightly_wf(
        self, db: AsyncSession, simulation_id: int
    ) -> List[SocialPost]:
        """
        Generate social posts from a completed nightly walk-forward simulation.

        Returns list of created SocialPost records (status=draft).
        """
        # Load simulation
        result = await db.execute(
            select(WalkForwardSimulation).where(
                WalkForwardSimulation.id == simulation_id
            )
        )
        sim = result.scalar_one_or_none()
        if not sim or not sim.trades_json:
            logger.warning(f"No trades found for simulation {simulation_id}")
            return []

        trades = json.loads(sim.trades_json)
        if not trades:
            return []

        posts = []

        # Filter to recent profitable closed trades (>5% return)
        profitable = [
            t for t in trades
            if t.get("pnl_pct", 0) > 5.0 and t.get("exit_date")
        ]
        profitable.sort(key=lambda t: t.get("pnl_pct", 0), reverse=True)

        # Generate trade_result posts for top 3 winners
        for trade in profitable[:3]:
            twitter_post = self._make_trade_result_twitter(trade)
            insta_post = self._make_trade_result_instagram(trade)

            for post in [twitter_post, insta_post]:
                post.source_simulation_id = simulation_id
                post.source_trade_json = json.dumps(trade)
                db.add(post)
                posts.append(post)

        # Generate missed_opportunity posts for next 2
        for trade in profitable[3:5]:
            twitter_post = self._make_missed_opportunity_twitter(trade)
            insta_post = self._make_missed_opportunity_instagram(trade)

            for post in [twitter_post, insta_post]:
                post.source_simulation_id = simulation_id
                post.source_trade_json = json.dumps(trade)
                db.add(post)
                posts.append(post)

        # Generate regime commentary
        regime_posts = await self._make_regime_commentary(db, simulation_id)
        for post in regime_posts:
            db.add(post)
            posts.append(post)

        await db.commit()
        logger.info(f"Generated {len(posts)} social posts from simulation {simulation_id}")
        return posts

    async def generate_weekly_recap(self, db: AsyncSession) -> List[SocialPost]:
        """
        Generate weekly recap posts (called on Fridays).

        Aggregates the week's nightly WF trades into a summary post.
        """
        # Get all nightly WF simulations from the past 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)
        result = await db.execute(
            select(WalkForwardSimulation)
            .where(WalkForwardSimulation.is_nightly_missed_opps == True)
            .where(WalkForwardSimulation.status == "completed")
            .where(WalkForwardSimulation.simulation_date >= week_ago)
            .order_by(desc(WalkForwardSimulation.simulation_date))
            .limit(1)
        )
        sim = result.scalar_one_or_none()

        if not sim or not sim.trades_json:
            return []

        trades = json.loads(sim.trades_json)
        if not trades:
            return []

        # Aggregate stats
        closed_trades = [t for t in trades if t.get("exit_date")]
        winning = [t for t in closed_trades if t.get("pnl_pct", 0) > 0]
        total = len(closed_trades)
        wins = len(winning)
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_return = (
            sum(t.get("pnl_pct", 0) for t in closed_trades) / total
            if total > 0
            else 0
        )
        best_trade = max(closed_trades, key=lambda t: t.get("pnl_pct", 0)) if closed_trades else None
        total_pnl = sum(t.get("pnl_dollars", 0) for t in closed_trades)

        posts = []

        # Twitter recap
        twitter_text = (
            f"Weekly Recap (Walk-Forward Verified)\n\n"
            f"Trades: {total}\n"
            f"Win Rate: {win_rate:.0f}%\n"
            f"Avg Return: {avg_return:+.1f}%\n"
        )
        if best_trade:
            twitter_text += f"Best: ${best_trade['symbol']} {best_trade.get('pnl_pct', 0):+.1f}%\n"
        twitter_text += f"\nNo hindsight bias. Every signal verified."

        twitter_post = SocialPost(
            post_type="weekly_recap",
            platform="twitter",
            status="draft",
            text_content=twitter_text,
            hashtags=self.HASHTAGS["weekly_recap"],
            source_data_json=json.dumps({
                "total_trades": total,
                "wins": wins,
                "win_rate": round(win_rate, 1),
                "avg_return": round(avg_return, 1),
                "total_pnl": round(total_pnl, 2),
            }),
        )
        posts.append(twitter_post)

        # Instagram recap
        insta_text = (
            f"Weekly Walk-Forward Recap\n\n"
            f"Total Trades: {total}\n"
            f"Winning: {wins} | Losing: {total - wins}\n"
            f"Win Rate: {win_rate:.0f}%\n"
            f"Average Return: {avg_return:+.1f}%\n"
        )
        if best_trade:
            insta_text += (
                f"\nTop Trade: ${best_trade['symbol']}\n"
                f"  Entry: ${best_trade.get('entry_price', 0):.2f}\n"
                f"  Exit: ${best_trade.get('exit_price', 0):.2f}\n"
                f"  Return: {best_trade.get('pnl_pct', 0):+.1f}%\n"
            )
        insta_text += (
            f"\nAll results walk-forward verified.\n"
            f"No hindsight bias. No curve fitting.\n"
            f"\nrigacap.com"
        )

        insta_post = SocialPost(
            post_type="weekly_recap",
            platform="instagram",
            status="draft",
            text_content=insta_text,
            hashtags=self.HASHTAGS["weekly_recap"],
            source_data_json=json.dumps({
                "total_trades": total,
                "wins": wins,
                "win_rate": round(win_rate, 1),
                "avg_return": round(avg_return, 1),
                "total_pnl": round(total_pnl, 2),
            }),
        )
        posts.append(insta_post)

        for post in posts:
            db.add(post)
        await db.commit()

        return posts

    @staticmethod
    def _calc_days_held(trade: dict) -> int:
        """Calculate days held from entry/exit dates, falling back to days_held field."""
        entry = str(trade.get("entry_date", ""))[:10]
        exit_ = str(trade.get("exit_date", ""))[:10]
        if entry and exit_:
            try:
                d1 = datetime.strptime(entry, "%Y-%m-%d")
                d2 = datetime.strptime(exit_, "%Y-%m-%d")
                days = (d2 - d1).days
                return max(days, 1)
            except (ValueError, TypeError):
                pass
        return trade.get("days_held", 1)

    def _make_trade_result_twitter(self, trade: dict) -> SocialPost:
        """Create a Twitter-format trade result post (280 chars max)."""
        symbol = trade.get("symbol", "???")
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        pnl_pct = trade.get("pnl_pct", 0)
        entry_date = str(trade.get("entry_date", ""))[:10]
        exit_date = str(trade.get("exit_date", ""))[:10]
        exit_reason = trade.get("exit_reason", "trailing_stop")
        days_held = self._calc_days_held(trade)

        # Format exit reason for display
        exit_display = exit_reason.replace("_", " ").title() if exit_reason else "Exit"

        text = (
            f"Signal fired on ${symbol} at ${entry_price:.2f} on {entry_date}.\n"
            f"{exit_display} exit at ${exit_price:.2f} on {exit_date}.\n"
            f"{pnl_pct:+.1f}% in {days_held} days.\n"
            f"Walk-forward verified. No hindsight bias."
        )

        return SocialPost(
            post_type="trade_result",
            platform="twitter",
            status="draft",
            text_content=text,
            hashtags=self.HASHTAGS["trade_result"] + f" ${symbol}",
        )

    def _make_trade_result_instagram(self, trade: dict) -> SocialPost:
        """Create an Instagram-format trade result post (longer caption)."""
        symbol = trade.get("symbol", "???")
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        pnl_pct = trade.get("pnl_pct", 0)
        pnl_dollars = trade.get("pnl_dollars", 0)
        entry_date = str(trade.get("entry_date", ""))[:10]
        exit_date = str(trade.get("exit_date", ""))[:10]
        exit_reason = trade.get("exit_reason", "trailing_stop")
        days_held = self._calc_days_held(trade)
        strategy_name = trade.get("strategy_name", "Ensemble")

        exit_display = exit_reason.replace("_", " ").title() if exit_reason else "Exit"

        text = (
            f"Walk-forward verified trade result\n\n"
            f"${symbol}\n"
            f"Entry: ${entry_price:.2f} on {entry_date}\n"
            f"Exit: ${exit_price:.2f} on {exit_date} ({exit_display.lower()})\n"
            f"Return: {pnl_pct:+.1f}% in {days_held} days"
        )
        if pnl_dollars:
            text += f" (${pnl_dollars:+,.0f})"
        text += (
            f"\n\nStrategy: {strategy_name}\n"
            f"Walk-forward verified — no hindsight bias.\n"
            f"\nrigacap.com"
        )

        return SocialPost(
            post_type="trade_result",
            platform="instagram",
            status="draft",
            text_content=text,
            hashtags=self.HASHTAGS["trade_result"] + f" ${symbol}",
            image_metadata_json=json.dumps({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "card_type": "trade_result",
            }),
        )

    def _make_missed_opportunity_twitter(self, trade: dict) -> SocialPost:
        """Create a Twitter missed opportunity post."""
        symbol = trade.get("symbol", "???")
        pnl_pct = trade.get("pnl_pct", 0)
        entry_date = str(trade.get("entry_date", ""))[:10]
        days_held = self._calc_days_held(trade)

        text = (
            f"Did you catch ${symbol}?\n"
            f"Our algo flagged it on {entry_date}.\n"
            f"{pnl_pct:+.1f}% in {days_held} days.\n"
            f"Walk-forward verified — not backtested."
        )

        return SocialPost(
            post_type="missed_opportunity",
            platform="twitter",
            status="draft",
            text_content=text,
            hashtags=self.HASHTAGS["missed_opportunity"] + f" ${symbol}",
            source_trade_json=json.dumps(trade),
        )

    def _make_missed_opportunity_instagram(self, trade: dict) -> SocialPost:
        """Create an Instagram missed opportunity post."""
        symbol = trade.get("symbol", "???")
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        pnl_pct = trade.get("pnl_pct", 0)
        entry_date = str(trade.get("entry_date", ""))[:10]
        exit_date = str(trade.get("exit_date", ""))[:10]
        days_held = self._calc_days_held(trade)

        text = (
            f"Missed Opportunity\n\n"
            f"${symbol}\n"
            f"Signal: {entry_date} at ${entry_price:.2f}\n"
            f"Would-be exit: {exit_date} at ${exit_price:.2f}\n"
            f"Return: {pnl_pct:+.1f}% in {days_held} days\n\n"
            f"Walk-forward verified — this was a real-time signal,\n"
            f"not a backtest.\n"
            f"\nrigacap.com"
        )

        return SocialPost(
            post_type="missed_opportunity",
            platform="instagram",
            status="draft",
            text_content=text,
            hashtags=self.HASHTAGS["missed_opportunity"] + f" ${symbol}",
            image_metadata_json=json.dumps({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "card_type": "missed_opportunity",
            }),
        )

    async def _make_regime_commentary(
        self, db: AsyncSession, simulation_id: int
    ) -> List[SocialPost]:
        """Generate regime commentary posts based on current market conditions."""
        posts = []

        try:
            from app.services.market_regime import market_regime_service
            from app.services.scanner import scanner_service

            spy_df = scanner_service.data_cache.get("SPY")
            vix_df = scanner_service.data_cache.get("^VIX")

            if spy_df is None or len(spy_df) < 200:
                return posts

            forecast = market_regime_service.predict_transitions(
                spy_df=spy_df,
                universe_dfs=scanner_service.data_cache,
                vix_df=vix_df,
            )

            regime_name = forecast.current_regime_name
            regime_desc = forecast.outlook_detail
            risk_level = forecast.risk_change

            spy_price = round(float(spy_df.iloc[-1]["close"]), 2)
            vix_level = round(float(vix_df.iloc[-1]["close"]), 2) if vix_df is not None and len(vix_df) > 0 else None

            # Twitter
            twitter_text = (
                f"Market Regime: {regime_name}\n\n"
                f"SPY: ${spy_price}"
            )
            if vix_level is not None:
                twitter_text += f" | VIX: {vix_level}"
            twitter_text += (
                f"\nRisk: {risk_level.title()}\n\n"
                f"Our 7-regime detector adapts position sizing\n"
                f"and entry/exit rules automatically."
            )

            twitter_post = SocialPost(
                post_type="regime_commentary",
                platform="twitter",
                status="draft",
                text_content=twitter_text,
                hashtags=self.HASHTAGS["regime_commentary"],
                source_simulation_id=simulation_id,
                source_data_json=json.dumps({
                    "regime": regime_name,
                    "risk_level": risk_level,
                    "spy_price": spy_price,
                    "vix_level": vix_level,
                }),
            )
            posts.append(twitter_post)

            # Instagram
            insta_text = (
                f"Market Regime Update\n\n"
                f"Current: {regime_name}\n"
                f"SPY: ${spy_price}"
            )
            if vix_level is not None:
                insta_text += f" | VIX: {vix_level}"
            insta_text += (
                f"\nRisk Level: {risk_level.title()}\n\n"
                f"{regime_desc}\n\n"
                f"Our 7-regime market detector automatically adjusts\n"
                f"position sizing, entry triggers, and exit rules\n"
                f"based on current conditions.\n"
                f"\nrigacap.com"
            )

            insta_post = SocialPost(
                post_type="regime_commentary",
                platform="instagram",
                status="draft",
                text_content=insta_text,
                hashtags=self.HASHTAGS["regime_commentary"],
                source_simulation_id=simulation_id,
                source_data_json=json.dumps({
                    "regime": regime_name,
                    "risk_level": risk_level,
                    "spy_price": spy_price,
                    "vix_level": vix_level,
                }),
            )
            posts.append(insta_post)

        except Exception as e:
            logger.error(f"Regime commentary generation failed: {e}")

        return posts


# Singleton instance
social_content_service = SocialContentService()
