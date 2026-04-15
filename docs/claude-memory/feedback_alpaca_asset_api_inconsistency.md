---
name: Alpaca Trading API returns 404 for some symbols that exist in Data API
description: verify_asset_ids gets false-positive 'missing_in_alpaca' for real tradeable symbols (MMC class). Fix is moderate — use bulk asset listing or data-API fallback
type: feedback
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
## The problem

Alpaca's `TradingClient.get_asset(symbol)` hits `/v2/assets/{symbol}` which 404s for some symbols that are fully active and trading. Discovered Apr 15 2026 with **MMC** (Marsh McLennan — Fortune 500 NYSE-listed stock we have 1700+ days of data for via Alpaca's data API).

**Response:**
```json
{"code": 40410000, "message": "asset not found for MMC"}
```

**Yet our pickle has current MMC bars from Alpaca's data API.** Inconsistent coverage between the two APIs.

This means `symbol_metadata_service.verify_asset_ids()` marks MMC-class symbols as `missing_in_alpaca` during the nightly hygiene run. The nightly doesn't auto-quarantine based on missing_in_alpaca (only on `asset_id_changed`), but the admin email flags them as "missing" which is misleading.

## Why this happens (probably)

Alpaca's Trading API lists assets available for order routing through their broker. Their Data API serves market data for a broader universe (includes some symbols routed via other brokers or with data feeds but no Alpaca trading support).

## Fix approach (~45-90 min)

**Option A (~30 min): bulk pre-fetch via get_all_assets**

Call `TradingClient.get_all_assets()` ONCE at the start of verify, build a symbol→asset_id map in-memory, then look up each target symbol against the map. Bulk endpoint may include more than individual lookups, and a single request is cheaper than 4500.

```python
# Pseudocode
async def verify_asset_ids(symbols, ...):
    client = self._get_trading_client()
    all_assets = await run_in_executor(client.get_all_assets, GetAssetsRequest(status=AssetStatus.ACTIVE))
    asset_map = {a.symbol: a for a in all_assets}
    for sym in symbols:
        asset = asset_map.get(sym)
        # if asset is None → try data-API fallback (Option B)
```

**Option B (~30 min on top): data-API fallback**

If Trading API doesn't know the symbol, check whether our pickle has recent bars for it. If yes (active symbol in our data), mark `unverifiable` — not `missing_in_alpaca`. Only flag as `missing_in_alpaca` if BOTH APIs lack it.

```python
if sym not in asset_map:
    # Check data cache (pickle) for recent activity
    df = scanner_service.data_cache.get(sym)
    recent = df is not None and len(df) > 0 and df.index[-1] > today - 14
    if recent:
        status = "unverifiable"  # Active in data API but not Trading API
    else:
        status = "missing_in_alpaca"  # Genuinely missing
```

**Option C (~0 min): suppress the alarm**

The email handler already only alarms when `missing_in_alpaca > 20`. If false positives stay small (<20 per night), status stays "Healthy". This is already the behavior. Downside: a real delisting wave (20+ symbols) would get masked with 10+ MMC-class noise.

## Recommended sequence

1. Ship Option A first — it's the cleanest improvement and likely resolves most cases in one shot.
2. If Option A doesn't catch all (some symbols still missing from both Trading APIs), add Option B as fallback.
3. Keep Option C's threshold-based alarm as the last-line defense.

Total build: ~60 min if both A+B needed, ~30 min if A alone suffices.

## Tangential note

The `asset.attributes` bug we already fixed (was a list, code treated as dict) was different class of issue — that was our code wrongly handling the SDK response. The current MMC issue is Alpaca's API inconsistency.
