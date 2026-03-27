"""
AlphaBot Testnet Client — Wrapper for Binance Futures Testnet REST API.
Accepts a testnet=True flag that switches all base URLs.
NEVER mixes testnet and mainnet — controlled by ENVIRONMENT config.
"""

from __future__ import annotations

from typing import Optional

import ccxt.async_support as ccxt
from loguru import logger

from alphabot.config import settings


class BinanceTestnetClient:
    """
    Authenticated CCXT client for Binance Futures.
    Routes to Testnet or Mainnet based on ENVIRONMENT config.
    """

    def __init__(self):
        self._exchange: Optional[ccxt.binance] = None

    async def connect(self) -> None:
        """Initialize the CCXT Binance Futures client."""
        options = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        }

        if settings.is_testnet:
            TESTNET = "https://testnet.binancefuture.com"
            self._exchange = ccxt.binance({
                "apiKey": settings.binance_testnet_api_key,
                "secret": settings.binance_testnet_secret,
                "options": {
                    **options,
                    "recvWindow": 10000,
                    "fetchCurrencies": False,
                    "fetchMargins": False,
                },
            })
            # Override ALL fapi endpoints for Binance Futures Testnet
            api = self._exchange.urls["api"]
            api["fapiPublic"] = f"{TESTNET}/fapi/v1"
            api["fapiPublicV2"] = f"{TESTNET}/fapi/v2"
            api["fapiPublicV3"] = f"{TESTNET}/fapi/v3"
            api["fapiPrivate"] = f"{TESTNET}/fapi/v1"
            api["fapiPrivateV2"] = f"{TESTNET}/fapi/v2"
            api["fapiPrivateV3"] = f"{TESTNET}/fapi/v3"
            api["fapiData"] = f"{TESTNET}/futures/data"
            api["public"] = f"{TESTNET}/api/v3"
            api["private"] = f"{TESTNET}/api/v3"
            logger.info("[Client] Connected to Binance TESTNET")
        else:
            self._exchange = ccxt.binance({
                "apiKey": settings.binance_mainnet_api_key,
                "secret": settings.binance_mainnet_secret,
                "options": options,
            })
            logger.info("[Client] Connected to Binance MAINNET")

    async def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange:
            await self._exchange.close()
            logger.info("[Client] Exchange connection closed")

    @property
    def exchange(self) -> ccxt.binance:
        if self._exchange is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._exchange

    async def get_balance(self) -> dict:
        """Fetch account balance."""
        balance = await self.exchange.fetch_balance()
        return balance

    async def get_usdt_balance(self) -> float:
        """Get available USDT balance."""
        balance = await self.get_balance()
        usdt = balance.get("USDT", {})
        return float(usdt.get("free", 0))

    async def get_futures_account_snapshot(self) -> dict:
        """Fetch Binance Futures account snapshot.

        Returns a dict with keys (best-effort, may vary by CCXT version):
        - walletBalance
        - marginBalance
        - availableBalance
        - unrealizedProfit
        - totalInitialMargin / totalMaintMargin / totalMarginBalance, etc.

        This is the closest match to what Binance Futures UI displays.
        """
        def _pick_asset_field(info: dict, asset: str, field: str) -> Optional[float]:
            assets = info.get("assets")
            if not isinstance(assets, list):
                return None
            for a in assets:
                if not isinstance(a, dict):
                    continue
                if str(a.get("asset", "")).upper() != asset.upper():
                    continue
                try:
                    return float(a.get(field))
                except Exception:
                    return None
            return None

        try:
            # CCXT Binance: fapiPrivateV2GetAccount corresponds to /fapi/v2/account
            info = await self.exchange.fapiPrivateV2GetAccount()
        except Exception as e:
            logger.warning(f"[Client] Futures account snapshot failed: {e}")
            return {}

        # Binance Futures account response often provides per-asset balances under 'assets'.
        # Normalize to a flat shape so the dashboard can display the same numbers as UI.
        if isinstance(info, dict):
            usdt_wallet = _pick_asset_field(info, "USDT", "walletBalance")
            usdt_unreal = _pick_asset_field(info, "USDT", "unrealizedProfit")
            usdt_margin = _pick_asset_field(info, "USDT", "marginBalance")
            usdt_available = _pick_asset_field(info, "USDT", "availableBalance")

            normalized = dict(info)
            if usdt_wallet is not None:
                normalized["walletBalance"] = usdt_wallet
            if usdt_margin is not None:
                normalized["marginBalance"] = usdt_margin
            if usdt_available is not None:
                normalized["availableBalance"] = usdt_available
            if usdt_unreal is not None:
                normalized["unrealizedProfit"] = usdt_unreal

            return normalized

        return {}

    async def get_positions(self) -> list:
        """Fetch all open positions."""
        positions = await self.exchange.fetch_positions()
        return [p for p in positions if float(p.get("contracts", 0)) > 0]

    async def get_exchange_info(self, symbol: str) -> dict:
        """Get market info for a symbol (lot size, tick size, etc.)."""
        markets = await self.exchange.load_markets()
        return markets.get(symbol, {})
