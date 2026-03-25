"""
AlphaBot WebSocket Client — Binance Mainnet live data feed.
Connects to wss://fstream.binance.com for kline + markPrice streams.
Auto-reconnects within 5 seconds on disconnect.
Also fetches historical candles via REST on startup.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from decimal import Decimal
from typing import Callable, Dict, List, Optional

import aiohttp
from loguru import logger

from alphabot.config import settings
from alphabot.data.models import Candle, Ticker, OrderBookLevel, OrderBook
from alphabot.data.data_store import DataStore
from alphabot.utils.retry import retry_async


class BinanceWebSocketClient:
    """
    Async WebSocket client for Binance Futures Mainnet.
    Streams kline data and mark-price for configured pairs.
    """

    def __init__(self, data_store: DataStore, on_candle_close: Optional[Callable] = None):
        self.data_store = data_store
        self.on_candle_close = on_candle_close  # callback when a candle closes
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._running = False
        self._reconnect_delay = 1.0

    async def start(self) -> None:
        """Start WebSocket streams and historical data fetch."""
        self._running = True
        # Fetch historical candles first
        await self._fetch_all_historical()
        # Then start WebSocket
        asyncio.create_task(self._run_websocket())
        logger.info("WebSocket client started")

    async def stop(self) -> None:
        """Gracefully close WebSocket connection."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()
        logger.info("WebSocket client stopped")

    @retry_async(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    async def _fetch_historical_candles(self, symbol: str, timeframe: str,
                                         limit: int = 200) -> List[Candle]:
        """Fetch historical klines from Binance Mainnet REST API."""
        url = f"{settings.binance_mainnet_rest_url}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": limit,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise ConnectionError(
                        f"Failed to fetch klines for {symbol}: {resp.status} {text}"
                    )
                data = await resp.json()

        candles = []
        for k in data:
            candle = Candle(
                symbol=symbol,
                timeframe=timeframe,
                open_time=datetime.datetime.utcfromtimestamp(k[0] / 1000),
                open=Decimal(str(k[1])),
                high=Decimal(str(k[2])),
                low=Decimal(str(k[3])),
                close=Decimal(str(k[4])),
                volume=Decimal(str(k[5])),
                close_time=datetime.datetime.utcfromtimestamp(k[6] / 1000),
                is_closed=True,
            )
            candles.append(candle)
        return candles

    async def _fetch_all_historical(self) -> None:
        """Fetch historical data for all configured pairs and timeframes."""
        timeframes = list({settings.primary_timeframe, *getattr(settings, "entry_timeframes", []), *getattr(settings, "bias_timeframes", [])})
        # Fallback: ensure 1h is available for confirmation
        if "1h" not in timeframes and settings.primary_timeframe != "1h":
            timeframes.append("1h")

        for symbol in settings.trading_pairs:
            for tf in timeframes:
                try:
                    candles = await self._fetch_historical_candles(symbol, tf)
                    self.data_store.load_historical(symbol, tf, candles)
                    logger.info(
                        f"Loaded {len(candles)} historical {tf} candles for {symbol}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to fetch historical data for {symbol} {tf}: {e}"
                    )

    def _build_stream_url(self) -> str:
        """Build combined WebSocket stream URL for all pairs."""
        streams = []
        bias_tfs = list(getattr(settings, "bias_timeframes", []) or [])
        for symbol in settings.trading_pairs:
            sym = symbol.lower()
            streams.append(f"{sym}@kline_{settings.primary_timeframe}")
            for tf in bias_tfs:
                if tf and tf != settings.primary_timeframe:
                    streams.append(f"{sym}@kline_{tf}")
            streams.append(f"{sym}@markPrice@1s")
        stream_path = "/".join(streams)
        return f"wss://fstream.binance.com/stream?streams={stream_path}"

    async def _run_websocket(self) -> None:
        """Main WebSocket loop with auto-reconnect."""
        while self._running:
            try:
                url = self._build_stream_url()
                logger.info(f"Connecting to WebSocket: {url[:80]}...")

                self._ws_session = aiohttp.ClientSession()
                self._ws = await self._ws_session.ws_connect(
                    url, heartbeat=20, timeout=aiohttp.ClientWSTimeout(ws_close=30.0)
                )
                self._reconnect_delay = 1.0  # reset on successful connect
                logger.info("WebSocket connected successfully")

                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_message(json.loads(msg.data))
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        logger.warning(f"WebSocket closed/error: {msg.type}")
                        break

            except asyncio.CancelledError:
                logger.info("WebSocket task cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if self._ws and not self._ws.closed:
                    await self._ws.close()
                if self._ws_session and not self._ws_session.closed:
                    await self._ws_session.close()

            if self._running:
                logger.warning(
                    f"Reconnecting in {self._reconnect_delay:.1f}s..."
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def _handle_message(self, data: dict) -> None:
        """Route incoming WebSocket messages."""
        if "stream" not in data or "data" not in data:
            return

        stream = data["stream"]
        payload = data["data"]

        if "@kline_" in stream:
            await self._handle_kline(payload)
        elif "@markPrice" in stream:
            self._handle_mark_price(payload)

    async def _handle_kline(self, payload: dict) -> None:
        """Process kline/candlestick data."""
        k = payload.get("k", {})
        symbol = k.get("s", "")
        is_closed = k.get("x", False)

        candle = Candle(
            symbol=symbol,
            timeframe=k.get("i", ""),
            open_time=datetime.datetime.utcfromtimestamp(k.get("t", 0) / 1000),
            open=Decimal(str(k.get("o", "0"))),
            high=Decimal(str(k.get("h", "0"))),
            low=Decimal(str(k.get("l", "0"))),
            close=Decimal(str(k.get("c", "0"))),
            volume=Decimal(str(k.get("v", "0"))),
            close_time=datetime.datetime.utcfromtimestamp(k.get("T", 0) / 1000),
            is_closed=is_closed,
        )

        # Always update current price
        self.data_store.update_price(symbol, candle.close)

        if is_closed:
            self.data_store.add_candle(candle)
            logger.info(
                f"Candle closed: {symbol} {candle.timeframe} "
                f"O={candle.open} H={candle.high} L={candle.low} "
                f"C={candle.close} V={candle.volume}"
            )
            if self.on_candle_close:
                try:
                    await self.on_candle_close(candle)
                except Exception as e:
                    logger.error(f"Error in candle close callback: {e}")

    def _handle_mark_price(self, payload: dict) -> None:
        """Process mark price updates."""
        symbol = payload.get("s", "")
        price = Decimal(str(payload.get("p", "0")))
        self.data_store.update_price(symbol, price)
