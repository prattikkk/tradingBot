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
        self._ws_task: Optional[asyncio.Task] = None
        self._rest_fallback_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start WebSocket streams and historical data fetch."""
        self._running = True
        # Fetch historical candles first
        await self._fetch_all_historical()
        # Then start WebSocket
        self._ws_task = asyncio.create_task(self._run_websocket())
        if settings.market_data_rest_fallback_enabled:
            self._rest_fallback_task = asyncio.create_task(self._run_rest_fallback())
            logger.warning(
                f"[MarketData] REST fallback enabled — polling every "
                f"{settings.market_data_poll_interval_seconds}s"
            )
        logger.info("WebSocket client started")

    async def stop(self) -> None:
        """Gracefully close WebSocket connection."""
        self._running = False
        for task in (self._rest_fallback_task, self._ws_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
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
        now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        for k in data:
            close_time = datetime.datetime.utcfromtimestamp(k[6] / 1000)
            candle = Candle(
                symbol=symbol,
                timeframe=timeframe,
                open_time=datetime.datetime.utcfromtimestamp(k[0] / 1000),
                open=Decimal(str(k[1])),
                high=Decimal(str(k[2])),
                low=Decimal(str(k[3])),
                close=Decimal(str(k[4])),
                volume=Decimal(str(k[5])),
                close_time=close_time,
                is_closed=close_time <= now,
            )
            candles.append(candle)
        return candles

    async def _fetch_all_historical(self) -> None:
        """Fetch historical data for all configured pairs and timeframes."""
        timeframes = self._configured_timeframes()

        for symbol in settings.trading_pairs:
            for tf in timeframes:
                try:
                    candles = await self._fetch_historical_candles(symbol, tf)
                    closed_candles = [c for c in candles if c.is_closed]
                    self.data_store.load_historical(symbol, tf, closed_candles)
                    logger.info(
                        f"Loaded {len(closed_candles)} historical {tf} candles for {symbol}"
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

    @staticmethod
    def _configured_timeframes() -> List[str]:
        timeframes: List[str] = []
        for tf in [
            settings.primary_timeframe,
            *list(getattr(settings, "entry_timeframes", []) or []),
            *list(getattr(settings, "bias_timeframes", []) or []),
        ]:
            if tf and tf not in timeframes:
                timeframes.append(tf)
        if "1h" not in timeframes and settings.primary_timeframe != "1h":
            timeframes.append("1h")
        return timeframes

    @staticmethod
    def _latest_closed_candle(candles: List[Candle]) -> Optional[Candle]:
        for candle in reversed(candles):
            if candle.is_closed:
                return candle
        return None

    async def _run_rest_fallback(self) -> None:
        """Poll closed candles from REST so entry evaluation still runs if WebSocket is idle."""
        interval = int(settings.market_data_poll_interval_seconds)
        await asyncio.sleep(interval)

        while self._running:
            try:
                synced = 0
                for symbol in settings.trading_pairs:
                    for timeframe in self._configured_timeframes():
                        candles = await self._fetch_historical_candles(symbol, timeframe, limit=3)
                        latest_closed = self._latest_closed_candle(candles)
                        if latest_closed is None:
                            continue

                        previous_open_time = self.data_store.latest_open_time(symbol, timeframe)
                        self.data_store.add_candle(latest_closed)
                        self.data_store.update_price(symbol, latest_closed.close)

                        if previous_open_time == latest_closed.open_time:
                            continue

                        synced += 1
                        logger.warning(
                            f"[MarketData] REST fallback synced closed candle: "
                            f"{symbol} {timeframe} close={latest_closed.close}"
                        )
                        if self.on_candle_close:
                            await self.on_candle_close(latest_closed)

                if synced:
                    logger.warning(f"[MarketData] REST fallback applied {synced} new candles")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[MarketData] REST fallback error: {e}")

            await asyncio.sleep(interval)

    async def _run_websocket(self) -> None:
        """Main WebSocket loop with auto-reconnect."""
        while self._running:
            try:
                url = self._build_stream_url()
                logger.info(f"Connecting to WebSocket: {url[:80]}...")

                self._ws_session = aiohttp.ClientSession()
                self._ws = await self._ws_session.ws_connect(url, heartbeat=20)
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
