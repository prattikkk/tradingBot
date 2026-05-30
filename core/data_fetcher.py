"""
core/data_fetcher.py
Fetches OHLCV candle data from Binance MAINNET (read-only).
No orders are placed here — this is pure market data.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd
import requests

try:
    from binance import ThreadedWebsocketManager
except Exception:  # pragma: no cover - optional dependency at runtime
    ThreadedWebsocketManager = None

from config import CONFIG
from core.resilience import CircuitBreaker, TokenBucketLimiter, retry_delay_seconds
from utils.logger import get_logger

log = get_logger("DataFetcher")

MAINNET_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"

RETRYABLE_STATUS = {418, 429, 500, 502, 503, 504}

INTERVAL_MAP = {
    "1m":  60,    "3m":  180,   "5m":  300,
    "15m": 900,   "30m": 1800,  "1h":  3600,
    "2h":  7200,  "4h":  14400, "6h":  21600,
    "1d":  86400,
}


class DataFetcher:
    """
    Pulls OHLCV data from Binance Futures MAINNET.
    Uses the public REST endpoint — no API key required for candles.
    Falls back to spot endpoint if futures fails.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._cache_ttl = 30  # seconds
        self._volume_cache: dict[str, tuple[float, float]] = {}
        self._volume_ttl = 60  # seconds
        self._funding_cache: dict[str, tuple[float, float]] = {}
        self._funding_ttl = 60  # seconds
        self._exchange_info_cache: dict[str, tuple[float, dict]] = {}
        self._exchange_info_ttl = 300  # seconds

        api_cfg = CONFIG.api
        self._rate_limiter = TokenBucketLimiter(api_cfg.rate_limit_per_minute)
        self._circuit_breaker = CircuitBreaker(
            api_cfg.circuit_failures,
            api_cfg.circuit_cooldown_seconds,
        )
        self._retry_attempts = api_cfg.retry_attempts
        self._backoff_base = api_cfg.backoff_base_seconds
        self._backoff_cap = api_cfg.backoff_cap_seconds
        self._max_concurrent = max(1, int(api_cfg.max_concurrent_requests))

        self._ws_enabled = bool(CONFIG.trading.use_websocket_data)
        self._ws_manager = None
        self._ws_started = False
        self._ws_symbol_sockets: set[str] = set()
        self._ws_prices: dict[str, tuple[float, float]] = {}
        self._ws_ttl_seconds = 5.0

        # Background event loop running in a daemon thread for async API calls
        self._loop = None
        self._loop_thread = None
        self._start_background_loop()

    def _start_background_loop(self) -> None:
        import threading
        self._loop = asyncio.new_event_loop()
        def run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()
        self._loop_thread = threading.Thread(
            target=run_loop,
            args=(self._loop,),
            daemon=True,
            name="DataFetcherAsyncLoop"
        )
        self._loop_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 200,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame with columns:
        open_time, open, high, low, close, volume,
        close_time, quote_volume, trades, taker_buy_base, taker_buy_quote
        """
        cache_key = f"{symbol}_{interval}"
        if use_cache and cache_key in self._cache:
            ts, df = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return df.copy()

        df = self._fetch_futures_candles(symbol, interval, limit)
        if df is None:
            log.warning(f"Futures candles failed for {symbol}, trying spot…")
            df = self._fetch_spot_candles(symbol, interval, limit)

        if df is not None and not df.empty:
            df = self._add_derived_columns(df)
            self._cache[cache_key] = (time.time(), df)
            log.debug(f"Fetched {len(df)} candles [{symbol} {interval}]")

        return df

    def get_multi_tf(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Fetch primary + higher timeframes for confluence."""
        return self.get_multi_tf_bulk([symbol]).get(symbol, {})

    def start_price_stream(self, symbols: list[str]) -> bool:
        """Start websocket ticker streams for low-latency prices."""
        if not self._ws_enabled or ThreadedWebsocketManager is None:
            return False

        wanted = {s.upper() for s in symbols if s}
        if not wanted:
            return False

        if not self._ws_started:
            try:
                self._ws_manager = ThreadedWebsocketManager()
                self._ws_manager.start()
                self._ws_started = True
            except Exception as e:
                log.debug("Websocket manager start failed: %s", e)
                self._ws_manager = None
                self._ws_started = False
                return False

        added = False
        for symbol in sorted(wanted - self._ws_symbol_sockets):
            try:
                self._ws_manager.start_symbol_ticker_socket(
                    callback=self._on_ticker_message,
                    symbol=symbol.lower(),
                )
                self._ws_symbol_sockets.add(symbol)
                added = True
            except Exception as e:
                log.debug("Failed subscribing websocket ticker for %s: %s", symbol, e)

        return added or bool(self._ws_symbol_sockets)

    def stop_price_stream(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2.0)
            self._loop = None
            self._loop_thread = None

        if not self._ws_started:
            return
        try:
            if self._ws_manager is not None:
                self._ws_manager.stop()
        except Exception as e:
            log.debug("Websocket manager stop failed: %s", e)
        finally:
            self._ws_manager = None
            self._ws_started = False
            self._ws_symbol_sockets.clear()

    def _on_ticker_message(self, msg: dict) -> None:
        if not isinstance(msg, dict):
            return
        symbol = str(msg.get("s", "")).upper()
        if not symbol:
            return
        try:
            price = float(msg.get("c"))
        except Exception:
            return
        self._ws_prices[symbol] = (time.time(), price)

    def get_multi_tf_bulk(self, symbols: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
        """Parallel fetch primary + HTFs for many symbols using asyncio + aiohttp."""
        cfg = CONFIG.strategy
        tfs = [cfg.primary_tf, cfg.htf_1, cfg.htf_2]
        result: dict[str, dict[str, pd.DataFrame]] = {symbol: {} for symbol in symbols}

        now = time.time()
        missing: list[tuple[str, str]] = []
        for symbol in symbols:
            for tf in tfs:
                cache_key = f"{symbol}_{tf}"
                cached = self._cache.get(cache_key)
                if cached and now - cached[0] < self._cache_ttl:
                    result[symbol][tf] = cached[1].copy()
                    continue
                missing.append((symbol, tf))

        if missing:
            fetched = self._fetch_missing_multi_tf(missing, CONFIG.trading.candle_limit)
            for (symbol, tf), df in fetched.items():
                if df is not None:
                    result[symbol][tf] = df

        return result

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Fast ticker price from mainnet."""
        price, _source = self.get_current_price_with_meta(symbol)
        return price

    def get_current_price_with_meta(self, symbol: str) -> tuple[Optional[float], str]:
        """Return current price with source metadata (`ws`, `rest`, `none`)."""
        ws_price = self._ws_prices.get(symbol.upper())
        if ws_price and time.time() - ws_price[0] <= self._ws_ttl_seconds:
            return ws_price[1], "ws"

        payload = self._request_json_sync(
            url=f"{MAINNET_BASE}/fapi/v1/ticker/price",
            params={"symbol": symbol},
            timeout=5,
            endpoint_key="market:ticker_price",
        )
        if payload is None:
            return None, "none"
        try:
            return float(payload["price"]), "rest"
        except Exception as e:
            log.error(f"Price parse failed [{symbol}]: {e}")
            return None, "none"

    def get_24h_quote_volume(self, symbol: str, use_cache: bool = True) -> Optional[float]:
        """24h quote-volume in USDT terms (futures first, then spot fallback)."""
        if use_cache and symbol in self._volume_cache:
            ts, volume = self._volume_cache[symbol]
            if time.time() - ts < self._volume_ttl:
                return volume

        futures_volume = self._fetch_quote_volume(
            url=f"{MAINNET_BASE}/fapi/v1/ticker/24hr",
            symbol=symbol,
        )
        if futures_volume is not None:
            self._volume_cache[symbol] = (time.time(), futures_volume)
            return futures_volume

        spot_volume = self._fetch_quote_volume(
            url="https://api.binance.com/api/v3/ticker/24hr",
            symbol=symbol,
        )
        if spot_volume is not None:
            self._volume_cache[symbol] = (time.time(), spot_volume)
            return spot_volume

        return None

    def get_exchange_info(self, symbol: str) -> dict:
        """Get tick size, lot size, min notional for a symbol."""
        cached = self._exchange_info_cache.get(symbol)
        if cached and time.time() - cached[0] < self._exchange_info_ttl:
            return dict(cached[1])

        payload = self._request_json_sync(
            url=f"{MAINNET_BASE}/fapi/v1/exchangeInfo",
            params={},
            timeout=10,
            endpoint_key="market:exchange_info",
        )
        if payload is None:
            return {}

        try:
            for s in payload.get("symbols", []):
                if s.get("symbol") != symbol:
                    continue

                info = {"symbol": symbol}
                for f in s.get("filters", []):
                    ft = f.get("filterType")
                    if ft == "PRICE_FILTER":
                        info["tick_size"] = float(f["tickSize"])
                    elif ft == "LOT_SIZE":
                        info["step_size"] = float(f["stepSize"])
                        info["min_qty"] = float(f["minQty"])
                    elif ft == "MIN_NOTIONAL":
                        info["min_notional"] = float(f.get("notional", 5))

                self._exchange_info_cache[symbol] = (time.time(), info)
                return info
        except Exception as e:
            log.error(f"Exchange info parse failed [{symbol}]: {e}")

        return {}

    def get_funding_rate(self, symbol: str, use_cache: bool = True) -> Optional[float]:
        """Fetch latest funding rate for the symbol."""
        if use_cache and symbol in self._funding_cache:
            ts, funding = self._funding_cache[symbol]
            if time.time() - ts < self._funding_ttl:
                return funding

        payload = self._request_json_sync(
            url=f"{MAINNET_BASE}/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            timeout=5,
            endpoint_key="market:funding_rate",
        )
        if payload is None:
            return None

        try:
            funding = float(payload.get("lastFundingRate", 0.0))
            self._funding_cache[symbol] = (time.time(), funding)
            return funding
        except Exception as e:
            log.debug("Funding rate parse error [%s]: %s", symbol, e)
            return None

    def get_close_correlation(
        self,
        symbol_a: str,
        symbol_b: str,
        interval: str,
        lookback: int,
    ) -> Optional[float]:
        """Return correlation of close-to-close returns between two symbols."""
        look = max(50, int(lookback))
        df_a = self.get_ohlcv(symbol_a, interval=interval, limit=look + 5)
        df_b = self.get_ohlcv(symbol_b, interval=interval, limit=look + 5)
        if df_a is None or df_b is None or df_a.empty or df_b.empty:
            return None

        try:
            series_a = df_a["close"].pct_change().dropna().tail(look)
            series_b = df_b["close"].pct_change().dropna().tail(look)
            merged = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
            if len(merged) < 30:
                return None
            return float(merged.iloc[:, 0].corr(merged.iloc[:, 1]))
        except Exception as e:
            log.debug("Correlation calc failed [%s/%s]: %s", symbol_a, symbol_b, e)
            return None

    def get_close_correlation_matrix(
        self,
        symbols: list[str],
        interval: str,
        lookback: int,
    ) -> dict[tuple[str, str], float]:
        """Return pairwise close-return correlations for provided symbols."""
        unique = list(dict.fromkeys([s.upper() for s in symbols if s]))
        if len(unique) < 2:
            return {}

        look = max(50, int(lookback))
        series_by_symbol: dict[str, pd.Series] = {}

        for symbol in unique:
            df = self.get_ohlcv(symbol, interval=interval, limit=look + 5)
            if df is None or df.empty:
                continue
            returns = df["close"].pct_change().dropna().tail(look)
            if len(returns) >= 30:
                series_by_symbol[symbol] = returns

        if len(series_by_symbol) < 2:
            return {}

        joined = pd.DataFrame(series_by_symbol).dropna()
        if len(joined) < 30:
            return {}

        corr = joined.corr()
        out: dict[tuple[str, str], float] = {}

        for a in corr.columns:
            for b in corr.columns:
                if a == b:
                    continue
                val = corr.at[a, b]
                if pd.isna(val):
                    continue
                out[(str(a), str(b))] = float(val)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_futures_candles(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[pd.DataFrame]:
        payload = self._request_json_sync(
            url=f"{MAINNET_BASE}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
            endpoint_key="market:futures_klines",
        )
        if payload is None:
            return None
        try:
            return self._parse_candles(payload)
        except Exception as e:
            log.debug(f"Futures candle parse error [{symbol} {interval}]: {e}")
            return None

    def _fetch_spot_candles(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[pd.DataFrame]:
        payload = self._request_json_sync(
            url=f"{SPOT_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
            endpoint_key="market:spot_klines",
        )
        if payload is None:
            return None
        try:
            return self._parse_candles(payload)
        except Exception as e:
            log.debug(f"Spot candle parse error [{symbol} {interval}]: {e}")
            return None

    def _fetch_quote_volume(self, url: str, symbol: str) -> Optional[float]:
        payload = self._request_json_sync(
            url=url,
            params={"symbol": symbol},
            timeout=5,
            endpoint_key="market:24hr_ticker",
        )
        if payload is None:
            return None
        try:
            return float(payload.get("quoteVolume", 0.0))
        except Exception as e:
            log.debug(f"24h volume parse error [{symbol}] from {url}: {e}")
            return None

    def _fetch_missing_multi_tf(
        self,
        missing: list[tuple[str, str]],
        limit: int,
    ) -> dict[tuple[str, str], Optional[pd.DataFrame]]:
        try:
            asyncio.get_running_loop()
            return self._fetch_missing_multi_tf_sync(missing, limit)
        except RuntimeError:
            pass

        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._fetch_missing_multi_tf_async(missing, limit),
                self._loop
            )
            try:
                return future.result()
            except Exception as e:
                log.error(f"Async bulk fetch failed: {e}", exc_info=True)
                return self._fetch_missing_multi_tf_sync(missing, limit)
        else:
            return self._fetch_missing_multi_tf_sync(missing, limit)

    def _fetch_missing_multi_tf_sync(
        self,
        missing: list[tuple[str, str]],
        limit: int,
    ) -> dict[tuple[str, str], Optional[pd.DataFrame]]:
        out: dict[tuple[str, str], Optional[pd.DataFrame]] = {}
        for symbol, tf in missing:
            out[(symbol, tf)] = self.get_ohlcv(symbol, tf, limit=limit, use_cache=True)
        return out

    async def _fetch_missing_multi_tf_async(
        self,
        missing: list[tuple[str, str]],
        limit: int,
    ) -> dict[tuple[str, str], Optional[pd.DataFrame]]:
        out: dict[tuple[str, str], Optional[pd.DataFrame]] = {}
        timeout = aiohttp.ClientTimeout(total=12)
        connector = aiohttp.TCPConnector(limit=self._max_concurrent)
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"Content-Type": "application/json"},
        ) as session:
            tasks = [
                self._fetch_one_tf_async(session, semaphore, symbol, tf, limit)
                for symbol, tf in missing
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                log.debug(f"Async multi-tf fetch task failed: {result}")
                continue
            if result is None:
                continue
            key, df = result
            out[key] = df

        return out

    async def _fetch_one_tf_async(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        symbol: str,
        interval: str,
        limit: int,
    ) -> tuple[tuple[str, str], Optional[pd.DataFrame]]:
        cache_key = f"{symbol}_{interval}"
        async with semaphore:
            payload = await self._request_json_async(
                session=session,
                url=f"{MAINNET_BASE}/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=10,
                endpoint_key="market:futures_klines",
            )

            if payload is None:
                payload = await self._request_json_async(
                    session=session,
                    url=f"{SPOT_BASE}/api/v3/klines",
                    params={"symbol": symbol, "interval": interval, "limit": limit},
                    timeout=10,
                    endpoint_key="market:spot_klines",
                )

        if payload is None:
            return (symbol, interval), None

        try:
            df = self._parse_candles(payload)
        except Exception as e:
            log.debug(f"Async candle parse error [{symbol} {interval}]: {e}")
            return (symbol, interval), None

        if df.empty:
            return (symbol, interval), None

        df = self._add_derived_columns(df)
        self._cache[cache_key] = (time.time(), df)
        return (symbol, interval), df.copy()

    def _request_json_sync(
        self,
        url: str,
        params: dict,
        timeout: int,
        endpoint_key: str,
    ) -> Optional[dict | list]:
        if not self._circuit_breaker.allow(endpoint_key):
            log.warning("Circuit open for %s; skipping request", endpoint_key)
            return None

        for attempt in range(self._retry_attempts + 1):
            self._rate_limiter.acquire()
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
            except Exception as e:
                self._circuit_breaker.record_failure(endpoint_key)
                if attempt < self._retry_attempts:
                    time.sleep(retry_delay_seconds(attempt, self._backoff_base, self._backoff_cap))
                    continue
                log.debug("Request failed [%s]: %s", endpoint_key, e)
                return None

            if resp.status_code in (200, 201):
                self._circuit_breaker.record_success(endpoint_key)
                try:
                    return resp.json()
                except Exception as e:
                    log.debug("JSON decode failed [%s]: %s", endpoint_key, e)
                    return None

            self._circuit_breaker.record_failure(endpoint_key)
            if resp.status_code in RETRYABLE_STATUS and attempt < self._retry_attempts:
                time.sleep(retry_delay_seconds(attempt, self._backoff_base, self._backoff_cap))
                continue

            text = resp.text[:180].replace("\n", " ") if resp.text else ""
            log.debug("Request rejected [%s] %s: %s", endpoint_key, resp.status_code, text)
            return None

        return None

    async def _request_json_async(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict,
        timeout: int,
        endpoint_key: str,
    ) -> Optional[dict | list]:
        if not self._circuit_breaker.allow(endpoint_key):
            log.warning("Circuit open for %s; skipping async request", endpoint_key)
            return None

        for attempt in range(self._retry_attempts + 1):
            await self._rate_limiter.acquire_async()
            try:
                async with session.get(url, params=params, timeout=timeout) as resp:
                    if resp.status in (200, 201):
                        self._circuit_breaker.record_success(endpoint_key)
                        try:
                            return await resp.json(content_type=None)
                        except Exception as e:
                            log.debug("Async JSON decode failed [%s]: %s", endpoint_key, e)
                            return None

                    self._circuit_breaker.record_failure(endpoint_key)
                    if resp.status in RETRYABLE_STATUS and attempt < self._retry_attempts:
                        await asyncio.sleep(
                            retry_delay_seconds(attempt, self._backoff_base, self._backoff_cap)
                        )
                        continue

                    body = (await resp.text())[:180].replace("\n", " ")
                    log.debug("Async request rejected [%s] %s: %s", endpoint_key, resp.status, body)
                    return None
            except Exception as e:
                self._circuit_breaker.record_failure(endpoint_key)
                if attempt < self._retry_attempts:
                    await asyncio.sleep(
                        retry_delay_seconds(attempt, self._backoff_base, self._backoff_cap)
                    )
                    continue
                log.debug("Async request failed [%s]: %s", endpoint_key, e)
                return None

        return None

    @staticmethod
    def _parse_candles(raw: list) -> pd.DataFrame:
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "_ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        df.drop(columns=["_ignore"], inplace=True)

        for c in ["open", "high", "low", "close", "volume",
                  "quote_volume", "taker_buy_base", "taker_buy_quote"]:
            df[c] = df[c].astype(float)
        df["trades"] = df["trades"].astype(int)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df

    @staticmethod
    def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add taker buy ratio — useful for order flow analysis."""
        df = df.copy()
        df["taker_ratio"] = np.where(
            df["volume"] > 0,
            df["taker_buy_base"] / df["volume"],
            np.nan,
        )
        df["body"] = abs(df["close"] - df["open"])
        df["wick_upper"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["wick_lower"] = df[["open", "close"]].min(axis=1) - df["low"]
        return df
