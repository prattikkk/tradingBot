"""
AlphaBot Order Executor — Places, modifies, and cancels orders.
All orders go to the configured environment (Testnet in Phase 1).
Every order has retry logic (3 attempts with exponential backoff).
Stop-loss is placed IMMEDIATELY after entry order fills.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, cast

from loguru import logger

from alphabot.config import settings
from alphabot.execution.testnet_client import BinanceTestnetClient
from alphabot.utils.retry import retry_async


class OrderExecutor:
    """
    Places orders on Binance Futures (Testnet or Mainnet).
    All external API calls have retry logic.
    """

    def __init__(self, client: BinanceTestnetClient):
        self.client = client
        self._markets_loaded = False

    @staticmethod
    def _extract_order_id(order: Optional[dict]) -> str:
        # FIX[2]: Canonical ccxt id normalization used across executor and position manager.
        if not order or not isinstance(order, dict):
            return ""
        order_id = order.get("id") or order.get("orderId")
        if order_id:
            return str(order_id)
        info = order.get("info")
        if isinstance(info, dict):
            info_id = info.get("orderId") or info.get("id")
            if info_id:
                return str(info_id)
        return ""

    async def _ensure_markets_loaded(self) -> None:
        if self._markets_loaded:
            return
        await self.client.exchange.load_markets()
        self._markets_loaded = True

    async def _normalize_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to exchange precision and enforce minimum amount."""
        if quantity <= 0:
            raise ValueError(f"quantity must be positive for {symbol}")

        # FIX[3]: Cache exchange metadata; avoid reloading markets for every order.
        await self._ensure_markets_loaded()
        market = self.client.exchange.market(symbol)

        normalized = quantity
        try:
            precise_amount = self.client.exchange.amount_to_precision(symbol, quantity)
            if precise_amount is not None:
                normalized = float(precise_amount)
            else:
                normalized = float(quantity)
        except Exception:
            normalized = float(quantity)

        market_limits: dict[str, Any] = {}
        if isinstance(market, dict):
            market_limits = cast(dict[str, Any], (market.get("limits") or {}).get("amount") or {})
        min_amount = market_limits.get("min")
        if min_amount is not None:
            min_amount_f = float(min_amount)
            if normalized < min_amount_f:
                raise ValueError(
                    f"quantity {normalized} below min amount {min_amount_f} for {symbol}"
                )

        return normalized

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def set_margin_mode(self, symbol: str, margin_mode: str) -> None:
        """Ensure symbol margin mode is configured before opening positions."""
        mode = str(margin_mode or "isolated").lower()
        if mode not in {"isolated", "cross"}:
            raise ValueError(f"Unsupported margin mode: {margin_mode}")

        try:
            await self.client.exchange.set_margin_mode(mode, symbol)
            logger.info(f"[Executor] Margin mode set: {symbol} = {mode}")
        except Exception as e:
            msg = str(e).lower()
            if (
                "no need" in msg
                or "already" in msg
                or "-4046" in msg
                or "margin type cannot be changed" in msg
            ):
                logger.debug(f"[Executor] Margin mode already {mode} for {symbol}")
                return
            logger.error(f"[Executor] Set margin mode failed: {symbol} {mode} — {e}")
            raise

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""
        try:
            await self.client.exchange.set_leverage(leverage, symbol)
            logger.info(f"[Executor] Leverage set: {symbol} = {leverage}x")
        except Exception as e:
            # Some exchanges return error if leverage already set
            if "No need" in str(e) or "same" in str(e).lower():
                logger.debug(f"[Executor] Leverage already at {leverage}x for {symbol}")
            else:
                raise

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def place_market_order(self, symbol: str, side: str,
                                  quantity: float, reduce_only: bool = False) -> Optional[dict]:
        """
        Place a market order (guaranteed fill).
        Used for entry orders and emergency closes.
        """
        try:
            normalized_qty = await self._normalize_quantity(symbol, quantity)
            params: dict[str, Any] = {"newOrderRespType": "RESULT"}
            if reduce_only:
                params["reduceOnly"] = True
            order_side = "buy" if side.upper() == "BUY" else "sell"
            order = await self.client.exchange.create_order(
                symbol=symbol,
                type="market",
                side=order_side,
                amount=normalized_qty,
                params=params,
            )
            order_dict = cast(dict, order)
            logger.info(
                f"[Executor] Market order placed: {symbol} {side} qty={normalized_qty} "
                f"reduceOnly={reduce_only} orderId={self._extract_order_id(order_dict) or 'N/A'}"
            )
            return order_dict
        except Exception as e:
            logger.error(f"[Executor] Market order failed: {symbol} {side} {quantity} — {e}")
            raise

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def place_limit_order(self, symbol: str, side: str,
                                 quantity: float, price: float,
                                 reduce_only: bool = False) -> Optional[dict]:
        """
        Place a GTC limit order (used for TP orders).
        """
        try:
            normalized_qty = await self._normalize_quantity(symbol, quantity)
            params: dict[str, Any] = {"timeInForce": "GTC"}
            if reduce_only:
                params["reduceOnly"] = True
            order_side = "buy" if side.upper() == "BUY" else "sell"
            order = await self.client.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=order_side,
                amount=normalized_qty,
                price=price,
                params=params,
            )
            order_dict = cast(dict, order)
            logger.info(
                f"[Executor] Limit order placed: {symbol} {side} qty={normalized_qty} "
                f"price={price} reduceOnly={reduce_only} orderId={self._extract_order_id(order_dict) or 'N/A'}"
            )
            return order_dict
        except Exception as e:
            logger.error(f"[Executor] Limit order failed: {symbol} {side} — {e}")
            raise

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def place_stop_market(self, symbol: str, side: str,
                                 quantity: float, stop_price: float,
                                 reduce_only: bool = True) -> Optional[dict]:
        """
        Place a stop-market order (used for SL).
        Executes at market when stop_price is reached.
        """
        try:
            normalized_qty = await self._normalize_quantity(symbol, quantity)
            params: dict[str, Any] = {
                "stopPrice": stop_price,
                "workingType": "MARK_PRICE",
            }
            if reduce_only:
                params["reduceOnly"] = True
            order_side = "buy" if side.upper() == "BUY" else "sell"
            order = await self.client.exchange.create_order(
                symbol=symbol,
                type=cast(Any, "stop_market"),
                side=order_side,
                amount=normalized_qty,
                params=params,
            )
            order_dict = cast(dict, order)
            logger.info(
                f"[Executor] Stop-market order placed: {symbol} {side} "
                f"qty={normalized_qty} stopPrice={stop_price} reduceOnly={reduce_only} "
                f"orderId={self._extract_order_id(order_dict) or 'N/A'}"
            )
            return order_dict
        except Exception as e:
            logger.error(f"[Executor] Stop-market order failed: {symbol} — {e}")
            raise

    async def wait_for_order_fill(
        self,
        symbol: str,
        order_id: str,
        initial_order: Optional[dict] = None,
        timeout_seconds: Optional[float] = None,
        poll_seconds: float = 0.4,
    ) -> Optional[dict]:
        """Wait briefly for market fill confirmation before position bookkeeping."""
        def _is_filled(snapshot: dict) -> bool:
            status = str(snapshot.get("status", "")).lower()
            if status in {"closed", "filled"}:
                return True
            try:
                filled = float(snapshot.get("filled") or 0)
            except Exception:
                filled = 0.0
            return filled > 0

        current = cast(dict, initial_order or {})
        if current and _is_filled(current):
            return current

        order_id = str(order_id or self._extract_order_id(current)).strip()
        if not order_id:
            return None

        # FIX[1]: Poll exchange for a short, configurable fill-confirmation window.
        effective_timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "order_fill_timeout_seconds", 6.0)
        )
        deadline = asyncio.get_running_loop().time() + effective_timeout
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(poll_seconds)
            latest = await self.get_order(symbol, order_id)
            if not latest:
                continue
            if _is_filled(latest):
                return latest

        logger.warning(
            f"[Executor] Fill confirmation timeout: {symbol} orderId={order_id}"
        )
        return None

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def cancel_order(self, symbol: str, order_id: str) -> None:
        """Cancel a specific order."""
        try:
            await self.client.exchange.cancel_order(order_id, symbol)
            logger.info(f"[Executor] Order cancelled: {symbol} {order_id}")
        except Exception as e:
            if "Unknown order" in str(e) or "not found" in str(e).lower():
                logger.debug(f"[Executor] Order {order_id} already cancelled/filled")
            else:
                logger.error(f"[Executor] Cancel order failed: {symbol} {order_id} — {e}")
                raise

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""
        try:
            await self.client.exchange.cancel_all_orders(symbol)
            logger.info(f"[Executor] All orders cancelled for {symbol}")
        except Exception as e:
            logger.error(f"[Executor] Cancel all orders failed: {symbol} — {e}")
            raise

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def get_open_orders(self, symbol: str) -> list:
        """Get all open orders for a symbol."""
        try:
            orders = await self.client.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"[Executor] Fetch open orders failed: {symbol} — {e}")
            raise

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def get_order(self, symbol: str, order_id: str) -> Optional[dict]:
        """Fetch a specific order (open or closed) by id."""
        try:
            order = await self.client.exchange.fetch_order(order_id, symbol)
            return cast(dict, order)
        except Exception as e:
            msg = str(e).lower()
            if "unknown order" in msg or "not found" in msg or "does not exist" in msg:
                logger.debug(f"[Executor] Order not found during reconciliation: {symbol} {order_id}")
                return None
            logger.error(f"[Executor] Fetch order failed: {symbol} {order_id} — {e}")
            raise

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def get_my_trades(self, symbol: str, limit: int = 50) -> list:
        """Fetch recent user trades for reconciliation fallback."""
        try:
            trades = await self.client.exchange.fetch_my_trades(symbol=symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"[Executor] Fetch my trades failed: {symbol} — {e}")
            raise

    async def cleanup_stale_orders(
        self,
        symbol: str,
        max_age_minutes: int = 30,
        protected_ids: Optional[list[str]] = None,
    ) -> None:
        """Cancel non-protected orders older than max_age_minutes."""
        try:
            import datetime
            orders = await self.get_open_orders(symbol)
            now = datetime.datetime.now(datetime.UTC).timestamp() * 1000  # ms
            protected = {str(v) for v in (protected_ids or []) if v}

            for order in orders:
                order_dict = order if isinstance(order, dict) else {}
                # FIX[2]: Use normalized order-id extraction in stale cleanup path too.
                order_id = self._extract_order_id(cast(dict, order_dict))
                client_order_id = str(order_dict.get("clientOrderId", ""))
                if (
                    order_id in protected
                    or (client_order_id and client_order_id in protected)
                ):
                    continue
                if not order_id:
                    continue

                created = (
                    order_dict.get("timestamp")
                    or order_dict.get("lastTradeTimestamp")
                    or (
                        (order_dict.get("info", {}) if isinstance(order_dict.get("info"), dict) else {})
                    ).get("time")
                    or now
                )
                age_minutes = (now - created) / 60000
                if age_minutes > max_age_minutes:
                    await self.cancel_order(symbol, order_id)
                    logger.info(
                        f"[Executor] Stale order cancelled: {symbol} {order_id} "
                        f"(age: {age_minutes:.0f} min)"
                    )
        except Exception as e:
            logger.error(f"[Executor] Cleanup stale orders error: {e}")
