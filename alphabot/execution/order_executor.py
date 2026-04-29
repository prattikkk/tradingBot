"""
AlphaBot Order Executor — Places, modifies, and cancels orders.
All orders go to the configured environment (Testnet in Phase 1).
Every order has retry logic (3 attempts with exponential backoff).
Stop-loss is placed IMMEDIATELY after entry order fills.
"""

from __future__ import annotations

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
            params: dict[str, Any] = {"reduceOnly": True} if reduce_only else {}
            order_side = "buy" if side.upper() == "BUY" else "sell"
            order = await self.client.exchange.create_order(
                symbol=symbol,
                type="market",
                side=order_side,
                amount=quantity,
                params=params,
            )
            order_dict = cast(dict, order)
            logger.info(
                f"[Executor] Market order placed: {symbol} {side} qty={quantity} "
                f"reduceOnly={reduce_only} orderId={order_dict.get('id', 'N/A')}"
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
            params: dict[str, Any] = {"timeInForce": "GTC"}
            if reduce_only:
                params["reduceOnly"] = True
            order_side = "buy" if side.upper() == "BUY" else "sell"
            order = await self.client.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=order_side,
                amount=quantity,
                price=price,
                params=params,
            )
            order_dict = cast(dict, order)
            logger.info(
                f"[Executor] Limit order placed: {symbol} {side} qty={quantity} "
                f"price={price} reduceOnly={reduce_only} orderId={order_dict.get('id', 'N/A')}"
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
                amount=quantity,
                params=params,
            )
            order_dict = cast(dict, order)
            logger.info(
                f"[Executor] Stop-market order placed: {symbol} {side} "
                f"qty={quantity} stopPrice={stop_price} reduceOnly={reduce_only} "
                f"orderId={order_dict.get('id', 'N/A')}"
            )
            return order_dict
        except Exception as e:
            logger.error(f"[Executor] Stop-market order failed: {symbol} — {e}")
            raise

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
            protected = set(protected_ids or [])

            for order in orders:
                order_id = str(order.get("id", ""))
                if order_id in protected:
                    continue

                created = order.get("timestamp", now)
                age_minutes = (now - created) / 60000
                if age_minutes > max_age_minutes:
                    await self.cancel_order(symbol, order_id)
                    logger.info(
                        f"[Executor] Stale order cancelled: {symbol} {order_id} "
                        f"(age: {age_minutes:.0f} min)"
                    )
        except Exception as e:
            logger.error(f"[Executor] Cleanup stale orders error: {e}")
