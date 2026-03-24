"""
AlphaBot Order Executor — Places, modifies, and cancels orders.
All orders go to the configured environment (Testnet in Phase 1).
Every order has retry logic (3 attempts with exponential backoff).
Stop-loss is placed IMMEDIATELY after entry order fills.
"""

from __future__ import annotations

from typing import Optional

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
                                  quantity: float) -> Optional[dict]:
        """
        Place a market order (guaranteed fill).
        Used for entry orders and emergency closes.
        """
        try:
            order = await self.client.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side.lower(),
                amount=quantity,
            )
            logger.info(
                f"[Executor] Market order placed: {symbol} {side} qty={quantity} "
                f"orderId={order.get('id', 'N/A')}"
            )
            return order
        except Exception as e:
            logger.error(f"[Executor] Market order failed: {symbol} {side} {quantity} — {e}")
            raise

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def place_limit_order(self, symbol: str, side: str,
                                 quantity: float, price: float) -> Optional[dict]:
        """
        Place a GTC limit order (used for TP orders).
        """
        try:
            order = await self.client.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side.lower(),
                amount=quantity,
                price=price,
                params={"timeInForce": "GTC"},
            )
            logger.info(
                f"[Executor] Limit order placed: {symbol} {side} qty={quantity} "
                f"price={price} orderId={order.get('id', 'N/A')}"
            )
            return order
        except Exception as e:
            logger.error(f"[Executor] Limit order failed: {symbol} {side} — {e}")
            raise

    @retry_async(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    async def place_stop_market(self, symbol: str, side: str,
                                 quantity: float, stop_price: float) -> Optional[dict]:
        """
        Place a stop-market order (used for SL).
        Executes at market when stop_price is reached.
        """
        try:
            order = await self.client.exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side=side.lower(),
                amount=quantity,
                params={
                    "stopPrice": stop_price,
                    "closePosition": False,
                    "workingType": "MARK_PRICE",
                },
            )
            logger.info(
                f"[Executor] Stop-market order placed: {symbol} {side} "
                f"qty={quantity} stopPrice={stop_price} orderId={order.get('id', 'N/A')}"
            )
            return order
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

    async def cleanup_stale_orders(self, symbol: str, max_age_minutes: int = 30) -> None:
        """Cancel orders older than max_age_minutes."""
        try:
            import datetime
            orders = await self.get_open_orders(symbol)
            now = datetime.datetime.now(datetime.UTC).timestamp() * 1000  # ms

            for order in orders:
                created = order.get("timestamp", now)
                age_minutes = (now - created) / 60000
                if age_minutes > max_age_minutes:
                    await self.cancel_order(symbol, order["id"])
                    logger.info(
                        f"[Executor] Stale order cancelled: {symbol} {order['id']} "
                        f"(age: {age_minutes:.0f} min)"
                    )
        except Exception as e:
            logger.error(f"[Executor] Cleanup stale orders error: {e}")
