"""
core/executor.py
Places orders on Binance Futures TESTNET only.
Uses HMAC-signed REST requests — no SDK dependency.
"""
from __future__ import annotations
import hashlib
import hmac
import time
import math
import os
import requests
from typing import Optional, Any
from urllib.parse import urlencode

from core.resilience import CircuitBreaker, TokenBucketLimiter, retry_delay_seconds
from core.signal import Direction
from core.risk_manager import PositionSize
from config import CONFIG
from utils.logger import get_logger

log = get_logger("Executor")

TESTNET_BASE = "https://testnet.binancefuture.com"
PAPI_BASE = "https://papi.binance.com"


class TestnetExecutor:
    """
    All order execution happens on Binance Futures TESTNET.
    Market data is NEVER fetched here — see DataFetcher.
    """

    def __init__(self):
        self.api_key    = CONFIG.binance.testnet_api_key
        self.api_secret = CONFIG.binance.testnet_secret
        self.session    = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json",
        })
        self.papi_base = os.getenv(
            "BINANCE_PAPI_BASE_URL",
            TESTNET_BASE if CONFIG.binance.testnet else PAPI_BASE,
        )
        self._exchange_cache: dict[str, dict] = {}

        api_cfg = CONFIG.api
        self._rate_limiter = TokenBucketLimiter(api_cfg.rate_limit_per_minute)
        self._circuit_breaker = CircuitBreaker(
            api_cfg.circuit_failures,
            api_cfg.circuit_cooldown_seconds,
        )
        self._retry_attempts = api_cfg.retry_attempts
        self._backoff_base = api_cfg.backoff_base_seconds
        self._backoff_cap = api_cfg.backoff_cap_seconds
        self._last_api_error_code: Optional[int] = None
        self._last_api_error_status: Optional[int] = None
        self._last_api_error_path: str = ""
        self._last_api_error_message: str = ""

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def open_position(self, pos: PositionSize) -> dict:
        """
        Places:
          1. Market entry order
                    2. Stop-Loss order (UM algo conditional STOP_MARKET)
                    3. TP1 order (UM algo conditional TAKE_PROFIT_MARKET, 50%)
                    4. TP2 order (UM algo conditional TAKE_PROFIT_MARKET, remaining 50%)
        Returns dict of order IDs.
        """
        side       = "BUY"  if pos.direction == Direction.LONG  else "SELL"
        close_side = "SELL" if pos.direction == Direction.LONG  else "BUY"
        use_exchange_protection = bool(getattr(CONFIG.trading, "exchange_protective_orders", False))
        strict_protection_required = bool(getattr(CONFIG.trading, "strict_protection_required", False))

        if strict_protection_required and not use_exchange_protection:
            log.error(
                "Strict protection is enabled but exchange protective orders are disabled; skipping %s",
                pos.symbol,
            )
            return {}

        # 1. Set leverage
        self._set_leverage(pos.symbol, pos.leverage)

        # 2. Entry — MARKET
        entry_order = self._place_order(
            symbol=pos.symbol,
            side=side,
            order_type="MARKET",
            quantity=pos.quantity,
        )
        if not entry_order:
            if self._last_api_error_code == -2019:
                log.warning("Entry order skipped for %s: insufficient margin", pos.symbol)
            else:
                log.error("Entry order failed for %s", pos.symbol)
            return {}

        if not use_exchange_protection:
            ids = {
                "entry": entry_order.get("orderId"),
                "sl": None,
                "tp1": None,
                "tp2": None,
            }
            log.info("Orders placed [%s] with client-side protection only: %s", pos.symbol, ids)
            return ids

        qty_half = self._round_qty(pos.quantity / 2, pos.symbol)
        qty_rest = self._round_qty(pos.quantity - qty_half, pos.symbol)

        # 3. Stop Loss — algo conditional
        sl_order = self._place_protective_order(
            symbol=pos.symbol,
            side=close_side,
            order_type="STOP_MARKET",
            quantity=pos.quantity,
            trigger_price=self._tick_round(pos.stop_loss, pos.symbol),
        )

        # 4. TP1 — partial algo conditional
        tp1_order = self._place_protective_order(
            symbol=pos.symbol,
            side=close_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=qty_half,
            trigger_price=self._tick_round(pos.take_profit_1, pos.symbol),
        )

        # 5. TP2 — rest algo conditional
        tp2_order = self._place_protective_order(
            symbol=pos.symbol,
            side=close_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=qty_rest,
            trigger_price=self._tick_round(pos.take_profit_2, pos.symbol),
        )

        sl_id = self._extract_protective_id(sl_order)
        tp1_id = self._extract_protective_id(tp1_order)
        tp2_id = self._extract_protective_id(tp2_order)
        protective_ids = [sl_id, tp1_id, tp2_id]

        protection_incomplete = (not sl_order or not tp1_order or not tp2_order)
        if not protection_incomplete:
            protection_incomplete = any(order_id is None for order_id in protective_ids)

        require_confirmed_ids = bool(
            getattr(CONFIG.trading, "strict_protection_require_confirmed_ids", True)
        )
        if strict_protection_required and require_confirmed_ids and not protection_incomplete:
            # A pending client ID is only an async acknowledgement, not proof the
            # protective order is live on exchange.
            protection_incomplete = any(
                isinstance(order_id, str) and order_id.startswith("pending:")
                for order_id in protective_ids
            )

        if protection_incomplete:
            if strict_protection_required:
                log.error(
                    "[%s] protective orders incomplete or unconfirmed; flattening entry because strict protection is enabled",
                    pos.symbol,
                )
                self._cancel_protective_candidate(pos.symbol, sl_order)
                self._cancel_protective_candidate(pos.symbol, tp1_order)
                self._cancel_protective_candidate(pos.symbol, tp2_order)

                flattened = self.close_position_market(pos.symbol, pos.direction, pos.quantity)
                if not flattened:
                    log.error(
                        "[%s] failed to flatten unprotected entry after protection failure",
                        pos.symbol,
                    )
                return {}

            log.warning(
                "[%s] protective orders incomplete (SL/TP); using client-side protection fallback",
                pos.symbol,
            )
            ids = {
                "entry": entry_order.get("orderId"),
                "sl": None,
                "tp1": None,
                "tp2": None,
            }
            return ids

        ids = {
            "entry": entry_order.get("orderId"),
            "sl":    sl_id,
            "tp1":   tp1_id,
            "tp2":   tp2_id,
        }
        log.info("Orders placed [%s]: %s", pos.symbol, ids)
        return ids

    def cancel_order(self, symbol: str, order_id: int | str) -> bool:
        order_ref = str(order_id)
        if order_ref.startswith("pending:"):
            # Async-accepted algo orders may not return immediate server IDs.
            log.debug("[%s] best-effort cancel for unconfirmed protective id %s", symbol, order_ref)
            return True

        if order_ref.startswith("algo:"):
            algo_id = order_ref.split(":", 1)[1]
            return self._cancel_um_algo_order(algo_id)

        if order_ref.startswith("cond:"):
            strategy_id = order_ref.split(":", 1)[1]
            return self._cancel_um_conditional_order(symbol, strategy_id)

        params = {"symbol": symbol, "orderId": order_id}
        resp = self._signed_delete("/fapi/v1/order", params)
        return resp is not None

    def cancel_all_open_orders(self, symbol: str) -> bool:
        params = {"symbol": symbol}
        resp = self._signed_delete("/fapi/v1/allOpenOrders", params)
        return resp is not None

    def get_position(self, symbol: str) -> Optional[dict]:
        """Query current testnet position."""
        resp = self._signed_get("/fapi/v2/positionRisk", {"symbol": symbol})
        if resp and isinstance(resp, list) and len(resp) > 0:
            return resp[0]
        return None

    def get_account(self) -> Optional[dict]:
        return self._signed_get("/fapi/v2/account", {})

    def get_quote_asset_balance(self, quote_asset: str | None = None) -> Optional[dict]:
        """Return wallet and available balance for the quote asset (default USDT)."""
        account = self.get_account()
        if not isinstance(account, dict):
            return None

        asset = (quote_asset or CONFIG.binance.quote_asset or "USDT").upper()
        assets = account.get("assets", [])
        if isinstance(assets, list):
            for row in assets:
                if str(row.get("asset", "")).upper() != asset:
                    continue
                wallet = self._to_float(row.get("walletBalance", 0.0))
                available = self._to_float(row.get("availableBalance", wallet))
                return {
                    "asset": asset,
                    "wallet_balance": wallet,
                    "available_balance": available,
                }

        # Fallback when per-asset rows are absent.
        wallet = self._to_float(account.get("totalWalletBalance", 0.0))
        available = self._to_float(account.get("availableBalance", wallet))
        if wallet > 0 or available > 0:
            return {
                "asset": asset,
                "wallet_balance": wallet,
                "available_balance": available,
            }
        return None

    def get_order_status(self, symbol: str, order_id: int) -> Optional[dict]:
        return self._signed_get("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def get_open_positions(self, symbols: Optional[list[str]] = None) -> dict[str, dict]:
        """Get non-zero futures positions keyed by symbol."""
        account = self.get_account()
        if not account:
            return {}

        symbol_filter = set(symbols or [])
        positions: dict[str, dict] = {}
        for pos in account.get("positions", []):
            symbol = pos.get("symbol")
            if not symbol:
                continue
            if symbol_filter and symbol not in symbol_filter:
                continue

            qty = self._to_float(pos.get("positionAmt", 0))
            if abs(qty) <= 0:
                continue

            positions[symbol] = {
                "symbol": symbol,
                "quantity": qty,
                "entry_price": self._to_float(pos.get("entryPrice", 0)),
                "mark_price": self._to_float(pos.get("markPrice", 0)),
                "notional": self._to_float(pos.get("notional", 0)),
                "leverage": int(self._to_float(pos.get("leverage", 1)) or 1),
            }

        return positions

    def get_open_orders(self, symbol: str) -> list[dict]:
        """Return open orders for a symbol (best-effort, empty list on failure)."""
        resp = self._signed_get("/fapi/v1/openOrders", {"symbol": symbol})
        if resp is None:
            return []

        payload = resp
        if isinstance(resp, dict) and "data" in resp:
            payload = resp.get("data")

        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []

    def place_stop_loss(
        self,
        symbol: str,
        direction: Direction | str,
        quantity: float,
        stop_price: float,
    ) -> Optional[str | int]:
        """Create (or recreate) a protective stop-loss order for an open position."""
        if not self.api_key or not self.api_secret:
            return "DRY_RUN"

        close_side = "SELL" if self._is_long(direction) else "BUY"
        order = self._place_protective_order(
            symbol=symbol,
            side=close_side,
            order_type="STOP_MARKET",
            quantity=quantity,
            trigger_price=self._tick_round(stop_price, symbol),
        )
        return self._extract_protective_id(order)

    def close_position_market(
        self,
        symbol: str,
        direction: Direction | str,
        quantity: float,
    ) -> bool:
        """Emergency market close for the remaining quantity."""
        if not self.api_key or not self.api_secret:
            return True

        close_side = "SELL" if self._is_long(direction) else "BUY"
        close_order = self._place_order(
            symbol=symbol,
            side=close_side,
            order_type="MARKET",
            quantity=quantity,
            reduce_only=True,
        )
        if close_order is None:
            # Some account modes can reject reduceOnly on market close.
            close_order = self._place_order(
                symbol=symbol,
                side=close_side,
                order_type="MARKET",
                quantity=quantity,
            )
        return close_order is not None

    def place_trailing_stop(
        self,
        symbol: str,
        direction: Direction | str,
        quantity: float,
        callback_rate_pct: float,
        activation_price: float | None = None,
    ) -> Optional[str | int]:
        """Create exchange-side trailing stop to protect the remaining position."""
        if not self.has_credentials:
            return None

        qty = self._round_qty(quantity, symbol)
        if qty <= 0:
            return None

        close_side = "SELL" if self._is_long(direction) else "BUY"
        callback = max(0.1, min(10.0, float(callback_rate_pct)))

        params = {
            "symbol": symbol,
            "side": close_side,
            "type": "TRAILING_STOP_MARKET",
            "quantity": qty,
            "callbackRate": round(callback, 2),
            "workingType": "CONTRACT_PRICE",
            "reduceOnly": "true",
        }
        if activation_price is not None and activation_price > 0:
            params["activationPrice"] = self._tick_round(float(activation_price), symbol)

        resp = self._signed_post("/fapi/v1/order", params)
        if not resp:
            # Some account modes reject reduceOnly for trailing stops.
            params.pop("reduceOnly", None)
            resp = self._signed_post("/fapi/v1/order", params)

        if not resp:
            return None
        return resp.get("orderId")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        stop_price: Optional[float] = None,
        price: Optional[float] = None,
        close_position: bool = False,
        reduce_only: bool = False,
    ) -> Optional[dict]:
        params: dict = {
            "symbol":   symbol,
            "side":     side,
            "type":     order_type,
            "quantity": quantity,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if stop_price:
            params["stopPrice"] = stop_price
        if price:
            params["price"] = price
            params["timeInForce"] = "GTC"
        if close_position:
            params["closePosition"] = "true"
            params.pop("quantity", None)

        resp = self._signed_post("/fapi/v1/order", params)
        if resp:
            log.debug(f"  Order: {order_type} {side} {symbol} qty={quantity} -> id={resp.get('orderId')}")
        return resp

    def _set_leverage(self, symbol: str, leverage: int):
        self._signed_post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def _place_protective_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        trigger_price: float,
    ) -> Optional[dict]:
        """Place protective conditional order via UM algo endpoint."""
        qty = self._round_qty(quantity, symbol)
        if qty <= 0:
            log.debug("[%s] protective order qty rounded to zero: %.6f -> %.6f", symbol, quantity, qty)
            return None

        client_id = f"ab_{int(time.time() * 1000)}"

        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": qty,
            "triggerPrice": trigger_price,
            "workingType": "CONTRACT_PRICE",
            "priceProtect": "false",
            "reduceOnly": "true",
            "newOrderRespType": "ACK",
            "clientAlgoId": client_id,
        }
        resp = self._signed_post_papi("/papi/v1/um/algo/order", params)
        if resp and resp.get("accepted") and "client_id" not in resp:
            resp["client_id"] = client_id
        if resp:
            return resp

        # Fallback for accounts that only support legacy conditional endpoint.
        fallback = {
            "symbol": symbol,
            "side": side,
            "strategyType": order_type,
            "quantity": qty,
            "stopPrice": trigger_price,
            "workingType": "CONTRACT_PRICE",
            "priceProtect": "false",
            "reduceOnly": "true",
            "newClientStrategyId": client_id,
        }
        fallback_resp = self._signed_post_papi("/papi/v1/um/conditional/order", fallback)
        if fallback_resp and fallback_resp.get("accepted") and "client_id" not in fallback_resp:
            fallback_resp["client_id"] = client_id
        return fallback_resp

    def _cancel_um_algo_order(self, algo_id: str) -> bool:
        resp = self._signed_delete_papi("/papi/v1/um/algo/order", {"algoId": algo_id})
        if isinstance(resp, dict) and "complete" in resp:
            return bool(resp.get("complete"))
        return resp is not None

    def _cancel_um_conditional_order(self, symbol: str, strategy_id: str) -> bool:
        params = {
            "symbol": symbol,
            "strategyId": strategy_id,
        }
        resp = self._signed_delete_papi("/papi/v1/um/conditional/order", params)
        return resp is not None

    @staticmethod
    def _extract_protective_id(order: Optional[dict]) -> Optional[str | int]:
        if not order:
            return None
        if order.get("client_id"):
            return f"pending:{order['client_id']}"
        if "algoId" in order:
            return f"algo:{order['algoId']}"
        if "strategyId" in order:
            return f"cond:{order['strategyId']}"
        if "orderId" in order:
            return order["orderId"]
        return None

    def _cancel_protective_candidate(self, symbol: str, order: Optional[dict]) -> None:
        order_ref = self._extract_protective_id(order)
        if not order_ref:
            return
        self.cancel_order(symbol, order_ref)

    def _signed_post(self, path: str, params: dict) -> Optional[dict]:
        return self._signed_request("POST", path, params, TESTNET_BASE)

    def _signed_post_papi(self, path: str, params: dict) -> Optional[dict]:
        return self._signed_request("POST", path, params, self.papi_base)

    def _signed_get(self, path: str, params: dict) -> Optional[dict]:
        return self._signed_request("GET", path, params, TESTNET_BASE)

    def _signed_delete(self, path: str, params: dict) -> Optional[dict]:
        return self._signed_request("DELETE", path, params, TESTNET_BASE)

    def _signed_delete_papi(self, path: str, params: dict) -> Optional[dict]:
        return self._signed_request("DELETE", path, params, self.papi_base)

    def _signed_request(
        self,
        method: str,
        path: str,
        params: dict,
        base_url: str,
    ) -> Optional[dict]:
        if not self.api_key or not self.api_secret:
            log.warning("Testnet API keys not configured — order skipped (dry run mode)")
            return {"orderId": 0, "status": "DRY_RUN"}

        endpoint_key = f"{method}:{path}"
        if not self._circuit_breaker.allow(endpoint_key):
            log.warning("Circuit open for %s; skipping request", endpoint_key)
            return None

        self._reset_last_api_error()

        url = f"{base_url}{path}"
        retryable_status = {418, 429, 500, 502, 503, 504}

        for attempt in range(self._retry_attempts + 1):
            params_with_ts = dict(params)
            params_with_ts["timestamp"] = int(time.time() * 1000)
            query = urlencode(params_with_ts)
            sig = hmac.new(
                self.api_secret.encode(), query.encode(), hashlib.sha256
            ).hexdigest()
            query += f"&signature={sig}"

            try:
                self._rate_limiter.acquire()
                if method == "POST":
                    resp = self.session.post(f"{url}?{query}", timeout=10)
                elif method == "DELETE":
                    resp = self.session.delete(f"{url}?{query}", timeout=10)
                else:
                    resp = self.session.get(f"{url}?{query}", timeout=10)

                if resp.status_code in (200, 201, 202):
                    self._circuit_breaker.record_success(endpoint_key)
                    self._reset_last_api_error()
                    try:
                        data = resp.json()
                        if isinstance(data, dict):
                            return data
                        return {"data": data}
                    except Exception:
                        if resp.status_code == 202:
                            return {"accepted": True}
                        return {}

                error_code, text = self._parse_api_error(resp)

                # Margin insufficiency is a business rejection, not a transport failure.
                # Keep processing the remaining symbols instead of opening the shared circuit.
                if error_code != -2019:
                    self._circuit_breaker.record_failure(endpoint_key)

                if (
                    resp.status_code in retryable_status
                    and attempt < self._retry_attempts
                    and self._is_retryable_request(method, path)
                ):
                    time.sleep(retry_delay_seconds(attempt, self._backoff_base, self._backoff_cap))
                    continue

                self._record_last_api_error(path, resp.status_code, error_code, text)
                self._log_api_http_failure(
                    method,
                    path,
                    base_url,
                    resp.status_code,
                    error_code,
                    text,
                )
                return None
            except Exception as e:
                self._circuit_breaker.record_failure(endpoint_key)
                self._record_last_api_error(path, None, None, str(e))
                if attempt < self._retry_attempts and self._is_retryable_request(method, path):
                    time.sleep(retry_delay_seconds(attempt, self._backoff_base, self._backoff_cap))
                    continue
                log.error(f"Testnet request failed [{path}]: {e}")
                return None

        return None

    def _record_last_api_error(
        self,
        path: str,
        status_code: Optional[int],
        error_code: Optional[int],
        message: str,
    ) -> None:
        self._last_api_error_path = path
        self._last_api_error_status = status_code
        self._last_api_error_code = error_code
        self._last_api_error_message = message

    def _reset_last_api_error(self) -> None:
        self._last_api_error_path = ""
        self._last_api_error_status = None
        self._last_api_error_code = None
        self._last_api_error_message = ""

    @staticmethod
    def _parse_api_error(resp: requests.Response) -> tuple[Optional[int], str]:
        text = resp.text[:200].replace("\n", " ") if resp.text else ""
        try:
            data: Any = resp.json()
        except Exception:
            return None, text

        if not isinstance(data, dict):
            return None, text

        raw_code = data.get("code")
        try:
            code = int(raw_code)
        except (TypeError, ValueError):
            code = None
        return code, text

    def _log_api_http_failure(
        self,
        method: str,
        path: str,
        base_url: str,
        status_code: int,
        error_code: Optional[int],
        text: str,
    ) -> None:
        message = f"API {method} {base_url}{path} -> {status_code}: {text}"
        if error_code == -2019:
            log.warning("%s [insufficient margin]", message)
            return
        if error_code == -4120:
            log.info("%s [unsupported endpoint for this order type]", message)
            return
        if error_code == -1021:
            log.warning("%s [timestamp outside recvWindow]", message)
            return
        log.error(message)

    @staticmethod
    def _is_long(direction: Direction | str) -> bool:
        if isinstance(direction, Direction):
            return direction == Direction.LONG
        return str(direction).upper() == Direction.LONG.value

    @staticmethod
    def _is_retryable_request(method: str, path: str) -> bool:
        # Avoid blind retries on order-placement endpoints to reduce duplicate-order risk.
        if method != "POST":
            return True
        if path in {
            "/fapi/v1/order",
            "/papi/v1/um/algo/order",
            "/papi/v1/um/conditional/order",
        }:
            return False
        return True

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _round_qty(self, qty: float, symbol: str) -> float:
        step = self._get_exchange_info(symbol).get("step_size", 0.001)
        if step <= 0:
            return round(qty, 3)
        precision = max(0, -int(math.floor(math.log10(step))))
        factor = 10 ** precision
        return math.floor(qty * factor) / factor

    def _tick_round(self, price: float, symbol: str) -> float:
        tick = self._get_exchange_info(symbol).get("tick_size", 0.01)
        if tick <= 0:
            return round(price, 2)
        precision = max(0, -int(math.floor(math.log10(tick))))
        return round(round(price / tick) * tick, precision)

    def _get_exchange_info(self, symbol: str) -> dict:
        if symbol in self._exchange_cache:
            return self._exchange_cache[symbol]
        # Attempt live fetch from testnet
        try:
            r = self.session.get(
                f"{TESTNET_BASE}/fapi/v1/exchangeInfo",
                timeout=10
            )
            if r.status_code == 200:
                for s in r.json().get("symbols", []):
                    if s["symbol"] == symbol:
                        info = {}
                        for f in s.get("filters", []):
                            ft = f["filterType"]
                            if ft == "PRICE_FILTER":
                                info["tick_size"] = float(f["tickSize"])
                            elif ft == "LOT_SIZE":
                                info["step_size"] = float(f["stepSize"])
                                info["min_qty"] = float(f["minQty"])
                        self._exchange_cache[symbol] = info
                        return info
        except Exception:
            pass
        return {"tick_size": 0.01, "step_size": 0.001, "min_qty": 0.001}
