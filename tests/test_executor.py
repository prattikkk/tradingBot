import json
import unittest
from unittest import mock

from core import executor as executor_module
from core.executor import TestnetExecutor
from core.risk_manager import PositionSize
from core.signal import Direction
from config import CONFIG


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = int(status_code)
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.headers = {}

    def post(self, url, timeout=10):
        return self.response

    def get(self, url, timeout=10):
        return self.response

    def delete(self, url, timeout=10):
        return self.response


class ExecutorLoggingTests(unittest.TestCase):
    @staticmethod
    def _sample_position() -> PositionSize:
        return PositionSize(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=0.01,
            notional_usdt=1.0,
            stop_loss=99.0,
            take_profit_1=101.0,
            take_profit_2=102.0,
            risk_usdt=0.1,
            leverage=1,
        )

    def _build_executor(self, response):
        ex = TestnetExecutor()
        ex.api_key = "test-key"
        ex.api_secret = "test-secret"
        ex.session = _FakeSession(response)
        ex._retry_attempts = 0
        return ex

    def test_margin_error_is_warning(self):
        ex = self._build_executor(_FakeResponse(400, {"code": -2019, "msg": "Margin is insufficient."}))

        with mock.patch.object(executor_module, "log") as mock_log:
            result = ex._signed_post(
                "/fapi/v1/order",
                {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": 0.01},
            )

        self.assertIsNone(result)
        self.assertEqual(ex._last_api_error_code, -2019)
        self.assertTrue(mock_log.warning.called)
        self.assertFalse(mock_log.error.called)

    def test_margin_error_does_not_open_circuit(self):
        ex = self._build_executor(_FakeResponse(400, {"code": -2019, "msg": "Margin is insufficient."}))
        endpoint_key = "POST:/fapi/v1/order"

        for _ in range(6):
            ex._signed_post(
                "/fapi/v1/order",
                {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": 0.01},
            )

        self.assertTrue(ex._circuit_breaker.allow(endpoint_key))

    def test_unsupported_endpoint_error_is_info(self):
        ex = self._build_executor(
            _FakeResponse(
                400,
                {
                    "code": -4120,
                    "msg": "Order type not supported for this endpoint. Please use the Algo Order API endpoints instead.",
                },
            )
        )

        with mock.patch.object(executor_module, "log") as mock_log:
            result = ex._signed_post(
                "/fapi/v1/order",
                {
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "type": "TRAILING_STOP_MARKET",
                    "quantity": 0.01,
                    "callbackRate": 0.5,
                },
            )

        self.assertIsNone(result)
        self.assertEqual(ex._last_api_error_code, -4120)
        self.assertTrue(mock_log.info.called)
        self.assertFalse(mock_log.error.called)

    def test_unknown_api_error_stays_error(self):
        ex = self._build_executor(_FakeResponse(400, {"code": -9999, "msg": "Unknown error"}))

        with mock.patch.object(executor_module, "log") as mock_log:
            result = ex._signed_get("/fapi/v2/account", {})

        self.assertIsNone(result)
        self.assertEqual(ex._last_api_error_code, -9999)
        self.assertTrue(mock_log.error.called)

    def test_open_position_margin_reject_is_not_error(self):
        ex = TestnetExecutor()
        ex._last_api_error_code = -2019

        pos = PositionSize(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry_price=100.0,
            quantity=0.01,
            notional_usdt=1.0,
            stop_loss=99.0,
            take_profit_1=101.0,
            take_profit_2=102.0,
            risk_usdt=0.1,
            leverage=1,
        )

        with mock.patch.object(ex, "_set_leverage", return_value=None), mock.patch.object(
            ex, "_place_order", return_value=None
        ), mock.patch.object(executor_module, "log") as mock_log:
            result = ex.open_position(pos)

        self.assertEqual(result, {})
        self.assertTrue(mock_log.warning.called)
        self.assertFalse(mock_log.error.called)

    def test_strict_protection_requires_exchange_orders(self):
        ex = TestnetExecutor()
        pos = self._sample_position()

        with mock.patch.object(CONFIG.trading, "strict_protection_required", True), mock.patch.object(
            CONFIG.trading,
            "exchange_protective_orders",
            False,
        ), mock.patch.object(ex, "_set_leverage", return_value=None), mock.patch.object(
            ex,
            "_place_order",
            return_value={"orderId": 123},
        ) as mock_entry:
            result = ex.open_position(pos)

        self.assertEqual(result, {})
        mock_entry.assert_not_called()

    def test_strict_protection_flattens_when_protection_incomplete(self):
        ex = TestnetExecutor()
        pos = self._sample_position()

        with mock.patch.object(CONFIG.trading, "strict_protection_required", True), mock.patch.object(
            CONFIG.trading,
            "exchange_protective_orders",
            True,
        ), mock.patch.object(ex, "_set_leverage", return_value=None), mock.patch.object(
            ex,
            "_place_order",
            return_value={"orderId": 123},
        ), mock.patch.object(
            ex,
            "_place_protective_order",
            side_effect=[
                None,
                {"accepted": True, "client_id": "tp1_pending"},
                {"accepted": True, "client_id": "tp2_pending"},
            ],
        ), mock.patch.object(
            ex,
            "close_position_market",
            return_value=True,
        ) as mock_flatten:
            result = ex.open_position(pos)

        self.assertEqual(result, {})
        mock_flatten.assert_called_once_with("BTCUSDT", Direction.LONG, 0.01)

    def test_strict_protection_flattens_when_orders_unconfirmed(self):
        ex = TestnetExecutor()
        pos = self._sample_position()

        with mock.patch.object(CONFIG.trading, "strict_protection_required", True), mock.patch.object(
            CONFIG.trading,
            "exchange_protective_orders",
            True,
        ), mock.patch.object(ex, "_set_leverage", return_value=None), mock.patch.object(
            ex,
            "_place_order",
            return_value={"orderId": 123},
        ), mock.patch.object(
            ex,
            "_place_protective_order",
            side_effect=[
                {"accepted": True, "client_id": "sl_pending"},
                {"accepted": True, "client_id": "tp1_pending"},
                {"accepted": True, "client_id": "tp2_pending"},
            ],
        ), mock.patch.object(
            ex,
            "close_position_market",
            return_value=True,
        ) as mock_flatten:
            result = ex.open_position(pos)

        self.assertEqual(result, {})
        mock_flatten.assert_called_once_with("BTCUSDT", Direction.LONG, 0.01)

    def test_strict_protection_allows_pending_when_confirmation_disabled(self):
        ex = TestnetExecutor()
        pos = self._sample_position()

        with mock.patch.object(CONFIG.trading, "strict_protection_required", True), mock.patch.object(
            CONFIG.trading,
            "exchange_protective_orders",
            True,
        ), mock.patch.object(
            CONFIG.trading,
            "strict_protection_require_confirmed_ids",
            False,
        ), mock.patch.object(ex, "_set_leverage", return_value=None), mock.patch.object(
            ex,
            "_place_order",
            return_value={"orderId": 123},
        ), mock.patch.object(
            ex,
            "_place_protective_order",
            side_effect=[
                {"accepted": True, "client_id": "sl_pending"},
                {"accepted": True, "client_id": "tp1_pending"},
                {"accepted": True, "client_id": "tp2_pending"},
            ],
        ), mock.patch.object(
            ex,
            "close_position_market",
            return_value=True,
        ) as mock_flatten:
            result = ex.open_position(pos)

        self.assertEqual(
            result,
            {
                "entry": 123,
                "sl": "pending:sl_pending",
                "tp1": "pending:tp1_pending",
                "tp2": "pending:tp2_pending",
            },
        )
        mock_flatten.assert_not_called()


if __name__ == "__main__":
    unittest.main()
