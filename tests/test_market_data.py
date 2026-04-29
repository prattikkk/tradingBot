import datetime
from decimal import Decimal

from alphabot.data.models import Candle
from alphabot.data.websocket_client import BinanceWebSocketClient


def _make_candle(minutes_ago: int, minutes_span: int, close: str) -> Candle:
    now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    open_time = now - datetime.timedelta(minutes=minutes_ago)
    close_time = open_time + datetime.timedelta(minutes=minutes_span)
    return Candle(
        symbol="BTCUSDT",
        timeframe="15m",
        open_time=open_time,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal("100"),
        close_time=close_time,
        is_closed=close_time <= now,
    )


def test_latest_closed_candle_skips_open_bar():
    closed_older = _make_candle(minutes_ago=45, minutes_span=15, close="100")
    closed_latest = _make_candle(minutes_ago=30, minutes_span=15, close="101")
    still_open = _make_candle(minutes_ago=5, minutes_span=15, close="102")

    result = BinanceWebSocketClient._latest_closed_candle(
        [closed_older, closed_latest, still_open]
    )

    assert result is not None
    assert result.open_time == closed_latest.open_time
    assert result.close == closed_latest.close