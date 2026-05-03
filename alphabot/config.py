"""
AlphaBot Configuration — Pydantic Settings
Loads from .env and config.yaml, validates on startup.
All parameters are type-safe and validated at boot.
"""

from __future__ import annotations

import yaml
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


_BASE_DIR = Path(__file__).resolve().parent.parent


def _load_yaml_config() -> dict:
    """Load config.yaml and return as dict."""
    yaml_path = _BASE_DIR / "config.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


_YAML = _load_yaml_config()


def _yaml_get(*keys: str, default: Any = None) -> Any:
    """Safely fetch nested values from config.yaml."""
    node: Any = _YAML
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def _as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_decimal(value, fallback: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(fallback)


def _as_str_list(value, fallback: List[str]) -> List[str]:
    if value is None:
        return fallback
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        items = [p for p in parts if p]
        return items or fallback
    return fallback


def _read_env_file_pairs() -> Dict[str, str]:
    """Read KEY=VALUE pairs from .env without exporting them to process env."""
    env_path = _BASE_DIR / ".env"
    if not env_path.exists():
        return {}

    result: Dict[str, str] = {}
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            result[key] = value.strip().strip('"').strip("'")
    except Exception:
        return {}

    return result


class Settings(BaseSettings):
    """Master settings — merges .env + config.yaml with validation."""

    # ---- Binance API Keys ----
    binance_mainnet_api_key: str = ""
    binance_mainnet_secret: str = ""
    binance_testnet_api_key: str = "your_testnet_api_key_here"
    binance_testnet_secret: str = "your_testnet_secret_here"

    # ---- Telegram ----
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ---- Environment ----
    environment: str = Field(default="testnet", pattern=r"^(testnet|mainnet)$")

    # ---- Trading Parameters ----
    trading_pairs: List[str] = Field(default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    primary_timeframe: str = Field(default="15m")
    candle_lookback: int = Field(default=200, ge=50, le=1000)

    # ---- Multi-timeframe ----
    entry_timeframes: List[str] = Field(
        default=_as_str_list(_yaml_get("multi_timeframe", "entry_timeframes", default=None), ["15m"])
    )
    bias_timeframes: List[str] = Field(
        default=_as_str_list(_yaml_get("multi_timeframe", "bias_timeframes", default=None), ["1h", "4h"])
    )
    # FIX[5]: Bound HTF freshness gate in units of bars to prevent stale bias alignment.
    max_htf_staleness_bars: float = Field(
        default=_as_float(_yaml_get("multi_timeframe", "max_htf_staleness_bars", default=2.5), 2.5),
        ge=1.0,
        le=10.0,
    )

    # ---- Risk Parameters ----
    risk_per_trade_pct: Decimal = Field(
        default=_as_decimal(_yaml_get("risk", "risk_per_trade_pct", default="1.0"), "1.0"),
        ge=Decimal("0.5"),
        le=Decimal("2.0"),
    )
    max_risk_per_trade_pct: Decimal = Field(
        default=_as_decimal(_yaml_get("risk", "max_risk_per_trade_pct", default="2.0"), "2.0")
    )
    daily_loss_cap_pct: Decimal = Field(
        default=_as_decimal(_yaml_get("risk", "daily_loss_cap_pct", default="2.0"), "2.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
    )
    max_drawdown_pct: Decimal = Field(
        default=_as_decimal(_yaml_get("risk", "max_drawdown_pct", default="5.0"), "5.0"),
        ge=Decimal("3.0"),
        le=Decimal("10.0"),
    )
    max_concurrent_positions: int = Field(
        default=_as_int(_yaml_get("risk", "max_concurrent_positions", default=3), 3),
        ge=1,
        le=5,
    )
    max_leverage: int = Field(
        default=_as_int(_yaml_get("risk", "max_leverage", default=5), 5),
        ge=1,
        le=20,
    )
    margin_mode: str = Field(
        default=str(_yaml_get("risk", "margin_mode", default="isolated") or "isolated").lower(),
        pattern=r"^(isolated|cross)$",
    )
    min_signal_confidence: int = Field(
        default=_as_int(_yaml_get("signal_scoring", "min_confidence", default=68), 68),
        ge=50,
        le=90,
    )
    min_risk_reward: Decimal = Field(
        default=_as_decimal(_yaml_get("risk", "min_risk_reward", default="1.5"), "1.5"),
        ge=Decimal("1.2"),
        le=Decimal("3.0"),
    )
    min_net_risk_reward: Decimal = Field(
        default=_as_decimal(_yaml_get("risk", "min_net_risk_reward", default="1.15"), "1.15"),
        ge=Decimal("1.0"),
        le=Decimal("3.0"),
    )
    estimated_roundtrip_fee_rate: Decimal = Field(
        default=_as_decimal(
            _yaml_get("risk", "estimated_roundtrip_fee_rate", default="0.0008"), "0.0008"
        ),
        ge=Decimal("0.0"),
        le=Decimal("0.005"),
    )
    min_stop_distance_pct: Decimal = Field(
        default=_as_decimal(_yaml_get("risk", "min_stop_distance_pct", default="0.5"), "0.5"),
        ge=Decimal("0.0"),
        le=Decimal("5.0"),
    )
    blocked_strategy_directions: List[str] = Field(
        default=_as_str_list(_yaml_get("risk", "blocked_strategy_directions", default=None), [])
    )

    # ---- Position Management ----
    trailing_stop_activation_r: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.5"), le=Decimal("2.0"))
    breakeven_activation_r: Decimal = Field(
        default=_as_decimal(_yaml_get("risk", "breakeven_activation_r", default="0.8"), "0.8"),
        ge=Decimal("0.3"),
        le=Decimal("2.0"),
    )

    # ---- Indicator Parameters ----
    rsi_period: int = Field(
        # FIX[8]: Enforce RSI period floor at 14 to avoid unstable/noisy signal generation.
        default=_as_int(_yaml_get("indicators", "rsi_period", default=14), 14),
        ge=14,
        le=50,
    )
    volume_sma_period: int = Field(
        default=_as_int(_yaml_get("indicators", "volume_sma_period", default=20), 20),
        ge=5,
        le=200,
    )
    ema_slope_period: int = Field(
        default=_as_int(_yaml_get("indicators", "ema_slope_period", default=5), 5),
        ge=2,
        le=20,
    )
    ema_long_period: int = Field(
        default=_as_int(_yaml_get("indicators", "ema_long_period", default=200), 200),
        ge=50,
        le=400,
    )
    supertrend_period: int = Field(
        default=_as_int(_yaml_get("indicators", "supertrend_period", default=10), 10),
        ge=5,
        le=50,
    )
    supertrend_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("indicators", "supertrend_multiplier", default="3.0"), "3.0"),
        ge=Decimal("1.0"),
        le=Decimal("5.0"),
    )

    # ---- Strategy Parameters: Supertrend + RSI ----
    supertrend_rsi_long_min: int = Field(
        default=_as_int(_yaml_get("strategies", "supertrend_rsi", "rsi_long_min", default=55), 55),
        ge=40,
        le=70,
    )
    supertrend_rsi_short_max: int = Field(
        default=_as_int(_yaml_get("strategies", "supertrend_rsi", "rsi_short_max", default=45), 45),
        ge=30,
        le=60,
    )
    supertrend_volume_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_rsi", "volume_multiplier", default="1.1"), "1.1"),
        ge=Decimal("0.5"),
        le=Decimal("3.0"),
    )
    supertrend_atr_sl_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_rsi", "atr_sl_multiplier", default="1.6"), "1.6"),
        ge=Decimal("0.8"),
        le=Decimal("3.0"),
    )
    supertrend_atr_tp1_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_rsi", "atr_tp1_multiplier", default="2.5"), "2.5"),
        ge=Decimal("1.0"),
        le=Decimal("6.0"),
    )
    supertrend_atr_tp2_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_rsi", "atr_tp2_multiplier", default="4.0"), "4.0"),
        ge=Decimal("1.5"),
        le=Decimal("10.0"),
    )
    supertrend_max_extension_atr: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_rsi", "max_extension_atr", default="1.5"), "1.5"),
        ge=Decimal("0.5"),
        le=Decimal("4.0"),
    )

    # ---- Strategy Parameters: Supertrend Pullback ----
    supertrend_pullback_rsi_long_min: int = Field(
        default=_as_int(_yaml_get("strategies", "supertrend_pullback", "rsi_long_min", default=54), 54),
        ge=40,
        le=75,
    )
    supertrend_pullback_rsi_short_max: int = Field(
        default=_as_int(_yaml_get("strategies", "supertrend_pullback", "rsi_short_max", default=46), 46),
        ge=25,
        le=60,
    )
    supertrend_pullback_volume_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_pullback", "volume_multiplier", default="1.15"), "1.15"),
        ge=Decimal("0.8"),
        le=Decimal("3.0"),
    )
    supertrend_pullback_adx_min: int = Field(
        default=_as_int(_yaml_get("strategies", "supertrend_pullback", "adx_min", default=20), 20),
        ge=10,
        le=45,
    )
    supertrend_pullback_band_atr: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_pullback", "pullback_band_atr", default="0.6"), "0.6"),
        ge=Decimal("0.2"),
        le=Decimal("2.0"),
    )
    supertrend_pullback_atr_sl_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_pullback", "atr_sl_multiplier", default="1.4"), "1.4"),
        ge=Decimal("0.8"),
        le=Decimal("3.0"),
    )
    supertrend_pullback_atr_tp1_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_pullback", "atr_tp1_multiplier", default="2.8"), "2.8"),
        ge=Decimal("1.2"),
        le=Decimal("8.0"),
    )
    supertrend_pullback_atr_tp2_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_pullback", "atr_tp2_multiplier", default="4.6"), "4.6"),
        ge=Decimal("1.8"),
        le=Decimal("12.0"),
    )
    supertrend_pullback_max_extension_atr: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_pullback", "max_extension_atr", default="1.2"), "1.2"),
        ge=Decimal("0.5"),
        le=Decimal("4.0"),
    )

    # ---- Strategy Parameters: Supertrend Trail ----
    supertrend_trail_rsi_long_min: int = Field(
        default=_as_int(_yaml_get("strategies", "supertrend_trail", "rsi_long_min", default=52), 52),
        ge=40,
        le=75,
    )
    supertrend_trail_rsi_short_max: int = Field(
        default=_as_int(_yaml_get("strategies", "supertrend_trail", "rsi_short_max", default=48), 48),
        ge=25,
        le=60,
    )
    supertrend_trail_volume_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_trail", "volume_multiplier", default="1.1"), "1.1"),
        ge=Decimal("0.8"),
        le=Decimal("3.0"),
    )
    supertrend_trail_adx_min: int = Field(
        default=_as_int(_yaml_get("strategies", "supertrend_trail", "adx_min", default=18), 18),
        ge=10,
        le=45,
    )
    supertrend_trail_breakout_buffer_atr: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_trail", "breakout_buffer_atr", default="0.1"), "0.1"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
    )
    supertrend_trail_atr_sl_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_trail", "atr_sl_multiplier", default="1.2"), "1.2"),
        ge=Decimal("0.8"),
        le=Decimal("3.0"),
    )
    supertrend_trail_atr_tp1_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_trail", "atr_tp1_multiplier", default="1.8"), "1.8"),
        ge=Decimal("1.0"),
        le=Decimal("8.0"),
    )
    supertrend_trail_atr_tp2_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_trail", "atr_tp2_multiplier", default="2.4"), "2.4"),
        ge=Decimal("1.2"),
        le=Decimal("12.0"),
    )
    supertrend_trail_max_extension_atr: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "supertrend_trail", "max_extension_atr", default="1.6"), "1.6"),
        ge=Decimal("0.5"),
        le=Decimal("4.0"),
    )

    # ---- Strategy Parameters: Order Flow + Liquidity Sweep ----
    orderflow_sweep_lookback: int = Field(
        default=_as_int(_yaml_get("strategies", "orderflow_liquidity_sweep", "lookback", default=20), 20),
        ge=10,
        le=100,
    )
    orderflow_sweep_min_wick_ratio: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "orderflow_liquidity_sweep", "min_wick_ratio", default="0.45"), "0.45"),
        ge=Decimal("0.2"),
        le=Decimal("0.9"),
    )
    orderflow_sweep_min_imbalance: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "orderflow_liquidity_sweep", "min_imbalance", default="0.12"), "0.12"),
        ge=Decimal("0.05"),
        le=Decimal("0.8"),
    )
    orderflow_sweep_rsi_long_max: int = Field(
        default=_as_int(_yaml_get("strategies", "orderflow_liquidity_sweep", "rsi_long_max", default=52), 52),
        ge=35,
        le=65,
    )
    orderflow_sweep_rsi_short_min: int = Field(
        default=_as_int(_yaml_get("strategies", "orderflow_liquidity_sweep", "rsi_short_min", default=48), 48),
        ge=35,
        le=65,
    )
    orderflow_sweep_volume_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "orderflow_liquidity_sweep", "volume_multiplier", default="1.1"), "1.1"),
        ge=Decimal("0.8"),
        le=Decimal("3.0"),
    )
    orderflow_sweep_stop_buffer_atr: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "orderflow_liquidity_sweep", "stop_buffer_atr", default="0.2"), "0.2"),
        ge=Decimal("0.0"),
        le=Decimal("1.5"),
    )
    orderflow_sweep_atr_tp1_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "orderflow_liquidity_sweep", "atr_tp1_multiplier", default="1.8"), "1.8"),
        ge=Decimal("0.8"),
        le=Decimal("8.0"),
    )
    orderflow_sweep_atr_tp2_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "orderflow_liquidity_sweep", "atr_tp2_multiplier", default="3.2"), "3.2"),
        ge=Decimal("1.2"),
        le=Decimal("12.0"),
    )
    orderflow_sweep_max_reclaim_atr: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "orderflow_liquidity_sweep", "max_reclaim_atr", default="1.4"), "1.4"),
        ge=Decimal("0.4"),
        le=Decimal("4.0"),
    )

    # ---- Strategy Parameters: Liquidity Sweep + Order Flow ----
    liquidity_sweep_orderflow_enabled: bool = Field(
        default=bool(_yaml_get("strategies", "liquidity_sweep_orderflow", "enabled", default=True))
    )
    liquidity_sweep_orderflow_swing_lookback: int = Field(
        default=_as_int(_yaml_get("strategies", "liquidity_sweep_orderflow", "swing_lookback", default=10), 10),
        ge=3,
        le=100,
    )
    liquidity_sweep_orderflow_sweep_min_wick_pct: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "liquidity_sweep_orderflow", "sweep_min_wick_pct", default="0.05"), "0.05"),
        ge=Decimal("0.01"),
        le=Decimal("1.0"),
    )
    liquidity_sweep_orderflow_delta_window: int = Field(
        default=_as_int(_yaml_get("strategies", "liquidity_sweep_orderflow", "delta_window", default=5), 5),
        ge=2,
        le=100,
    )
    liquidity_sweep_orderflow_cvd_slope_window: int = Field(
        default=_as_int(_yaml_get("strategies", "liquidity_sweep_orderflow", "cvd_slope_window", default=20), 20),
        ge=5,
        le=300,
    )
    liquidity_sweep_orderflow_min_delta_ratio: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "liquidity_sweep_orderflow", "min_delta_ratio", default="0.1"), "0.1"),
        ge=Decimal("0.01"),
        le=Decimal("2.0"),
    )
    liquidity_sweep_orderflow_htf_ema_fast: int = Field(
        default=_as_int(_yaml_get("strategies", "liquidity_sweep_orderflow", "htf_ema_fast", default=20), 20),
        ge=5,
        le=200,
    )
    liquidity_sweep_orderflow_htf_ema_slow: int = Field(
        default=_as_int(_yaml_get("strategies", "liquidity_sweep_orderflow", "htf_ema_slow", default=50), 50),
        ge=10,
        le=400,
    )
    liquidity_sweep_orderflow_min_confidence: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "liquidity_sweep_orderflow", "min_confidence", default="0.45"), "0.45"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
    )
    liquidity_sweep_orderflow_atr_period: int = Field(
        default=_as_int(_yaml_get("strategies", "liquidity_sweep_orderflow", "atr_period", default=14), 14),
        ge=5,
        le=100,
    )
    liquidity_sweep_orderflow_sl_atr_mult: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "liquidity_sweep_orderflow", "sl_atr_mult", default="1.5"), "1.5"),
        ge=Decimal("0.5"),
        le=Decimal("6.0"),
    )
    liquidity_sweep_orderflow_tp1_atr_mult: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "liquidity_sweep_orderflow", "tp1_atr_mult", default="2.5"), "2.5"),
        ge=Decimal("1.0"),
        le=Decimal("10.0"),
    )
    liquidity_sweep_orderflow_tp2_atr_mult: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "liquidity_sweep_orderflow", "tp2_atr_mult", default="4.0"), "4.0"),
        ge=Decimal("1.2"),
        le=Decimal("15.0"),
    )

    # ---- Strategy Parameters: EMA + ADX + Volume ----
    ema_fast: int = Field(
        default=_as_int(_yaml_get("strategies", "ema_adx_volume", "ema_fast", default=9), 9),
        ge=5,
        le=50,
    )
    ema_slow: int = Field(
        default=_as_int(_yaml_get("strategies", "ema_adx_volume", "ema_slow", default=21), 21),
        ge=10,
        le=200,
    )
    ema_adx_min: int = Field(
        default=_as_int(_yaml_get("strategies", "ema_adx_volume", "adx_min", default=22), 22),
        ge=10,
        le=40,
    )
    ema_adx_volume_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "ema_adx_volume", "volume_multiplier", default="1.2"), "1.2"),
        ge=Decimal("0.8"),
        le=Decimal("3.0"),
    )
    ema_adx_atr_sl_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "ema_adx_volume", "atr_sl_multiplier", default="1.6"), "1.6"),
        ge=Decimal("0.8"),
        le=Decimal("3.0"),
    )
    ema_adx_atr_tp1_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "ema_adx_volume", "atr_tp1_multiplier", default="2.5"), "2.5"),
        ge=Decimal("1.0"),
        le=Decimal("6.0"),
    )
    ema_adx_atr_tp2_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "ema_adx_volume", "atr_tp2_multiplier", default="3.5"), "3.5"),
        ge=Decimal("1.5"),
        le=Decimal("10.0"),
    )
    ema_adx_max_entry_range_atr: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "ema_adx_volume", "max_entry_range_atr", default="1.7"), "1.7"),
        ge=Decimal("1.0"),
        le=Decimal("4.0"),
    )
    ema_adx_crossover_only: bool = Field(
        default=bool(_yaml_get("strategies", "ema_adx_volume", "crossover_only", default=False))
    )
    ema_adx_min_net_rr: Decimal = Field(
        default=_as_decimal(
            _yaml_get("strategies", "ema_adx_volume", "min_net_rr", default="1.15"), "1.15"
        ),
        ge=Decimal("1.0"),
        le=Decimal("3.0"),
    )

    # ---- Dashboard ----
    dashboard_host: str = Field(default="0.0.0.0")
    dashboard_port: int = Field(default=8080, ge=1024, le=65535)

    # ---- Market Data ----
    market_data_rest_fallback_enabled: bool = Field(
        default=bool(_yaml_get("market_data", "rest_fallback_enabled", default=True))
    )
    market_data_poll_interval_seconds: int = Field(
        default=_as_int(_yaml_get("market_data", "poll_interval_seconds", default=30), 30),
        ge=5,
        le=300,
    )

    # ---- Logging ----
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")

    # ---- Max Consecutive Losses ----
    max_consecutive_losses: int = Field(
        default=_as_int(_yaml_get("risk", "max_consecutive_losses", default=3), 3),
        ge=1,
        le=10,
    )
    consecutive_loss_cooldown_minutes: int = Field(
        default=_as_int(_yaml_get("risk", "consecutive_loss_cooldown_minutes", default=30), 30),
        ge=5,
        le=120,
    )

    # ---- Time stop ----
    time_stop_hours: int = Field(
        default=_as_int(_yaml_get("risk", "time_stop_hours", default=4), 4),
        ge=1,
        le=24,
    )
    time_stop_progress_pct: Decimal = Field(
        default=_as_decimal(_yaml_get("risk", "time_stop_progress_pct", default="20"), "20"),
        ge=Decimal("5"),
        le=Decimal("50"),
    )

    # ---- Take Profit ----
    tp1_r_multiple: Decimal = Field(default=Decimal("1.0"))
    tp1_close_pct: int = Field(default=50)
    tp2_r_multiple: Decimal = Field(default=Decimal("2.0"))
    tp2_close_pct: int = Field(default=30)
    trailing_pct: int = Field(default=20)

    # ---- Regime Detection ----
    adx_period: int = Field(default=_as_int(_yaml_get("regime_detection", "adx_period", default=14), 14))
    atr_period: int = Field(default=_as_int(_yaml_get("regime_detection", "atr_period", default=14), 14))
    adx_trending_threshold: int = Field(
        default=_as_int(_yaml_get("regime_detection", "adx_trending_threshold", default=25), 25)
    )
    adx_ranging_threshold: int = Field(
        default=_as_int(_yaml_get("regime_detection", "adx_ranging_threshold", default=20), 20)
    )
    atr_volatility_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("regime_detection", "atr_volatility_multiplier", default="2.0"), "2.0")
    )

    # ---- Cooldown ----
    min_candles_between_trades: int = Field(
        default=_as_int(_yaml_get("cooldown", "min_candles_between_trades", default=1), 1)
    )
    stale_order_minutes: int = Field(
        default=_as_int(_yaml_get("cooldown", "stale_order_minutes", default=30), 30)
    )
    # FIX[1]: Timeout used for exchange fill confirmation before position registration.
    order_fill_timeout_seconds: float = Field(
        default=_as_float(_yaml_get("execution", "order_fill_timeout_seconds", default=6.0), 6.0),
        ge=1.0,
        le=30.0,
    )

    @field_validator("trading_pairs", mode="before")
    @classmethod
    def parse_trading_pairs(cls, v):
        if isinstance(v, str):
            import json
            # First try direct JSON parsing (for JSON array format)
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(p).strip() for p in parsed if p]
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Fall back to comma-separated format
            # Remove brackets and quotes if JSON array format was attempted
            v = v.strip().lstrip('[').rstrip(']')
            # Split on commas and clean each pair
            pairs = [p.strip().strip('"') for p in v.split(",") if p.strip()]
            return [p for p in pairs if p]  # Filter out empty strings
        return v

    @property
    def is_testnet(self) -> bool:
        return self.environment == "testnet"

    @property
    def binance_futures_base_url(self) -> str:
        if self.is_testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"

    @property
    def binance_ws_base_url(self) -> str:
        """Always use Mainnet for price data."""
        return "wss://fstream.binance.com/ws"

    @property
    def binance_mainnet_rest_url(self) -> str:
        return "https://fapi.binance.com"

    model_config = {
        "env_file": str(_BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Singleton settings instance — import this everywhere
settings = Settings()


_ENV_AUDIT_MAP: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "PRIMARY_TIMEFRAME": ("primary_timeframe", ("primary_timeframe",)),
    "RISK_PER_TRADE_PCT": ("risk_per_trade_pct", ("risk", "risk_per_trade_pct")),
    "MAX_RISK_PER_TRADE_PCT": ("max_risk_per_trade_pct", ("risk", "max_risk_per_trade_pct")),
    "DAILY_LOSS_CAP_PCT": ("daily_loss_cap_pct", ("risk", "daily_loss_cap_pct")),
    "MAX_DRAWDOWN_PCT": ("max_drawdown_pct", ("risk", "max_drawdown_pct")),
    "MAX_CONCURRENT_POSITIONS": ("max_concurrent_positions", ("risk", "max_concurrent_positions")),
    "MAX_LEVERAGE": ("max_leverage", ("risk", "max_leverage")),
    "MIN_SIGNAL_CONFIDENCE": ("min_signal_confidence", ("signal_scoring", "min_confidence")),
    "MIN_RISK_REWARD": ("min_risk_reward", ("risk", "min_risk_reward")),
    "MIN_NET_RISK_REWARD": ("min_net_risk_reward", ("risk", "min_net_risk_reward")),
    "MIN_STOP_DISTANCE_PCT": ("min_stop_distance_pct", ("risk", "min_stop_distance_pct")),
    "STALE_ORDER_MINUTES": ("stale_order_minutes", ("cooldown", "stale_order_minutes")),
}


def get_effective_config_snapshot() -> Dict[str, object]:
    """Small runtime snapshot for startup logging and forensic verification."""
    return {
        "environment": settings.environment,
        "trading_pairs": list(settings.trading_pairs),
        "primary_timeframe": settings.primary_timeframe,
        "entry_timeframes": list(settings.entry_timeframes),
        "bias_timeframes": list(settings.bias_timeframes),
        "max_htf_staleness_bars": settings.max_htf_staleness_bars,
        "risk_per_trade_pct": str(settings.risk_per_trade_pct),
        "max_concurrent_positions": settings.max_concurrent_positions,
        "max_leverage": settings.max_leverage,
        "margin_mode": settings.margin_mode,
        "order_fill_timeout_seconds": settings.order_fill_timeout_seconds,
        "min_signal_confidence": settings.min_signal_confidence,
        "min_risk_reward": str(settings.min_risk_reward),
        "min_net_risk_reward": str(settings.min_net_risk_reward),
        "estimated_roundtrip_fee_rate": str(settings.estimated_roundtrip_fee_rate),
        "min_stop_distance_pct": str(settings.min_stop_distance_pct),
        "stale_order_minutes": settings.stale_order_minutes,
    }


def detect_env_yaml_overrides() -> List[str]:
    """Return human-readable warnings when .env overrides YAML values."""
    warnings: List[str] = []
    env_pairs = _read_env_file_pairs()

    for env_key, (attr_name, yaml_path) in _ENV_AUDIT_MAP.items():
        env_value = env_pairs.get(env_key)
        if env_value is None or env_value == "":
            continue

        yaml_value = _yaml_get(*yaml_path, default=None)
        if yaml_value is None:
            continue

        yaml_text = str(yaml_value).strip()
        env_text = str(env_value).strip()
        if yaml_text != env_text:
            effective = getattr(settings, attr_name, None)
            warnings.append(
                f".env override active: {env_key}={env_text} (yaml={yaml_text}, effective={effective})"
            )

    # FIX[7]: Security tripwire for committed API credentials in .env.
    for key_name in ("BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_SECRET"):
        value = (env_pairs.get(key_name) or "").strip()
        if not value:
            continue
        if value.lower().startswith("your_") or value.lower().startswith("replace_"):
            continue
        if len(value) >= 24:
            warnings.append(
                f"security warning: {key_name} appears populated in .env; rotate and move secret out of repository"
            )

    return warnings
