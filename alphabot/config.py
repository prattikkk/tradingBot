"""
AlphaBot Configuration — Pydantic Settings
Loads from .env and config.yaml, validates on startup.
All parameters are type-safe and validated at boot.
"""

from __future__ import annotations

import yaml
from decimal import Decimal
from pathlib import Path
from typing import List

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


def _yaml_get(*keys: str, default=None):
    """Safely fetch nested values from config.yaml."""
    node = _YAML
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


def _as_decimal(value, fallback: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(fallback)


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

    # ---- Strategy Parameters ----
    atr_sl_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "ema_crossover", "atr_sl_multiplier", default="1.5"), "1.5"),
        ge=Decimal("1.0"),
        le=Decimal("3.0"),
    )
    atr_tp_multiplier: Decimal = Field(
        default=_as_decimal(_yaml_get("strategies", "ema_crossover", "atr_tp_multiplier", default="3.0"), "3.0"),
        ge=Decimal("1.5"),
        le=Decimal("5.0"),
    )
    trailing_stop_activation_r: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.5"), le=Decimal("2.0"))
    ema_fast: int = Field(
        default=_as_int(_yaml_get("strategies", "ema_crossover", "ema_fast", default=20), 20),
        ge=5,
        le=50,
    )
    ema_slow: int = Field(
        default=_as_int(_yaml_get("strategies", "ema_crossover", "ema_slow", default=50), 50),
        ge=20,
        le=200,
    )
    rsi_oversold_long: int = Field(
        default=_as_int(_yaml_get("strategies", "bb_reversion", "rsi_oversold", default=35), 35),
        ge=20,
        le=45,
    )
    rsi_overbought_short: int = Field(
        default=_as_int(_yaml_get("strategies", "bb_reversion", "rsi_overbought", default=65), 65),
        ge=55,
        le=90,
    )
    stoch_rsi_oversold: int = Field(
        default=_as_int(_yaml_get("strategies", "bb_reversion", "stoch_rsi_oversold", default=20), 20),
        ge=5,
        le=40,
    )
    stoch_rsi_overbought: int = Field(
        default=_as_int(_yaml_get("strategies", "bb_reversion", "stoch_rsi_overbought", default=80), 80),
        ge=60,
        le=95,
    )

    # ---- Dashboard ----
    dashboard_host: str = Field(default="0.0.0.0")
    dashboard_port: int = Field(default=8080, ge=1024, le=65535)

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
    bb_period: int = Field(default=_as_int(_yaml_get("regime_detection", "bb_period", default=20), 20))
    bb_std: int = Field(default=_as_int(_yaml_get("regime_detection", "bb_std", default=2), 2))
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
