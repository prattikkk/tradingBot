"""
Trading Bot Configuration
=========================
Single active configuration model for AlphaBot.
"""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


class _Section:
    """Small helper for dot-access config sections."""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


class _Config:
    """Primary runtime configuration used across bot modules."""

    def __init__(self):
        self.exchange = _Section(
            name=os.getenv("EXCHANGE", "binance").strip().lower(),
        )

        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        self.binance = _Section(
            api_key=api_key,
            api_secret=api_secret,
            testnet=_env_bool("BINANCE_TESTNET", True),
            testnet_api_key=os.getenv("BINANCE_TESTNET_API_KEY", api_key),
            testnet_secret=os.getenv("BINANCE_TESTNET_SECRET", api_secret),
            trading_mode=os.getenv("TRADING_MODE", "spot").lower(),
            quote_asset=os.getenv("QUOTE_ASSET", "USDT"),
        )

        self.ai = _Section(
            provider=self._default_ai_provider(),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", os.getenv("GLM_TEMPERATURE", "0.3"))),
            openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", os.getenv("GLM_MAX_TOKENS", "2048"))),
            openai_timeout_seconds=int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", os.getenv("GLM_TEMPERATURE", "0.3"))),
            gemini_max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", os.getenv("GLM_MAX_TOKENS", "2048"))),
            gemini_timeout_seconds=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30")),
            glm_api_key=os.getenv("GLM_API_KEY", ""),
            glm_base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            glm_model=os.getenv("GLM_MODEL", "glm-5.1"),
            glm_temperature=float(os.getenv("GLM_TEMPERATURE", "0.3")),
            glm_max_tokens=int(os.getenv("GLM_MAX_TOKENS", "2048")),
            sentiment_enabled=_env_bool("AI_SENTIMENT_ENABLED", False),
            sentiment_timeout_seconds=int(os.getenv("AI_SENTIMENT_TIMEOUT_SECONDS", "8")),
            sentiment_cache_seconds=int(os.getenv("AI_SENTIMENT_CACHE_SECONDS", "300")),
            sentiment_max_adjustment=float(os.getenv("AI_SENTIMENT_MAX_ADJUSTMENT", "0.10")),
        )

        self.api = _Section(
            rate_limit_per_minute=int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "1000")),
            retry_attempts=int(os.getenv("API_RETRY_ATTEMPTS", "3")),
            backoff_base_seconds=float(os.getenv("API_BACKOFF_BASE_SECONDS", "0.5")),
            backoff_cap_seconds=float(os.getenv("API_BACKOFF_CAP_SECONDS", "8.0")),
            circuit_failures=int(os.getenv("API_CIRCUIT_FAILURES", "5")),
            circuit_cooldown_seconds=int(os.getenv("API_CIRCUIT_COOLDOWN_SECONDS", "30")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "8")),
            notify_rate_limit_per_minute=int(os.getenv("NOTIFY_RATE_LIMIT_PER_MINUTE", "30")),
        )

        self.strategy = _Section(
            primary_tf=os.getenv("PRIMARY_TF", "1h"),
            htf_1=os.getenv("HTF_1", "4h"),
            htf_2=os.getenv("HTF_2", "1d"),
            align_signal_to_new_candle=_env_bool("ALIGN_SIGNAL_TO_NEW_CANDLE", True),
            no_signal_fallback_enabled=_env_bool("NO_SIGNAL_FALLBACK_ENABLED", True),
            no_signal_fallback_cycles=int(os.getenv("NO_SIGNAL_FALLBACK_CYCLES", "3")),
            no_signal_fallback_confidence_relaxation=float(
                os.getenv("NO_SIGNAL_FALLBACK_CONFIDENCE_RELAXATION", "0.05")
            ),
            symbol_calibration_path=os.getenv("SYMBOL_CALIBRATION_PATH", "data/symbol_calibration.json"),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.60")),
            use_support_resistance=_env_bool("USE_SUPPORT_RESISTANCE", True),
            ensemble_min_signals=int(os.getenv("ENSEMBLE_MIN_SIGNALS", "2")),
            regime_adx_trending=float(os.getenv("REGIME_ADX_TRENDING", "23")),
            regime_vol_window=int(os.getenv("REGIME_VOL_WINDOW", "80")),
            regime_high_vol_quantile=float(os.getenv("REGIME_HIGH_VOL_QUANTILE", "0.80")),
            supertrend_period=int(os.getenv("SUPERTREND_PERIOD", "10")),
            supertrend_multiplier=float(os.getenv("SUPERTREND_MULTIPLIER", "3.0")),
            rsi_period=int(os.getenv("RSI_PERIOD", "14")),
            rsi_oversold=float(os.getenv("RSI_OVERSOLD", "35")),
            rsi_overbought=float(os.getenv("RSI_OVERBOUGHT", "65")),
            ema_fast=int(os.getenv("EMA_FAST", "9")),
            ema_slow=int(os.getenv("EMA_SLOW", "21")),
            ema_trend=int(os.getenv("EMA_TREND", "50")),
            adx_period=int(os.getenv("ADX_PERIOD", "14")),
            adx_threshold=float(os.getenv("ADX_THRESHOLD", "25")),
            volume_ma_period=int(os.getenv("VOLUME_MA_PERIOD", "20")),
            volume_spike_ratio=float(os.getenv("VOLUME_SPIKE_RATIO", "1.5")),
            breakout_period=int(os.getenv("BREAKOUT_PERIOD", "20")),
            adx_trend_fast_ema=int(os.getenv("ADX_TREND_FAST_EMA", "5")),
            adx_trend_slow_ema=int(os.getenv("ADX_TREND_SLOW_EMA", "13")),
            adx_trend_adx_period=int(os.getenv("ADX_TREND_ADX_PERIOD", "14")),
            adx_trend_adx_min=float(os.getenv("ADX_TREND_ADX_MIN", "22")),
            adx_trend_adx_rising_bars=int(os.getenv("ADX_TREND_ADX_RISING_BARS", "0")),
            adx_trend_vol_min=float(os.getenv("ADX_TREND_VOL_MIN", "0.7")),
            adx_trend_rsi_low=float(os.getenv("ADX_TREND_RSI_LOW", "25")),
            adx_trend_rsi_high=float(os.getenv("ADX_TREND_RSI_HIGH", "75")),
            adx_trend_body_atr_min=float(os.getenv("ADX_TREND_BODY_ATR_MIN", "0.2")),
            adx_trend_sl_mult=float(os.getenv("ADX_TREND_SL_MULT", "2.0")),
            adx_trend_tp1_mult=float(os.getenv("ADX_TREND_TP1_MULT", "2.0")),
            adx_trend_tp2_mult=float(os.getenv("ADX_TREND_TP2_MULT", "5.0")),
            adx_trend_htf_required=_env_bool("ADX_TREND_HTF_REQUIRED", False),
        )

        self.risk = _Section(
            initial_capital=float(os.getenv("INITIAL_CAPITAL_USDT", "1000")),
            max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", "0.015")),
            portfolio_risk_basis=os.getenv("PORTFOLIO_RISK_BASIS", "equity").strip().lower(),
            adaptive_sizing_enabled=_env_bool("ADAPTIVE_SIZING_ENABLED", True),
            adaptive_recent_trades=int(os.getenv("ADAPTIVE_RECENT_TRADES", "20")),
            adaptive_min_multiplier=float(os.getenv("ADAPTIVE_MIN_MULTIPLIER", "0.5")),
            adaptive_max_multiplier=float(os.getenv("ADAPTIVE_MAX_MULTIPLIER", "1.5")),
            correlation_management_enabled=_env_bool("CORRELATION_MANAGEMENT_ENABLED", True),
            correlation_threshold=float(os.getenv("CORRELATION_THRESHOLD", "0.80")),
            correlation_lookback=int(os.getenv("CORRELATION_LOOKBACK", "120")),
            max_correlated_positions=int(os.getenv("MAX_CORRELATED_POSITIONS", "1")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "4")),
            max_portfolio_risk=float(os.getenv("MAX_PORTFOLIO_RISK", "0.06")),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05")),
            leverage=int(os.getenv("LEVERAGE", "5")),
            atr_sl_multiplier=float(os.getenv("ATR_SL_MULTIPLIER", "1.5")),
            atr_tp1_multiplier=float(os.getenv("ATR_TP1_MULTIPLIER", "2.5")),
            atr_tp2_multiplier=float(os.getenv("ATR_TP2_MULTIPLIER", "4.0")),
            partial_exit_pct=float(os.getenv("PARTIAL_EXIT_PCT", "0.5")),
            min_rr_ratio=float(os.getenv("MIN_RR_RATIO", "1.5")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.03")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.06")),
            trailing_stop_pct=float(os.getenv("TRAILING_STOP_PCT", "0.015")),
        )

        default_blacklist = "USDCUSDT,BUSDUSDT,TUSDUSDT,USDPUSDT,FDUSDUSDT,DAIUSDT,EURUSDT,GBPUSDT"
        whitelist = _csv_env("WHITELIST")

        self.trading = _Section(
            candle_limit=int(os.getenv("CANDLE_LIMIT", os.getenv("CANDLES_PER_TIMEFRAME", "250"))),
            analysis_interval=int(os.getenv("ANALYSIS_INTERVAL", "300")),
            execution_timeframe=os.getenv("EXECUTION_TIMEFRAME", "1h"),
            use_websocket_data=_env_bool("USE_WEBSOCKET_DATA", True),
            reconciliation_enabled=_env_bool("RECONCILIATION_ENABLED", True),
            reconciliation_interval_seconds=float(os.getenv("RECONCILIATION_INTERVAL_SECONDS", "300")),
            force_close_untracked_exchange_positions=_env_bool("FORCE_CLOSE_UNTRACKED_EXCHANGE_POSITIONS", True),
            exchange_protective_orders=_env_bool("EXCHANGE_PROTECTIVE_ORDERS", False),
            strict_protection_required=_env_bool("STRICT_PROTECTION_REQUIRED", False),
            strict_protection_require_confirmed_ids=_env_bool("STRICT_PROTECTION_REQUIRE_CONFIRMED_IDS", True),
            position_check_interval_seconds=float(os.getenv("POSITION_CHECK_INTERVAL_SECONDS", "30")),
            min_volume_24h=float(os.getenv("MIN_VOLUME_24H", "5000000")),
            max_unfavorable_funding_rate=float(os.getenv("MAX_UNFAVORABLE_FUNDING_RATE", "0.0005")),
            max_hold_hours=float(os.getenv("MAX_HOLD_HOURS", "48")),
            backtest_slippage_bps=float(os.getenv("BACKTEST_SLIPPAGE_BPS", "5")),
            backtest_spread_bps=float(os.getenv("BACKTEST_SPREAD_BPS", "2")),
            backtest_taker_fee_bps=float(os.getenv("BACKTEST_TAKER_FEE_BPS", "4")),
            backtest_funding_rate_8h=float(os.getenv("BACKTEST_FUNDING_RATE_8H", "0.0005")),
            backtest_maintenance_margin_rate=float(os.getenv("BACKTEST_MAINTENANCE_MARGIN_RATE", "0.005")),
            backtest_liquidation_fee_bps=float(os.getenv("BACKTEST_LIQUIDATION_FEE_BPS", "30")),
            backtest_conservative_ohlc_path=_env_bool("BACKTEST_CONSERVATIVE_OHLC_PATH", True),
            backtest_max_hold_hours=float(os.getenv("BACKTEST_MAX_HOLD_HOURS", os.getenv("MAX_HOLD_HOURS", "48"))),
            max_positions=int(os.getenv("MAX_POSITIONS", "5")),
            close_on_shutdown=_env_bool("CLOSE_ON_SHUTDOWN", False),
            blacklist=_csv_env("BLACKLIST", default_blacklist),
            whitelist=whitelist if whitelist else None,
            dry_run=_env_bool("DRY_RUN", True),
            timeframes=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w", "1M"],
            analysis_timeframes=["5m", "15m", "1h", "4h", "1d", "1w"],
        )

        self.indicators = _Section(
            sma_periods=[20, 50, 100, 200],
            ema_periods=[9, 21, 55, 200],
            rsi_period=14,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            bb_period=20,
            bb_std=2,
            atr_period=14,
            volume_sma=20,
            stoch_k=14,
            stoch_d=3,
        )

        self.logging = _Section(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file=os.getenv("LOG_FILE", "trading_bot.log"),
            format=os.getenv("LOG_FORMAT", "text").strip().lower(),
        )

        self.dashboard = _Section(
            host=os.getenv("DASHBOARD_HOST", "127.0.0.1"),
            port=int(os.getenv("DASHBOARD_PORT", "8080")),
        )

    @staticmethod
    def _default_ai_provider() -> str:
        if os.getenv("AI_PROVIDER"):
            return os.getenv("AI_PROVIDER", "glm").lower()
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        return "glm"

    @property
    def AI_MODEL(self) -> str:
        if self.ai.provider == "openai":
            return self.ai.openai_model
        if self.ai.provider == "gemini":
            return self.ai.gemini_model
        return self.ai.glm_model

    @property
    def ACTIVE_AI_KEY(self) -> str:
        if self.ai.provider == "openai":
            return self.ai.openai_api_key
        if self.ai.provider == "gemini":
            return self.ai.gemini_api_key
        return self.ai.glm_api_key

    @property
    def ACTIVE_AI_KEY_NAME(self) -> str:
        if self.ai.provider == "openai":
            return "OPENAI_API_KEY"
        if self.ai.provider == "gemini":
            return "GEMINI_API_KEY"
        return "GLM_API_KEY"


# Backward-compatible alias and active instance.
Config = _Config
CONFIG = _Config()
