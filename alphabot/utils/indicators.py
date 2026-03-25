"""
AlphaBot Technical Indicator Helpers.
Implements EMA, ATR, RSI, ADX, Bollinger Bands, Stochastic RSI, and MACD
using pandas/numpy only.
All calculations on CLOSED candles only — never on the current in-progress candle.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
try:
    import pandas_ta as ta
except ImportError:  # pandas-ta-classic fallback
    import pandas_ta_classic as ta


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period, min_periods=period).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range — volatility measure."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index.
    Returns DataFrame with columns: ADX_{period}, DMP_{period}, DMN_{period}
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    alpha = 1.0 / period

    # Wilder-style smoothing for ADX family metrics.
    atr_val = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (plus_dm_smooth / atr_val.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_smooth / atr_val.replace(0, np.nan))

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / di_sum) * 100.0
    adx_val = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return pd.DataFrame(
        {
            f"ADX_{period}": adx_val,
            f"DMP_{period}": plus_di,
            f"DMN_{period}": minus_di,
        }
    )


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50.0)


def stochastic_rsi(close: pd.Series, period: int = 14,
                   rsi_period: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    """
    Stochastic RSI.
    Returns DataFrame with columns: STOCHRSIk_{period}_{rsi}_{k}_{d},
                                     STOCHRSId_{period}_{rsi}_{k}_{d}
    """
    rsi_series = rsi(close, rsi_period)
    rsi_min = rsi_series.rolling(period, min_periods=period).min()
    rsi_max = rsi_series.rolling(period, min_periods=period).max()
    denom = (rsi_max - rsi_min).replace(0, np.nan)

    stoch = ((rsi_series - rsi_min) / denom) * 100.0
    stoch_k = stoch.rolling(k, min_periods=k).mean()
    stoch_d = stoch_k.rolling(d, min_periods=d).mean()

    return pd.DataFrame(
        {
            f"STOCHRSIk_{period}_{rsi_period}_{k}_{d}": stoch_k,
            f"STOCHRSId_{period}_{rsi_period}_{k}_{d}": stoch_d,
        }
    )


def bollinger_bands(close: pd.Series, period: int = 20,
                    std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns DataFrame with columns: BBL, BBM, BBU, BBB, BBP
    (Lower, Mid, Upper, BandWidth, %B)
    """
    mid = close.rolling(period, min_periods=period).mean()
    sigma = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + std * sigma
    lower = mid - std * sigma
    width = upper - lower
    percent_b = (close - lower) / width.replace(0, np.nan)

    return pd.DataFrame(
        {
            f"BBL_{period}_{float(std)}": lower,
            f"BBM_{period}_{float(std)}": mid,
            f"BBU_{period}_{float(std)}": upper,
            f"BBB_{period}_{float(std)}": width,
            f"BBP_{period}_{float(std)}": percent_b,
        }
    )


def bollinger_width(close: pd.Series, period: int = 20,
                    std: float = 2.0) -> pd.Series:
    """Bollinger Band Width = (Upper - Lower) / Middle."""
    bb = bollinger_bands(close, period, std)
    if bb is None or bb.empty:
        return pd.Series(dtype=float)
    cols = bb.columns
    upper_col = [c for c in cols if c.startswith("BBU_")][0]
    lower_col = [c for c in cols if c.startswith("BBL_")][0]
    mid_col = [c for c in cols if c.startswith("BBM_")][0]
    return (bb[upper_col] - bb[lower_col]) / bb[mid_col]


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    MACD.
    Returns DataFrame with columns: MACD_fast_slow_signal,
                                     MACDh_fast_slow_signal,
                                     MACDs_fast_slow_signal
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    return pd.DataFrame(
        {
            f"MACD_{fast}_{slow}_{signal}": macd_line,
            f"MACDh_{fast}_{slow}_{signal}": hist,
            f"MACDs_{fast}_{slow}_{signal}": signal_line,
        }
    )


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average of volume."""
    return volume.rolling(period, min_periods=period).mean()


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Supertrend indicator.
    Returns DataFrame with SUPERT, SUPERTd, SUPERTl, SUPERTs columns.
    """
    hl2 = (high + low) / 2.0
    atr_val = atr(high, low, close, period)
    upperband = hl2 + multiplier * atr_val
    lowerband = hl2 - multiplier * atr_val
    direction = pd.Series(1, index=close.index)
    trend = pd.Series(np.nan, index=close.index)

    for i in range(1, len(close)):
        if close.iloc[i] > upperband.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] > 0:
                lowerband.iloc[i] = max(lowerband.iloc[i], lowerband.iloc[i - 1])
            else:
                upperband.iloc[i] = min(upperband.iloc[i], upperband.iloc[i - 1])

        trend.iloc[i] = lowerband.iloc[i] if direction.iloc[i] > 0 else upperband.iloc[i]

    return pd.DataFrame(
        {
            f"SUPERT_{period}_{float(multiplier)}": trend,
            f"SUPERTd_{period}_{float(multiplier)}": direction,
            f"SUPERTl_{period}_{float(multiplier)}": lowerband,
            f"SUPERTs_{period}_{float(multiplier)}": upperband,
        }
    )


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 20, multiplier: float = 1.5) -> pd.DataFrame:
    """Keltner Channels."""
    mid = ema(close, period)
    range_atr = atr(high, low, close, period)
    upper = mid + multiplier * range_atr
    lower = mid - multiplier * range_atr
    return pd.DataFrame(
        {
            f"KCL_{period}_{float(multiplier)}": lower,
            f"KCB_{period}_{float(multiplier)}": mid,
            f"KCU_{period}_{float(multiplier)}": upper,
        }
    )


def ema_slope(series: pd.Series, period: int = 5) -> pd.Series:
    """Calculate slope of a series over N periods (positive = up)."""
    return series.diff(period) / period


def compute_all_indicators(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Compute all indicators on a DataFrame with OHLCV columns.
    Expects columns: open, high, low, close, volume
    Adds indicator columns in-place and returns the DataFrame.
    """
    cfg = config or {}
    ema_fast_p = cfg.get("ema_fast", 20)
    ema_slow_p = cfg.get("ema_slow", 50)
    atr_p = cfg.get("atr_period", 14)
    adx_p = cfg.get("adx_period", 14)
    rsi_p = cfg.get("rsi_period", 14)
    bb_p = cfg.get("bb_period", 20)
    bb_s = cfg.get("bb_std", 2.0)

    # EMAs
    df["ema_fast"] = ema(df["close"], ema_fast_p)
    df["ema_slow"] = ema(df["close"], ema_slow_p)
    df["ema_9"] = ema(df["close"], 9)

    # ATR
    df["atr"] = atr(df["high"], df["low"], df["close"], atr_p)

    # ADX
    adx_df = adx(df["high"], df["low"], df["close"], adx_p)
    if adx_df is not None and not adx_df.empty:
        for col in adx_df.columns:
            df[col] = adx_df[col]

    # RSI
    df["rsi"] = rsi(df["close"], rsi_p)

    # Stochastic RSI
    stoch = stochastic_rsi(df["close"], rsi_p)
    if stoch is not None and not stoch.empty:
        for col in stoch.columns:
            df[col] = stoch[col]

    # Bollinger Bands
    bb = bollinger_bands(df["close"], bb_p, bb_s)
    if bb is not None and not bb.empty:
        for col in bb.columns:
            df[col] = bb[col]

    # BB Width
    df["bb_width"] = bollinger_width(df["close"], bb_p, bb_s)

    # MACD
    macd_df = macd(df["close"])
    if macd_df is not None and not macd_df.empty:
        for col in macd_df.columns:
            df[col] = macd_df[col]

    # Volume SMA
    df["volume_sma"] = volume_sma(df["volume"], 20)

    # EMA slope
    df["ema_fast_slope"] = ema_slope(df["ema_fast"], 5)
    df["ema_slow_slope"] = ema_slope(df["ema_slow"], 5)

    return df
