"""
backtest.py — Offline backtester using mainnet historical data.
Run this BEFORE going live to validate the strategy on your chosen symbols.

Usage:
    python backtest.py --symbol BTCUSDT --tf 1h --days 180 --strategy adx_trend
"""
from __future__ import annotations
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import CONFIG
from core.signal import Direction
from strategies.adx_trend import ADXTrendStrategy
from strategies.ensemble import EnsembleStrategy
from strategies.supertrend_rsi import SuperTrendRSIStrategy
from strategies.ema_adx_volume import EMAAdxVolumeStrategy
from strategies.breakout_momentum import BreakoutMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from utils.logger import get_logger

log = get_logger("Backtest")

STRATEGY_MAP = {
    "adx_trend":         ADXTrendStrategy,
    "ensemble":          EnsembleStrategy,
    "supertrend_rsi":    SuperTrendRSIStrategy,
    "ema_adx_volume":    EMAAdxVolumeStrategy,
    "breakout_momentum": BreakoutMomentumStrategy,
    "mean_reversion":    MeanReversionStrategy,
}

INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15,
    "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}

RESAMPLE_RULES = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1D",
}


def fetch_historical(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch up to `days` days of candles from Binance mainnet in chunks."""
    import requests
    limit = 1000
    ms_per_candle = INTERVAL_MINUTES.get(interval, 15) * 60 * 1000
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - days * 24 * 3600 * 1000

    all_rows = []
    session = requests.Session()
    while start_ts < end_ts:
        try:
            r = session.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={
                    "symbol": symbol, "interval": interval,
                    "startTime": start_ts, "limit": limit,
                },
                timeout=15,
            )
            data = r.json()
            if not data:
                break
            all_rows.extend(data)
            start_ts = data[-1][0] + ms_per_candle
            time.sleep(0.1)
        except Exception as e:
            log.error(f"Fetch error: {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols).drop(columns=["_ignore"])
    for c in ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df["taker_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    return df.drop_duplicates()


class Backtest:
    def __init__(
        self,
        symbol: str,
        interval: str,
        days: int,
        strategy_name: str,
        initial_capital: float = 1000.0,
        risk_per_trade: float = 0.015,
        leverage: int = 5,
        slippage_bps: float | None = None,
        spread_bps: float | None = None,
        fee_bps: float | None = None,
        funding_rate_8h: float | None = None,
        maintenance_margin_rate: float | None = None,
        liquidation_fee_bps: float | None = None,
        conservative_ohlc_path: bool | None = None,
        max_hold_hours: float | None = None,
    ):
        self.symbol          = symbol
        self.interval        = interval
        self.days            = days
        self.strategy        = STRATEGY_MAP.get(strategy_name, ADXTrendStrategy)()
        self.capital         = initial_capital
        self.risk_per_trade  = risk_per_trade
        self.leverage        = leverage
        self.sl_mult         = CONFIG.risk.atr_sl_multiplier
        self.tp1_mult        = CONFIG.risk.atr_tp1_multiplier
        self.tp2_mult        = CONFIG.risk.atr_tp2_multiplier
        self.slippage_bps    = (
            float(slippage_bps)
            if slippage_bps is not None
            else float(CONFIG.trading.backtest_slippage_bps)
        )
        self.spread_bps      = (
            float(spread_bps)
            if spread_bps is not None
            else float(CONFIG.trading.backtest_spread_bps)
        )
        self.taker_fee_bps   = (
            float(fee_bps)
            if fee_bps is not None
            else float(CONFIG.trading.backtest_taker_fee_bps)
        )
        self.funding_rate_8h = max(
            0.0,
            float(funding_rate_8h)
            if funding_rate_8h is not None
            else float(CONFIG.trading.backtest_funding_rate_8h),
        )
        self.maintenance_margin_rate = max(
            0.0,
            float(maintenance_margin_rate)
            if maintenance_margin_rate is not None
            else float(CONFIG.trading.backtest_maintenance_margin_rate),
        )
        self.liquidation_fee_bps = max(
            0.0,
            float(liquidation_fee_bps)
            if liquidation_fee_bps is not None
            else float(CONFIG.trading.backtest_liquidation_fee_bps),
        )
        self.conservative_ohlc_path = (
            bool(conservative_ohlc_path)
            if conservative_ohlc_path is not None
            else bool(CONFIG.trading.backtest_conservative_ohlc_path)
        )
        hold_hours = (
            float(max_hold_hours)
            if max_hold_hours is not None
            else float(CONFIG.trading.backtest_max_hold_hours)
        )
        self.max_hold_bars = self._bars_from_hours(hold_hours)
        self._tie_events = 0

    def run(self) -> dict:
        log.info(f"Fetching {self.days}d of {self.symbol} {self.interval} candles…")
        df = fetch_historical(self.symbol, self.interval, self.days)
        if df.empty or len(df) < 100:
            log.error("Not enough data")
            return {}

        self._tie_events = 0

        log.info(f"Got {len(df)} candles. Running backtest…")

        htf_1_full = self._build_htf_frame(df, CONFIG.strategy.htf_1)
        htf_2_full = self._build_htf_frame(df, CONFIG.strategy.htf_2)

        trades = []
        balance = self.capital
        warmup = 60  # bars needed before signaling

        for i in range(warmup + 1, len(df) - 1):
            # Keep one extra bar in the window so strategy closed-bar indexing (-2)
            # aligns with live behavior where newest candle can still be in-flight.
            window = df.iloc[: i + 1].copy()
            end_time = window.index[-1]
            htf_1_window = self._slice_htf_window(htf_1_full, end_time)
            htf_2_window = self._slice_htf_window(htf_2_full, end_time)

            signal = self.strategy.generate(
                self.symbol,
                window,
                htf_1_window,
                htf_2_window,
            )
            if signal is None or not signal.is_valid:
                continue

            # Position sizing
            entry   = self._apply_entry_cost(signal.entry_price, signal.direction)
            sl      = signal.stop_loss
            risk_amount = balance * self.risk_per_trade
            risk_per_unit = abs(entry - sl)
            if risk_per_unit <= 0:
                continue
            qty     = (risk_amount * self.leverage) / risk_per_unit
            notional = qty * entry

            # Simulate exit using future candles
            outcome = self._simulate_exit(df, i, signal, qty, entry)
            if outcome is None:
                continue

            pnl, exit_price, bars_held, reason, funding_cost = outcome
            balance += pnl

            trades.append({
                "entry_time":  df.index[i],
                "exit_time":   df.index[min(i + bars_held, len(df) - 1)],
                "direction":   signal.direction.value,
                "confidence":  signal.confidence,
                "entry":       entry,
                "entry_signal": signal.entry_price,
                "exit":        exit_price,
                "pnl":         pnl,
                "slippage_bps": self.slippage_bps,
                "spread_bps": self.spread_bps,
                "fee_bps": self.taker_fee_bps,
                "funding_cost": funding_cost,
                "balance":     balance,
                "reason":      reason,
                "rr":          signal.risk_reward,
            })

            if balance <= 0:
                log.warning("Account blown!")
                break

        return self._report(trades, balance)

    def _simulate_exit(self, df, entry_bar, signal, qty, entry_price):
        """Walk forward to find first SL/TP/liquidation hit."""
        direction = signal.direction
        entry = entry_price
        sl = signal.stop_loss
        tp1 = signal.take_profit_1
        tp2 = signal.take_profit_2
        tp1_hit = False
        partial_pnl = 0.0
        initial_qty = qty
        liquidation_price = self._liquidation_price(entry, direction)

        max_exit_bar = min(entry_bar + self.max_hold_bars, len(df) - 1)
        for j in range(entry_bar + 1, max_exit_bar + 1):
            bar = df.iloc[j]
            high, low = bar["high"], bar["low"]
            bars = j - entry_bar

            if self._liquidation_hit(direction, high, low, liquidation_price):
                liq_fill = self._apply_liquidation_fill(liquidation_price, direction)
                return self._finalize_exit(
                    direction=direction,
                    entry=entry,
                    qty_remaining=qty,
                    partial_pnl=partial_pnl,
                    bars_held=bars,
                    reason="LIQUIDATION",
                    initial_qty=initial_qty,
                    fill_override=liq_fill,
                )

            if direction == Direction.LONG:
                sl_hit = low <= sl
                tp1_now = (not tp1_hit) and high >= tp1

                if not tp1_hit:
                    if sl_hit and tp1_now and self.conservative_ohlc_path:
                        self._tie_events += 1
                        return self._finalize_exit(
                            direction=direction,
                            entry=entry,
                            qty_remaining=qty,
                            partial_pnl=partial_pnl,
                            bars_held=bars,
                            reason="SL",
                            initial_qty=initial_qty,
                            target_price=sl,
                            is_stop=True,
                        )
                    if sl_hit:
                        return self._finalize_exit(
                            direction=direction,
                            entry=entry,
                            qty_remaining=qty,
                            partial_pnl=partial_pnl,
                            bars_held=bars,
                            reason="SL",
                            initial_qty=initial_qty,
                            target_price=sl,
                            is_stop=True,
                        )
                    if tp1_now:
                        tp1_fill = self._apply_exit_cost(tp1, direction, is_stop=False)
                        partial_qty_pct = float(CONFIG.risk.partial_exit_pct)
                        partial_pnl = (tp1_fill - entry) * qty * partial_qty_pct
                        qty *= partial_qty_pct
                        tp1_hit = True

                tp2_now = high >= tp2
                sl_hit = low <= sl
                if sl_hit and tp2_now and self.conservative_ohlc_path:
                    self._tie_events += 1
                    return self._finalize_exit(
                        direction=direction,
                        entry=entry,
                        qty_remaining=qty,
                        partial_pnl=partial_pnl,
                        bars_held=bars,
                        reason="SL",
                        initial_qty=initial_qty,
                        target_price=sl,
                        is_stop=True,
                    )
                if sl_hit:
                    return self._finalize_exit(
                        direction=direction,
                        entry=entry,
                        qty_remaining=qty,
                        partial_pnl=partial_pnl,
                        bars_held=bars,
                        reason="SL",
                        initial_qty=initial_qty,
                        target_price=sl,
                        is_stop=True,
                    )
                if tp2_now:
                    return self._finalize_exit(
                        direction=direction,
                        entry=entry,
                        qty_remaining=qty,
                        partial_pnl=partial_pnl,
                        bars_held=bars,
                        reason="TP2",
                        initial_qty=initial_qty,
                        target_price=tp2,
                        is_stop=False,
                    )

            else:
                sl_hit = high >= sl
                tp1_now = (not tp1_hit) and low <= tp1

                if not tp1_hit:
                    if sl_hit and tp1_now and self.conservative_ohlc_path:
                        self._tie_events += 1
                        return self._finalize_exit(
                            direction=direction,
                            entry=entry,
                            qty_remaining=qty,
                            partial_pnl=partial_pnl,
                            bars_held=bars,
                            reason="SL",
                            initial_qty=initial_qty,
                            target_price=sl,
                            is_stop=True,
                        )
                    if sl_hit:
                        return self._finalize_exit(
                            direction=direction,
                            entry=entry,
                            qty_remaining=qty,
                            partial_pnl=partial_pnl,
                            bars_held=bars,
                            reason="SL",
                            initial_qty=initial_qty,
                            target_price=sl,
                            is_stop=True,
                        )
                    if tp1_now:
                        tp1_fill = self._apply_exit_cost(tp1, direction, is_stop=False)
                        partial_qty_pct = float(CONFIG.risk.partial_exit_pct)
                        partial_pnl = (entry - tp1_fill) * qty * partial_qty_pct
                        qty *= partial_qty_pct
                        tp1_hit = True

                tp2_now = low <= tp2
                sl_hit = high >= sl
                if sl_hit and tp2_now and self.conservative_ohlc_path:
                    self._tie_events += 1
                    return self._finalize_exit(
                        direction=direction,
                        entry=entry,
                        qty_remaining=qty,
                        partial_pnl=partial_pnl,
                        bars_held=bars,
                        reason="SL",
                        initial_qty=initial_qty,
                        target_price=sl,
                        is_stop=True,
                    )
                if sl_hit:
                    return self._finalize_exit(
                        direction=direction,
                        entry=entry,
                        qty_remaining=qty,
                        partial_pnl=partial_pnl,
                        bars_held=bars,
                        reason="SL",
                        initial_qty=initial_qty,
                        target_price=sl,
                        is_stop=True,
                    )
                if tp2_now:
                    return self._finalize_exit(
                        direction=direction,
                        entry=entry,
                        qty_remaining=qty,
                        partial_pnl=partial_pnl,
                        bars_held=bars,
                        reason="TP2",
                        initial_qty=initial_qty,
                        target_price=tp2,
                        is_stop=False,
                    )

        # Timeout — exit at last close
        last = df.iloc[max_exit_bar]
        return self._finalize_exit(
            direction=direction,
            entry=entry,
            qty_remaining=qty,
            partial_pnl=partial_pnl,
            bars_held=max_exit_bar - entry_bar,
            reason="TIMEOUT",
            initial_qty=initial_qty,
            target_price=float(last["close"]),
            is_stop=False,
        )

    def _report(self, trades: list, final_balance: float) -> dict:
        if not trades:
            print("\n❌ No trades generated.")
            return {}

        tdf = pd.DataFrame(trades)
        wins   = tdf[tdf["pnl"] > 0]
        losses = tdf[tdf["pnl"] <= 0]
        wr     = len(wins) / len(tdf) * 100
        pf     = abs(wins["pnl"].sum() / losses["pnl"].sum()) if not losses.empty else float("inf")
        max_dd = self._max_drawdown(tdf["balance"].tolist())

        report = {
            "symbol":        self.symbol,
            "interval":      self.interval,
            "strategy":      self.strategy.name,
            "days":          self.days,
            "slippage_bps":  self.slippage_bps,
            "spread_bps":    self.spread_bps,
            "fee_bps":       self.taker_fee_bps,
            "funding_rate_8h": self.funding_rate_8h,
            "maintenance_margin_rate": self.maintenance_margin_rate,
            "liquidation_fee_bps": self.liquidation_fee_bps,
            "conservative_ohlc_path": self.conservative_ohlc_path,
            "max_hold_bars": self.max_hold_bars,
            "total_trades":  len(tdf),
            "win_rate":      round(wr, 1),
            "profit_factor": round(pf, 2),
            "total_pnl":     round(tdf["pnl"].sum(), 2),
            "total_funding_cost": round(tdf["funding_cost"].sum(), 2),
            "avg_win":       round(wins["pnl"].mean(), 2) if not wins.empty else 0,
            "avg_loss":      round(losses["pnl"].mean(), 2) if not losses.empty else 0,
            "max_drawdown":  round(max_dd, 2),
            "final_balance": round(final_balance, 2),
            "return_pct":    round((final_balance - self.capital) / self.capital * 100, 1),
            "tp2_rate":      round(len(tdf[tdf["reason"]=="TP2"]) / len(tdf) * 100, 1),
            "sl_rate":       round(len(tdf[tdf["reason"]=="SL"]) / len(tdf) * 100, 1),
            "liquidation_rate": round(len(tdf[tdf["reason"]=="LIQUIDATION"]) / len(tdf) * 100, 1),
            "tie_events": int(self._tie_events),
        }

        # Print
        print("\n" + "=" * 55)
        print(f"  📊 BACKTEST RESULTS — {self.symbol} {self.interval}")
        print("=" * 55)
        for k, v in report.items():
            print(f"  {k:<20}: {v}")
        print("=" * 55)

        # Save CSV
        out = Path(f"data/backtest_{self.symbol}_{self.interval}.csv")
        out.parent.mkdir(exist_ok=True)
        tdf.to_csv(out, index=False)
        print(f"\n  Detailed trades → {out}\n")

        return report

    @staticmethod
    def _max_drawdown(balances: list) -> float:
        peak = balances[0]
        max_dd = 0.0
        for b in balances:
            if b > peak:
                peak = b
            dd = (peak - b) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _bars_from_hours(self, hours: float) -> int:
        if hours <= 0:
            return 200
        minutes = INTERVAL_MINUTES.get(self.interval, 15)
        return max(1, int((hours * 60) / minutes))

    def _build_htf_frame(self, base_df: pd.DataFrame, target_interval: str | None) -> pd.DataFrame | None:
        if not target_interval:
            return None

        target_interval = str(target_interval)
        if target_interval == self.interval:
            return base_df

        base_minutes = INTERVAL_MINUTES.get(self.interval)
        target_minutes = INTERVAL_MINUTES.get(target_interval)
        rule = RESAMPLE_RULES.get(target_interval)
        if not base_minutes or not target_minutes or not rule:
            return None
        if target_minutes < base_minutes or target_minutes % base_minutes != 0:
            return None

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        for col in ("quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
            if col in base_df.columns:
                agg[col] = "sum"

        htf = base_df.resample(rule).agg(agg)
        htf = htf.dropna(subset=["open", "high", "low", "close"])

        if "taker_buy_base" in htf.columns and "volume" in htf.columns:
            htf["taker_ratio"] = np.where(
                htf["volume"] > 0,
                htf["taker_buy_base"] / htf["volume"],
                np.nan,
            )

        return htf

    @staticmethod
    def _slice_htf_window(htf_df: pd.DataFrame | None, end_time: pd.Timestamp) -> pd.DataFrame | None:
        if htf_df is None:
            return None
        idx = htf_df.index.searchsorted(end_time, side="right")
        window = htf_df.iloc[:idx]
        return window if not window.empty else None

    def _fee_rate(self) -> float:
        return self.taker_fee_bps / 10000.0

    def _apply_entry_cost(self, price: float, direction: Direction) -> float:
        half_spread = self.spread_bps / 20000.0
        slip = self.slippage_bps / 10000.0
        fee_rate = self._fee_rate()
        if direction == Direction.LONG:
            return price * (1 + half_spread + slip) * (1 + fee_rate)
        return price * (1 - half_spread - slip) * (1 - fee_rate)

    def _apply_exit_cost(self, price: float, direction: Direction, is_stop: bool) -> float:
        half_spread = self.spread_bps / 20000.0
        slip = self.slippage_bps / 10000.0
        impact = slip * (1.5 if is_stop else 1.0)
        fee_rate = self._fee_rate()
        if direction == Direction.LONG:
            return price * (1 - half_spread - impact) * (1 - fee_rate)
        return price * (1 + half_spread + impact) * (1 + fee_rate)

    def _funding_cost(self, entry_price: float, quantity: float, bars_held: int) -> float:
        if self.funding_rate_8h <= 0 or bars_held <= 0 or quantity <= 0:
            return 0.0
        minutes_per_bar = INTERVAL_MINUTES.get(self.interval, 15)
        hold_hours = (bars_held * minutes_per_bar) / 60.0
        return entry_price * quantity * self.funding_rate_8h * (hold_hours / 8.0)

    def _liquidation_price(self, entry_price: float, direction: Direction) -> float | None:
        if self.leverage <= 0:
            return None

        move_fraction = (1.0 / float(self.leverage)) - self.maintenance_margin_rate
        if move_fraction <= 0:
            return None

        move_fraction = min(move_fraction, 0.99)
        if direction == Direction.LONG:
            return max(0.0, entry_price * (1.0 - move_fraction))
        return entry_price * (1.0 + move_fraction)

    @staticmethod
    def _liquidation_hit(direction: Direction, high: float, low: float, liquidation_price: float | None) -> bool:
        if liquidation_price is None:
            return False
        if direction == Direction.LONG:
            return low <= liquidation_price
        return high >= liquidation_price

    def _apply_liquidation_fill(self, liquidation_price: float, direction: Direction) -> float:
        penalty = self.liquidation_fee_bps / 10000.0
        if direction == Direction.LONG:
            penalized = liquidation_price * (1.0 - penalty)
        else:
            penalized = liquidation_price * (1.0 + penalty)
        return self._apply_exit_cost(penalized, direction, is_stop=True)

    @staticmethod
    def _realized_pnl(direction: Direction, entry: float, exit_fill: float, qty_remaining: float, partial_pnl: float) -> float:
        if direction == Direction.LONG:
            return partial_pnl + (exit_fill - entry) * qty_remaining
        return partial_pnl + (entry - exit_fill) * qty_remaining

    def _finalize_exit(
        self,
        *,
        direction: Direction,
        entry: float,
        qty_remaining: float,
        partial_pnl: float,
        bars_held: int,
        reason: str,
        initial_qty: float,
        target_price: float | None = None,
        is_stop: bool = False,
        fill_override: float | None = None,
    ):
        if fill_override is not None:
            exit_fill = fill_override
        else:
            if target_price is None:
                raise ValueError("target_price is required when fill_override is not provided")
            exit_fill = self._apply_exit_cost(target_price, direction, is_stop=is_stop)

        gross_pnl = self._realized_pnl(
            direction=direction,
            entry=entry,
            exit_fill=exit_fill,
            qty_remaining=qty_remaining,
            partial_pnl=partial_pnl,
        )
        funding_cost = self._funding_cost(entry, initial_qty, bars_held)
        net_pnl = gross_pnl - funding_cost
        return net_pnl, exit_fill, bars_held, reason, funding_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaBot Backtester")
    parser.add_argument("--symbol",   default="BTCUSDT")
    parser.add_argument("--tf",       default=CONFIG.strategy.primary_tf)
    parser.add_argument("--days",     type=int, default=60)
    parser.add_argument("--strategy", default="adx_trend",
                        choices=list(STRATEGY_MAP.keys()))
    parser.add_argument("--capital",  type=float, default=1000.0)
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=CONFIG.trading.backtest_slippage_bps,
        help="One-way slippage in basis points applied to entry/exit fills",
    )
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=CONFIG.trading.backtest_spread_bps,
        help="Bid/ask spread in basis points",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=CONFIG.trading.backtest_taker_fee_bps,
        help="Per-side taker fee in basis points",
    )
    parser.add_argument(
        "--funding-rate-8h",
        type=float,
        default=CONFIG.trading.backtest_funding_rate_8h,
        help="Funding charge rate per 8 hours",
    )
    parser.add_argument(
        "--maintenance-margin-rate",
        type=float,
        default=CONFIG.trading.backtest_maintenance_margin_rate,
        help="Approximate maintenance margin ratio for liquidation modeling",
    )
    parser.add_argument(
        "--liquidation-fee-bps",
        type=float,
        default=CONFIG.trading.backtest_liquidation_fee_bps,
        help="Additional liquidation penalty in basis points",
    )
    parser.add_argument(
        "--conservative-ohlc-path",
        action="store_true",
        default=CONFIG.trading.backtest_conservative_ohlc_path,
        help="If TP and SL are hit in the same candle, prioritize SL",
    )
    parser.add_argument(
        "--non-conservative-ohlc-path",
        action="store_false",
        dest="conservative_ohlc_path",
        help="Disable conservative same-candle conflict handling",
    )
    parser.add_argument(
        "--max-hold-hours",
        type=float,
        default=CONFIG.trading.backtest_max_hold_hours,
        help="Force timeout exit after this many hours",
    )
    args = parser.parse_args()

    bt = Backtest(
        symbol=args.symbol,
        interval=args.tf,
        days=args.days,
        strategy_name=args.strategy,
        initial_capital=args.capital,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        fee_bps=args.fee_bps,
        funding_rate_8h=args.funding_rate_8h,
        maintenance_margin_rate=args.maintenance_margin_rate,
        liquidation_fee_bps=args.liquidation_fee_bps,
        conservative_ohlc_path=args.conservative_ohlc_path,
        max_hold_hours=args.max_hold_hours,
    )
    bt.run()
