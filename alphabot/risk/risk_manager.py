"""
AlphaBot Risk Manager — Enforces ALL hard risk rules.
Every signal MUST pass through this before order placement.
Rules are code, not suggestions — nothing bypasses this layer.

Rules enforced:
  - Daily loss cap (% of balance)
  - Max drawdown from peak (emergency halt)
  - Max concurrent positions
  - Max exposure per trade (% of balance)
  - Max total exposure
  - Max leverage
  - Minimum Risk:Reward ratio
  - Minimum signal confidence
  - Max consecutive losses (cooldown)
  - Correlation block (same direction, same pair)
  - High volatility throttle
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from loguru import logger

from alphabot.config import settings
from alphabot.database.db import Database
from alphabot.database.models import SignalLog
from alphabot.risk.position_sizer import PositionSizer
from alphabot.strategies.signal import Signal, SignalDirection


class RiskManager:
    """
    Validates every signal against risk rules.
    Returns (approved: bool, reason: str, position_size: dict).
    """

    def __init__(self, database: Database):
        self.db = database
        self.sizer = PositionSizer()

        # Runtime state
        self._peak_balance: Decimal = Decimal("0")
        self._session_start_balance: Decimal = Decimal("0")
        self._daily_pnl: Decimal = Decimal("0")
        self._daily_loss_halt: bool = False
        self._drawdown_halt: bool = False
        self._consecutive_losses: int = 0
        self._cooldown_until: Optional[datetime.datetime] = None
        self._consecutive_losses_by_strategy: Dict[str, int] = {}
        self._cooldown_until_by_strategy: Dict[str, datetime.datetime] = {}
        self._last_trade_time: Dict[str, datetime.datetime] = {}  # When trade CLOSED
        self._last_trade_open_time: Dict[str, datetime.datetime] = {}  # When position OPENED
        self._halted: bool = False

    def initialize(
        self,
        account_balance: Decimal,
        db: Optional[Database] = None,
        reset_runtime: bool = True,
    ) -> None:
        """Set balance/peak and restore persisted peak when available."""
        self._session_start_balance = account_balance
        state_db = db or self.db
        if state_db:
            saved_peak = state_db.get_state("peak_balance")
            if saved_peak:
                try:
                    saved_peak_dec = Decimal(saved_peak)
                    self._peak_balance = max(saved_peak_dec, account_balance)
                except Exception:
                    self._peak_balance = account_balance
            else:
                self._peak_balance = account_balance
        else:
            self._peak_balance = account_balance

        if state_db:
            state_db.save_state("peak_balance", str(self._peak_balance))

        if reset_runtime:
            self._daily_pnl = Decimal("0")
            self._daily_loss_halt = False
            self._drawdown_halt = False
        logger.info(
            f"[Risk] Initialized: balance={account_balance} peak={self._peak_balance}"
        )

    @property
    def is_halted(self) -> bool:
        return self._halted or self._drawdown_halt

    @property
    def is_daily_halted(self) -> bool:
        return self._daily_loss_halt

    def validate_signal(
        self,
        signal: Signal,
        account_balance: Decimal,
        open_positions: list,
        existing_exposure: Decimal = Decimal("0"),
    ) -> Tuple[bool, str, dict]:
        """
        Validate a trading signal against all risk rules.

        Returns:
            (approved, rejection_reason, position_size_info)
        """
        # Update peak balance
        if account_balance > self._peak_balance:
            self._peak_balance = account_balance

        # ---- Emergency halt check ----
        if self._drawdown_halt:
            reason = "DRAWDOWN HALT — bot is halted, manual restart required"
            self._log_rejection(signal, reason)
            return False, reason, {}

        # ---- Daily loss cap ----
        if self._daily_loss_halt:
            reason = f"DAILY LOSS CAP — trading halted for the day (loss: {self._daily_pnl})"
            self._log_rejection(signal, reason)
            return False, reason, {}

        # ---- Max drawdown from peak ----
        drawdown_pct = self._current_drawdown(account_balance)
        if drawdown_pct >= float(settings.max_drawdown_pct):
            self._drawdown_halt = True
            self._halted = True
            reason = (
                f"MAX DRAWDOWN BREACHED: {drawdown_pct:.2f}% >= {settings.max_drawdown_pct}% — "
                f"EMERGENCY HALT"
            )
            logger.critical(reason)
            self._log_rejection(signal, reason)
            return False, reason, {}

        # ---- Cooldown from consecutive losses ----
        cooldown_key = signal.strategy_name or "unknown"
        cooldown_until = self._cooldown_until_by_strategy.get(cooldown_key)
        if cooldown_until and datetime.datetime.now(datetime.UTC) < cooldown_until:
            remaining = (cooldown_until - datetime.datetime.now(datetime.UTC)).seconds // 60
            losses = self._consecutive_losses_by_strategy.get(cooldown_key, 0)
            reason = (
                f"COOLDOWN — {remaining} min remaining after {losses} "
                f"consecutive losses ({cooldown_key})"
            )
            self._log_rejection(signal, reason)
            return False, reason, {}

        # ---- Max concurrent positions ----
        open_count = len([p for p in open_positions if p.get("status") in ("OPEN", "PARTIAL")])
        if open_count >= settings.max_concurrent_positions:
            reason = f"MAX POSITIONS — {open_count}/{settings.max_concurrent_positions} positions open"
            self._log_rejection(signal, reason)
            return False, reason, {}

        # ---- Symbol block: one active position per pair ----
        for pos in open_positions:
            if pos.get("symbol") == signal.symbol and pos.get("status") in ("OPEN", "PARTIAL"):
                if pos.get("direction") == signal.direction.value:
                    reason = f"CORRELATION BLOCK — already {signal.direction.value} on {signal.symbol}"
                else:
                    reason = f"SYMBOL BLOCK — opposite position already open on {signal.symbol}"
                self._log_rejection(signal, reason)
                return False, reason, {}

        # ---- Regime-direction alignment ----
        if not self._is_regime_aligned(signal.regime, signal.direction):
            reason = (
                f"REGIME ALIGNMENT BLOCK — {signal.direction.value} conflicts with {signal.regime}"
            )
            self._log_rejection(signal, reason)
            return False, reason, {}

        # ---- Minimum signal confidence ----
        min_confidence = self._min_confidence_for_signal(signal)
        if signal.confidence < min_confidence:
            reason = f"LOW CONFIDENCE — {signal.confidence:.1f} < {min_confidence}"
            self._log_rejection(signal, reason)
            return False, reason, {}

        # ---- Minimum Risk:Reward ratio ----
        rr = signal.risk_reward_ratio
        if rr < float(settings.min_risk_reward):
            reason = f"LOW R:R — {rr:.2f} < {settings.min_risk_reward}"
            self._log_rejection(signal, reason)
            return False, reason, {}

        net_rr = self._net_rr_after_fees(signal)
        min_net_rr = float(getattr(settings, "min_net_risk_reward", settings.min_risk_reward))
        if net_rr < min_net_rr:
            reason = f"LOW NET R:R — {net_rr:.2f} after fees < {min_net_rr}"
            self._log_rejection(signal, reason)
            return False, reason, {}

        min_stop_pct = float(getattr(settings, "min_stop_distance_pct", 0.0) or 0.0)
        if min_stop_pct > 0 and signal.sl_distance_pct < min_stop_pct:
            reason = (
                f"TIGHT STOP — {signal.sl_distance_pct:.2f}% < {min_stop_pct}%"
            )
            self._log_rejection(signal, reason)
            return False, reason, {}

        # ---- Calculate position size (validates per-trade risk too) ----
        size_info = self.sizer.calculate_position_size(
            account_balance=account_balance,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            leverage=settings.max_leverage,
            regime=signal.regime,
            existing_exposure=existing_exposure,
        )

        if size_info["quantity"] <= 0:
            reason = size_info.get("rejection_reason") or "POSITION TOO SMALL — calculated quantity is zero"
            self._log_rejection(signal, reason)
            return False, reason, {}

        # ---- Min candle gap between trades (check from last OPEN, not close) ----
        last_open = self._last_trade_open_time.get(signal.symbol)
        if last_open:
            # Time gap between NOW and the time the LAST position was OPENED
            elapsed = (datetime.datetime.now(datetime.UTC) - last_open).total_seconds()
            tf_seconds = self._timeframe_to_seconds(signal.timeframe)
            min_gap = tf_seconds * settings.min_candles_between_trades
            if elapsed < min_gap:
                reason = f"COOLDOWN — {elapsed:.0f}s since last position opened, need {min_gap}s"
                self._log_rejection(signal, reason)
                return False, reason, {}

        # ---- ALL CHECKS PASSED ----
        logger.info(
            f"[Risk] APPROVED: {signal.symbol} {signal.direction.value} "
            f"conf={signal.confidence:.1f} R:R={rr:.2f} net_R:R={net_rr:.2f} qty={size_info['quantity']}"
        )

        # Log approved signal
        self._log_signal(signal, approved=True)

        return True, "APPROVED", size_info

    def record_trade_opened(
        self,
        symbol: str,
    ) -> None:
        """Called when a position is OPENED — updates open-time cooldown."""
        self._last_trade_open_time[symbol] = datetime.datetime.now(datetime.UTC)
        logger.debug(f"[Risk] Position opened for {symbol} — cooldown reset")

    def record_trade_result(
        self,
        symbol: str,
        pnl: Decimal,
        is_win: bool,
        strategy_name: str = "",
        db: Optional[Database] = None,
    ) -> None:
        """Called after each trade closes to update risk state."""
        self._daily_pnl += pnl
        self._last_trade_time[symbol] = datetime.datetime.now(datetime.UTC)
        strategy_key = strategy_name or "unknown"

        # Consecutive losses
        if not is_win:
            self._consecutive_losses += 1
            losses = self._consecutive_losses_by_strategy.get(strategy_key, 0) + 1
            self._consecutive_losses_by_strategy[strategy_key] = losses
            if losses >= settings.max_consecutive_losses:
                cooldown_until = (
                    datetime.datetime.now(datetime.UTC)
                    + datetime.timedelta(minutes=settings.consecutive_loss_cooldown_minutes)
                )
                self._cooldown_until_by_strategy[strategy_key] = cooldown_until
                self._cooldown_until = max(self._cooldown_until or cooldown_until, cooldown_until)
                logger.warning(
                    f"[Risk] {losses} consecutive losses for {strategy_key} — "
                    f"cooldown until {cooldown_until}"
                )
        else:
            self._consecutive_losses = 0
            self._consecutive_losses_by_strategy[strategy_key] = 0
            self._cooldown_until_by_strategy.pop(strategy_key, None)
            if self._cooldown_until_by_strategy:
                self._cooldown_until = max(self._cooldown_until_by_strategy.values())
            else:
                self._cooldown_until = None

        # Check daily loss cap
        daily_loss_pct = abs(self._daily_pnl / self._session_start_balance * 100) if self._session_start_balance > 0 else Decimal("0")
        if self._daily_pnl < 0 and daily_loss_pct >= settings.daily_loss_cap_pct:
            self._daily_loss_halt = True
            logger.warning(
                f"[Risk] DAILY LOSS CAP HIT — loss:{self._daily_pnl} ({daily_loss_pct:.2f}%)"
            )

        state_db = db or self.db
        if state_db:
            state_db.save_state("peak_balance", str(self._peak_balance))

    def reset_daily(self) -> None:
        """Reset daily counters — called at UTC midnight."""
        self._daily_pnl = Decimal("0")
        self._daily_loss_halt = False
        logger.info("[Risk] Daily limits reset")

    def manual_resume(self) -> None:
        """Manual restart after drawdown halt."""
        self._drawdown_halt = False
        self._halted = False
        logger.info("[Risk] Manual resume — drawdown halt cleared")

    def _current_drawdown(self, balance: Decimal) -> float:
        """Calculate current drawdown from peak as percentage."""
        if self._peak_balance == 0:
            return 0.0
        return float((self._peak_balance - balance) / self._peak_balance * 100)

    def _log_rejection(self, signal: Signal, reason: str) -> None:
        logger.warning(
            f"[Risk] REJECTED: {signal.symbol} {signal.direction.value} — {reason}"
        )
        self._log_signal(signal, approved=False, reason=reason)

    def _log_signal(self, signal: Signal, approved: bool, reason: str = "") -> None:
        try:
            log_entry = SignalLog(
                symbol=signal.symbol,
                direction=signal.direction.value,
                confidence=signal.confidence,
                strategy_name=signal.strategy_name,
                regime=signal.regime,
                entry_price=float(signal.entry_price),
                sl_price=float(signal.stop_loss),
                tp_price=float(signal.take_profit_1),
                approved=1 if approved else 0,
                rejection_reason=reason if not approved else None,
            )
            self.db.log_signal(log_entry)
        except Exception as e:
            logger.error(f"[Risk] Failed to log signal: {e}")

    @staticmethod
    def _timeframe_to_seconds(tf: str) -> int:
        """Convert timeframe string to seconds."""
        multipliers = {"m": 60, "h": 3600, "d": 86400}
        unit = tf[-1].lower()
        value = int(tf[:-1])
        return value * multipliers.get(unit, 60)

    @staticmethod
    def _net_rr_after_fees(signal: Signal) -> float:
        entry = float(signal.entry_price)
        sl_dist = abs(float(signal.entry_price) - float(signal.stop_loss))
        tp_dist = abs(float(signal.take_profit_1) - float(signal.entry_price))
        fee_per_unit = entry * float(settings.estimated_roundtrip_fee_rate)

        effective_reward = tp_dist - fee_per_unit
        effective_risk = sl_dist + fee_per_unit

        if effective_reward <= 0 or effective_risk <= 0:
            return 0.0
        return effective_reward / effective_risk

    @staticmethod
    def _min_confidence_for_signal(signal: Signal) -> float:
        if signal.strategy_name == "liquidity_sweep_orderflow":
            return float(settings.liquidity_sweep_orderflow_min_confidence) * 100.0
        return float(settings.min_signal_confidence)

    @staticmethod
    def _is_regime_aligned(regime: str, direction: SignalDirection) -> bool:
        if regime == "TRENDING_UP":
            return direction == SignalDirection.LONG
        if regime == "TRENDING_DOWN":
            return direction == SignalDirection.SHORT
        return True

    def get_status(self) -> dict:
        """Return current risk state for dashboard."""
        return {
            "daily_pnl": float(self._daily_pnl),
            "peak_balance": float(self._peak_balance),
            "consecutive_losses": self._consecutive_losses,
            "daily_halt": self._daily_loss_halt,
            "drawdown_halt": self._drawdown_halt,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "halted": self._halted,
        }
