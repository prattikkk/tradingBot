"""
AlphaBot Position Sizer — Fixed Fractional + Kelly Criterion.
Calculates position size from stop-loss distance.
Enforces max risk per trade and total exposure limits.

Formula: Position Size = (Account Balance × Risk%) ÷ (Entry - Stop Loss)
In HIGH_VOLATILITY: Risk Per Trade is halved automatically.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Optional

from loguru import logger

from alphabot.config import settings


class PositionSizer:
    """
    Calculates position size using fixed-fractional method.
    All calculations use Decimal for precision.
    """

    def calculate_position_size(
        self,
        account_balance: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        leverage: int | None = None,
        regime: str = "",
        existing_exposure: Decimal = Decimal("0"),
    ) -> dict:
        """
        Calculate position size in base units and USDT.

        Returns dict with:
            quantity: base asset quantity
            size_usdt: notional value in USDT
            risk_amount: dollar amount risked
            leverage: effective leverage
            risk_pct: actual risk percentage used
        """
        lev = min(leverage or settings.max_leverage, settings.max_leverage)

        # Determine risk percentage
        risk_pct = settings.risk_per_trade_pct

        # Halve risk in high-volatility regime
        if "HIGH_VOLATILITY" in regime.upper():
            risk_pct = risk_pct / Decimal("2")
            logger.info(f"[Sizer] High volatility — risk reduced to {risk_pct}%")

        # Cap at max
        risk_pct = min(risk_pct, settings.max_risk_per_trade_pct)

        # Risk amount in USDT
        risk_amount = account_balance * risk_pct / Decimal("100")

        # SL distance
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance == 0:
            logger.error("[Sizer] Stop-loss distance is zero — cannot size position")
            return {"quantity": Decimal("0"), "size_usdt": Decimal("0"),
                    "risk_amount": Decimal("0"), "leverage": lev, "risk_pct": risk_pct}

        sl_distance_pct = sl_distance / entry_price

        # Base position size (without leverage)
        position_size_usdt = risk_amount / sl_distance_pct

        # Apply leverage: actual margin used
        margin_required = position_size_usdt / Decimal(str(lev))

        # Scale total exposure limit by account size tier.
        if account_balance < Decimal("5000"):
            exposure_pct = Decimal("15")
        elif account_balance < Decimal("20000"):
            exposure_pct = Decimal("10")
        else:
            exposure_pct = Decimal("6")

        max_total_exposure = account_balance * exposure_pct / Decimal("100")
        if existing_exposure + margin_required > max_total_exposure:
            available = max_total_exposure - existing_exposure
            if available <= 0:
                logger.warning("[Sizer] Max total exposure reached — no new position")
                return {"quantity": Decimal("0"), "size_usdt": Decimal("0"),
                        "risk_amount": Decimal("0"), "leverage": lev, "risk_pct": risk_pct}
            margin_required = available
            position_size_usdt = margin_required * Decimal(str(lev))
            logger.info(f"[Sizer] Position reduced to fit exposure limit: {position_size_usdt}")

        # Check per-trade exposure (max 2% of balance)
        max_trade_exposure = account_balance * settings.max_risk_per_trade_pct / Decimal("100")
        if margin_required > max_trade_exposure:
            margin_required = max_trade_exposure
            position_size_usdt = margin_required * Decimal(str(lev))

        # Calculate quantity in base asset
        quantity = (position_size_usdt / entry_price).quantize(
            Decimal("0.0001"), rounding=ROUND_DOWN
        )

        logger.info(
            f"[Sizer] Position sized: qty={quantity} usdt={position_size_usdt:.2f} "
            f"risk=${risk_amount:.2f} ({risk_pct}%) lev={lev}x margin={margin_required:.2f}"
        )

        return {
            "quantity": quantity,
            "size_usdt": position_size_usdt.quantize(Decimal("0.01"), rounding=ROUND_DOWN),
            "risk_amount": risk_amount.quantize(Decimal("0.01"), rounding=ROUND_DOWN),
            "leverage": lev,
            "risk_pct": risk_pct,
        }
