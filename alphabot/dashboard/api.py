"""
AlphaBot FastAPI Dashboard — Lightweight web API for monitoring.
Exposes bot state as JSON at http://localhost:8080.
Also serves a simple HTML dashboard.
"""

from __future__ import annotations

import asyncio
import math
from typing import Optional

from loguru import logger

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app(get_state_fn) -> "FastAPI":
    """Create the FastAPI app with routes."""
    app = FastAPI(title="AlphaBot Dashboard", version="1.0")

    def _safe_json(data):
        # Starlette's JSONResponse disallows NaN/Infinity by default.
        # Convert non-finite floats to None so the dashboard doesn't crash.
        return _sanitize_for_json(data)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        state = get_state_fn()
        return _render_html(state)

    @app.get("/api/status")
    async def api_status():
        return JSONResponse(_safe_json(get_state_fn()))

    @app.get("/api/positions")
    async def api_positions():
        state = get_state_fn()
        return JSONResponse(_safe_json(state.get("open_positions", [])))

    @app.get("/api/trades")
    async def api_trades():
        state = get_state_fn()
        return JSONResponse(_safe_json(state.get("recent_trades", [])))

    @app.get("/api/stats")
    async def api_stats():
        state = get_state_fn()
        return JSONResponse(_safe_json(state.get("stats", {})))

    @app.get("/api/risk")
    async def api_risk():
        state = get_state_fn()
        return JSONResponse(_safe_json(state.get("risk_status", {})))

    return app


def _sanitize_for_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


class DashboardServer:
    """Runs the FastAPI dashboard in a background task."""

    def __init__(self, get_state_fn):
        self._get_state = get_state_fn
        self._server: Optional[uvicorn.Server] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        if not HAS_FASTAPI:
            logger.warning("[Dashboard] FastAPI not installed — web dashboard disabled")
            return

        app = create_app(self._get_state)
        config = uvicorn.Config(
            app, host=host, port=port,
            log_level="warning", access_log=False
        )
        self._server = uvicorn.Server(config)

        async def _safe_serve():
            try:
                await self._server.serve()
            except SystemExit:
                logger.warning(f"[Dashboard] Web server exited (port {port} may be in use)")
            except Exception as e:
                logger.warning(f"[Dashboard] Web server error: {e}")

        self._task = asyncio.create_task(_safe_serve())
        await asyncio.sleep(0.5)
        logger.info(f"[Dashboard] Web dashboard started at http://{host}:{port}")

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
            if self._task:
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        logger.info("[Dashboard] Web dashboard stopped")


def _render_html(state: dict) -> str:
    """Simple HTML dashboard page."""
    positions = state.get("open_positions", [])
    trades = state.get("recent_trades", [])
    stats = state.get("stats", {})
    risk = state.get("risk_status", {})
    bot_status = state.get("bot_status", "UNKNOWN")
    balance = state.get("balance", 0)
    daily_pnl = state.get("daily_pnl", 0)
    total_pnl = state.get("total_pnl", 0)

    pos_rows = ""
    for p in positions:
        pnl = p.get("unrealized_pnl", 0)
        color = "#00ff88" if pnl >= 0 else "#ff4444"
        pos_rows += f"""
        <tr>
            <td>{p.get('symbol','')}</td>
            <td>{p.get('direction','')}</td>
            <td>{p.get('quantity',0):.4f}</td>
            <td>{p.get('entry_price',0):.2f}</td>
            <td>{p.get('current_price',0):.2f}</td>
            <td style="color:{color}">${pnl:.2f}</td>
            <td>{p.get('sl_price',0):.2f}</td>
            <td>{p.get('tp1_price',0):.2f}</td>
            <td>{p.get('strategy','')}</td>
        </tr>"""

    trade_rows = ""
    for t in trades[:20]:
        pnl = t.get("net_pnl", 0)
        color = "#00ff88" if pnl >= 0 else "#ff4444"
        trade_rows += f"""
        <tr>
            <td>{t.get('symbol','')}</td>
            <td>{t.get('direction','')}</td>
            <td style="color:{color}">${pnl:.2f}</td>
            <td>{t.get('exit_reason','')}</td>
            <td>{t.get('duration_minutes',0):.0f}m</td>
            <td>{t.get('strategy_name','')}</td>
        </tr>"""

    status_color = "#00ff88" if bot_status == "ACTIVE" else "#ff4444"
    pnl_color = "#00ff88" if daily_pnl >= 0 else "#ff4444"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AlphaBot Dashboard</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: 'Segoe UI', monospace; background: #0d1117; color: #c9d1d9; margin: 20px; }}
            h1 {{ color: #58a6ff; }}
            h2 {{ color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
            .status {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; }}
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 12px 0; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
            .metric {{ background: #21262d; padding: 12px; border-radius: 6px; text-align: center; }}
            .metric .value {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
            .metric .label {{ font-size: 12px; color: #8b949e; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ background: #21262d; padding: 8px; text-align: left; color: #8b949e; }}
            td {{ padding: 8px; border-bottom: 1px solid #21262d; }}
        </style>
    </head>
    <body>
        <h1>⚡ AlphaBot Dashboard</h1>
        <div class="card">
            <span class="status" style="background:{status_color}33; color:{status_color}">{bot_status}</span>
            <span style="margin-left:20px">Balance: <b>${balance:,.2f}</b></span>
            <span style="margin-left:20px">Daily PnL: <b style="color:{pnl_color}">${daily_pnl:,.2f}</b></span>
            <span style="margin-left:20px">Total PnL: <b>${total_pnl:,.2f}</b></span>
        </div>

        <div class="metrics">
            <div class="metric"><div class="value">{stats.get('win_rate',0):.1f}%</div><div class="label">Win Rate</div></div>
            <div class="metric"><div class="value">{stats.get('profit_factor',0):.2f}</div><div class="label">Profit Factor</div></div>
            <div class="metric"><div class="value">{stats.get('total_trades',0)}</div><div class="label">Total Trades</div></div>
            <div class="metric"><div class="value">{stats.get('sharpe_ratio',0):.2f}</div><div class="label">Sharpe Ratio</div></div>
            <div class="metric"><div class="value">${stats.get('avg_win',0):.2f}</div><div class="label">Avg Win</div></div>
            <div class="metric"><div class="value">${stats.get('avg_loss',0):.2f}</div><div class="label">Avg Loss</div></div>
        </div>

        <h2>📊 Open Positions ({len(positions)})</h2>
        <div class="card">
            <table>
                <tr><th>Symbol</th><th>Direction</th><th>Qty</th><th>Entry</th><th>Current</th><th>PnL</th><th>SL</th><th>TP1</th><th>Strategy</th></tr>
                {pos_rows if pos_rows else '<tr><td colspan="9" style="text-align:center">No open positions</td></tr>'}
            </table>
        </div>

        <h2>📜 Recent Trades</h2>
        <div class="card">
            <table>
                <tr><th>Symbol</th><th>Direction</th><th>PnL</th><th>Reason</th><th>Duration</th><th>Strategy</th></tr>
                {trade_rows if trade_rows else '<tr><td colspan="6" style="text-align:center">No trades yet</td></tr>'}
            </table>
        </div>

        <div class="card" style="font-size:12px; color:#8b949e">
            Risk: Daily Loss: {risk.get('daily_pnl',0):.2f} | Consecutive Losses: {risk.get('consecutive_losses',0)} | Halted: {'YES' if risk.get('halted') else 'No'}
        </div>
    </body>
    </html>
    """
