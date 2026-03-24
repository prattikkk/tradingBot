# ⚡ AlphaBot — Adaptive Algorithmic Trading System

An autonomous, adaptive crypto futures trading bot for Binance USD-M Futures. Paper trades on Testnet with live Mainnet price data.

## Features

- **Adaptive Strategy Engine** — Auto-detects market regime (Trending/Ranging/Volatile) and selects optimal strategy
- **3 Built-in Strategies**:
  - EMA Crossover (trending markets)
  - Bollinger Band Mean Reversion (ranging markets)
  - ATR Breakout (high-volatility breakouts)
- **Risk Management** — Hard-coded limits: daily loss cap, max drawdown, max positions, leverage caps
- **Position Manager** — Full lifecycle with partial exits, trailing stops, breakeven moves
- **Real-time Dashboard** — Terminal UI (Rich) + Web dashboard (FastAPI) at http://localhost:8080
- **Telegram Alerts** — Trade opened/closed, TP hit, risk events, daily summaries
- **Trade Journal** — CSV + SQLite persistence for every trade
- **Crash Recovery** — Open positions persisted to SQLite, recovered on restart

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required for paper trading:**
- `BINANCE_TESTNET_API_KEY` — Get from [testnet.binancefuture.com](https://testnet.binancefuture.com)
- `BINANCE_TESTNET_SECRET`

**Optional:**
- `TELEGRAM_BOT_TOKEN` — From @BotFather
- `TELEGRAM_CHAT_ID` — Your Telegram user ID

### 3. Run the Bot

```bash
python -m alphabot.main
```

The bot will:
1. Fetch 200 historical candles for each configured pair
2. Connect to Binance Mainnet WebSocket for live price data
3. Start the strategy engine, risk manager, and position monitor
4. Open the web dashboard at http://localhost:8080

### 4. Monitor

- **Terminal**: Real-time Rich dashboard in console
- **Web**: http://localhost:8080 (auto-refreshes every 5s)
- **Telegram**: Configure bot token for mobile alerts
- **Logs**: `logs/` directory (structured JSON, rotated at 50MB)
- **Journal**: `trade_journal.csv` in project root

## Project Structure

```
alphabot/
  main.py                 — Entry point, async orchestrator
  config.py               — Pydantic settings (loads .env + config.yaml)
  data/
    websocket_client.py   — Binance Mainnet WebSocket feed
    data_store.py         — In-memory rolling OHLCV buffer
    models.py             — Candle, Ticker, OrderBook models
  regime/
    detector.py           — ADX/ATR/BBW regime classifier
  strategies/
    base.py               — Abstract strategy interface
    ema_crossover.py      — Strategy A: EMA trend following
    bb_reversion.py       — Strategy B: Bollinger mean reversion
    atr_breakout.py       — Strategy C: ATR breakout
    signal.py             — Signal model + confidence scoring
    engine.py             — Regime-to-strategy router
  risk/
    risk_manager.py       — All hard risk rules
    position_sizer.py     — Fixed-fractional position calculator
  positions/
    position_manager.py   — Position lifecycle orchestrator
    pnl_tracker.py        — Trade journal + PnL statistics
  execution/
    order_executor.py     — Order placement with retry logic
    testnet_client.py     — CCXT Binance client wrapper
  dashboard/
    terminal_ui.py        — Rich terminal dashboard
    api.py                — FastAPI web dashboard
  notifications/
    telegram_bot.py       — Telegram alert sender
  database/
    models.py             — SQLAlchemy ORM models
    db.py                 — SQLite CRUD helpers
  utils/
    logger.py             — Loguru structured logging
    indicators.py         — pandas-ta indicator wrappers
    retry.py              — Exponential backoff decorator
  tests/
    test_regime.py        — Regime detector tests
    test_strategies.py    — Strategy signal tests
    test_risk.py          — Risk rule tests
```

## Configuration

### Risk Parameters (`.env`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RISK_PER_TRADE_PCT` | 1.0 | % of balance risked per trade |
| `MAX_RISK_PER_TRADE_PCT` | 2.0 | Hard cap per trade |
| `DAILY_LOSS_CAP_PCT` | 2.0 | Halt trading at this daily loss % |
| `MAX_DRAWDOWN_PCT` | 5.0 | Emergency halt threshold |
| `MAX_CONCURRENT_POSITIONS` | 3 | Max open trades |
| `MAX_LEVERAGE` | 5 | Max leverage per position |
| `MIN_SIGNAL_CONFIDENCE` | 60 | Min confidence score (0-100) |
| `MIN_RISK_REWARD` | 1.5 | Min R:R to accept a trade |

### Strategy Parameters (`config.yaml`)

Modify `config.yaml` for indicator periods, ATR multipliers, and signal scoring weights.

## Mainnet Promotion Criteria

Before switching to Mainnet (`ENVIRONMENT=mainnet`):

- [ ] 200+ paper trades completed
- [ ] Win rate ≥ 55%
- [ ] Profit factor ≥ 1.2
- [ ] Max drawdown ≤ 5%
- [ ] No crashes in 7 consecutive days
- [ ] All risk rules verified
- [ ] Telegram alerts working
- [ ] Trade journal accurate

## Architecture

```
Binance Mainnet (prices) ──WebSocket──▶ Data Store ──▶ Regime Detector
                                                            │
                                                    Strategy Engine
                                                            │
                                                    Signal (with confidence)
                                                            │
                                                    Risk Manager (validate)
                                                            │
                                                    Position Manager
                                                            │
Binance Testnet (orders) ◀──REST──── Order Executor ────────┘
                                                            │
                                              ┌─────────────┼─────────────┐
                                         Dashboard      Telegram      Journal
```

## License

Private — not for redistribution.
