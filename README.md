# ⚡ AlphaBot — Adaptive Algorithmic Trading System

An autonomous, adaptive crypto futures trading bot for Binance USD-M Futures. Paper trades on Testnet with live Mainnet price data.

## Features

- **Adaptive Strategy Engine** — Auto-detects market regime and selects the optimal strategy
- **Built-in Strategy**:
  - Supertrend + RSI + EMA200 (trend confirmation)
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
- **Journal**: `data/trade_journal.csv` (CSV trade journal)

## One-time: Backfill missing DB trades

If your SQLite `trades` table is missing historical rows (e.g. DB was reset/overwritten), you can rebuild it from:
- the CSV journal (trade details)
- the structured JSON logs (open/close timestamps)

Dry-run (safe):

```bash
python backfill_trades.py --dry-run
```

Run (creates a timestamped DB backup first):

```bash
python backfill_trades.py
```

Notes:
- By default it reads `data/trade_journal.csv` and/or `trade_journal.csv` (if present) and scans `logs/`.
- It is idempotent and will not overwrite existing trades.
- On EC2, stop the bot/container first to avoid SQLite locks, then run backfill, then restart.

Verify on EC2:

```bash
python ec2_bot_status.py
```

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
    detector.py           — ADX/ATR/EMA slope regime classifier
  strategies/
    base.py               — Abstract strategy interface
    supertrend_rsi.py     — Strategy A: Supertrend + RSI + EMA200
    ema_adx_volume.py     — Strategy B: EMA 9/21 + ADX + Volume
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
    indicators.py         — Indicator helpers
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
| `MIN_NET_RISK_REWARD` | 1.15 | Min net R:R (after fees) to accept a trade |

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
