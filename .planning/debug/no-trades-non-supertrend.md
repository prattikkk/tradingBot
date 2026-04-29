---
status: verifying
trigger: "Investigate issue: no-trades-non-supertrend"
created: 2026-04-28T15:18:01.2035390+05:30
updated: 2026-04-29T15:12:00.0000000+05:30
---

## Current Focus
<!-- OVERWRITE on each update - reflects NOW -->

hypothesis: The deployed risk fix is live on EC2; current no-trade behavior after restart is caused by the 09:30 UTC candle producing no candidate signals across BTCUSDT, ETHUSDT, and SOLUSDT, not by the old global cooldown or tight-stop bug

test: Verify the rebuilt container contains strategy-scoped cooldown + min_stop_distance_pct, confirm candle-close evaluation still fires on EC2, and inspect live strategy dry-run output for current candidates

expecting: Post-deploy signals will log TIGHT STOP when stops are too small, and any future supertrend candidate will no longer inherit another strategy's cooldown

next_action: Keep watching the next live candidate signals; if a supertrend candidate appears, verify it is no longer rejected by another strategy's loss cooldown and that tight-stop setups log TIGHT STOP instead of opening

## Symptoms
<!-- Written during gathering, then IMMUTABLE -->

expected: supertrend, order flow, liquidity sweep should trade

actual: no trades since last two days; only supertrend trades before then

errors: none reported

reproduction: observed on EC2 logs/dashboard/DB

started: started after adding liquidity sweep + order flow strategies

## Eliminated
<!-- APPEND only - prevents re-investigating -->

- hypothesis: StrategyEngine registry/name mismatch prevents non-supertrend strategy execution
	evidence: engine maps regime to classes, uses class name attribute for lookup, and registry contains ema_adx_volume + orderflow_liquidity_sweep
	timestamp: 2026-04-28T15:36:20.0000000+05:30

## Evidence
<!-- APPEND only - facts discovered -->

- timestamp: 2026-04-28T15:18:36.2982975+05:30
	checked: .planning/debug/knowledge-base.md
	found: File not present
	implication: No known-pattern hypothesis available

- timestamp: 2026-04-28T15:21:01.1389572+05:30
	checked: alphabot/strategies/engine.py
	found: REGIME_STRATEGY_MAP routes TRENDING_* to supertrend strategies only, and RANGING/HIGH_VOLATILITY to LiquiditySweepOrderFlowStrategy only
	implication: orderflow_liquidity_sweep and ema_adx_volume strategies are not selected by regime routing

- timestamp: 2026-04-28T15:21:01.1389572+05:30
	checked: alphabot/strategies/orderflow_liquidity_sweep.py and liquidity_sweep_orderflow.py
	found: Both strategies exist with distinct names and logic, but only liquidity_sweep_orderflow appears in routing
	implication: orderflow_liquidity_sweep will never run unless routing is updated

- timestamp: 2026-04-28T15:23:20.0692473+05:30
	checked: workspace grep for orderflow_liquidity_sweep and ema_adx_volume
	found: EmaAdxVolumeStrategy only appears in its module/tests; OrderFlowLiquiditySweepStrategy appears in engine _strategies but not in REGIME_STRATEGY_MAP
	implication: Strategy engine never selects these strategies for evaluation

- timestamp: 2026-04-28T15:36:20.0000000+05:30
	checked: alphabot/strategies/engine.py
	found: StrategyEngine uses class.name for lookup and registry includes ema_adx_volume and orderflow_liquidity_sweep; REGIME_STRATEGY_MAP uses strategy classes
	implication: Strategy selection path should instantiate and run non-supertrend strategies when regime matches

- timestamp: 2026-04-28T15:36:20.0000000+05:30
	checked: alphabot/risk/risk_manager.py and non-supertrend strategy modules
	found: RiskManager enforces min confidence and regime alignment; ema_adx_volume and orderflow_liquidity_sweep strategies also enforce settings.min_signal_confidence internally
	implication: Signals can be filtered out before execution if confidence or regime alignment fails

- timestamp: 2026-04-28T15:38:05.0000000+05:30
	checked: local alphabot_data.db signal log query
	found: python not available in environment (command failed: Python was not found)
	implication: Need alternate method (sqlite3 CLI or user-run query) to inspect signals_log

- timestamp: 2026-04-28T15:39:10.0000000+05:30
	checked: command availability
	found: Python launcher (py.exe) is installed; sqlite3 CLI is not
	implication: Use py to query local alphabot_data.db for signals_log evidence

- timestamp: 2026-04-28T15:41:10.0000000+05:30
	checked: local alphabot_data.db signals_log output via py
	found: 16 total signal logs, all dated 2026-03-24 and strategy_name=ema_crossover; no recent entries for current strategies
	implication: Local DB is stale/not representative; need EC2 logs or DB query for current behavior

- timestamp: 2026-04-28T16:02:30.0000000+05:30
	checked: EC2 alphabot_data.db signals_log (last 2 days)
	found: 8 signals, all rejected with "LOW NET R:R — <value> after fees < 1.5"; includes liquidity_sweep_orderflow at net R:R 1.16
	implication: Net R:R gate is blocking all trades, including non-supertrend signals

- timestamp: 2026-04-28T16:26:00.0000000+05:30
	checked: EC2 deploy + signals_log after deploy
	found: Container rebuilt successfully; latest signals_log entries are still pre-deploy with LOW NET R:R < 1.5
	implication: Need to wait for new signals to verify approvals with min_net_risk_reward

- timestamp: 2026-04-28T16:40:00.0000000+05:30
	checked: EC2 signals_log re-check
	found: No new entries; still latest signals rejected with LOW NET R:R < 1.5
	implication: Wait for fresh signals before validating the new min_net_risk_reward

- timestamp: 2026-04-28T19:08:00.0000000+05:30
	checked: EC2 alphabot container logs and in-container websocket probes
	found: Container is healthy and connects to Binance websocket, but receives no live kline/markPrice frames; even a direct `btcusdt@markPrice` probe times out after connect
	implication: Signal generation is stalled upstream of strategies; the market-data feed is idle on this host

- timestamp: 2026-04-28T19:25:00.0000000+05:30
	checked: Local code path from websocket client to candle-close callback
	found: Strategy evaluation only runs from `on_candle_close`; no websocket frames means no `Entry eval` and no signals_log entries
	implication: Need fallback candle source independent of websocket delivery

- timestamp: 2026-04-28T19:27:00.0000000+05:30
	checked: Local tests for market-data fallback and risk logic
	found: `pytest tests/test_market_data.py tests/test_risk.py` passes (13 passed)
	implication: REST fallback patch is locally validated

- timestamp: 2026-04-28T19:28:00.0000000+05:30
	checked: EC2 startup logs after redeploy
	found: Historical REST load now excludes still-open bars (199 candles per timeframe on startup) and `[MarketData] REST fallback enabled — polling every 30s` is logged
	implication: The bot is armed to resume candle-close evaluations on the next real closed candle even if websocket stays silent

- timestamp: 2026-04-28T19:46:00.0000000+05:30
	checked: EC2 signals_log and container logs after the next 15m close
	found: REST fallback synced closed 15m candles at 14:00 and 14:15 UTC, triggered `Entry eval` for all pairs, and produced fresh approved signals including `ema_adx_volume` on SOLUSDT at 14:15 UTC
	implication: Candle-close evaluation is restored on EC2 even while websocket frames remain silent

- timestamp: 2026-04-28T19:46:00.0000000+05:30
	checked: EC2 container logs for execution outcome
	found: SOLUSDT `ema_adx_volume` short was approved with net R:R 1.26 and opened as position `a49d9a5f-7279-463b-a31f-1bd6ce8985c6`
	implication: The restored signal path is not just logging approvals; it is opening real trades again

- timestamp: 2026-04-29T00:00:00.0000000+05:30
	checked: User report (dashboard recent trades)
	found: Recent trades show multiple SL_HIT with 0m/1m durations and one TIME_STOP; trades come from ema_adx_volume and liquidity_sweep_orderflow, not supertrend strategies
	implication: Need to audit stop placement/trigger logic and why supertrend strategies are not being selected

- timestamp: 2026-04-29T00:05:00.0000000+05:30
	checked: alphabot/strategies/engine.py
	found: StrategyEngine evaluates only regime-mapped strategies and returns the single highest-confidence signal; it only prefers supertrend_trail/pullback over supertrend_rsi, not over ema_adx_volume
	implication: Even if supertrend signals exist, ema_adx_volume can win selection in TRENDING regimes

- timestamp: 2026-04-29T00:05:00.0000000+05:30
	checked: alphabot/risk/risk_manager.py
	found: Per-strategy min confidence is only applied to liquidity_sweep_orderflow; all other strategies (including supertrend) use global min_signal_confidence=68
	implication: Supertrend signals face a higher confidence gate than liquidity_sweep_orderflow

- timestamp: 2026-04-29T00:05:00.0000000+05:30
	checked: alphabot/positions/position_manager.py and alphabot/execution/order_executor.py
	found: Stop-market orders use MARK_PRICE; local stop checks use live mark/ticker price when available; time stop triggers after settings.time_stop_hours if progress < time_stop_progress_pct
	implication: If SL is tight or mark price is already beyond SL, trades can close almost immediately

- timestamp: 2026-04-29T00:05:00.0000000+05:30
	checked: strategy stop calculations + position sizing
	found: EMA/supertrend strategies derive SL from ATR multiples without enforcing a minimum stop distance; PositionSizer does not reject tight stops beyond zero distance
	implication: Small ATR values can create very tight stops and larger sized positions, increasing fast SL_HIT risk

- timestamp: 2026-04-29T00:10:00.0000000+05:30
	checked: EC2 alphabot_data.db trades and positions (latest 2 days)
	found: Recent SL_HIT trades show sl_distance_pct around 0.34% to 0.60% with 0.0m-0.5m durations; SL prices are on the correct side of entry; TIME_STOP fired at 240m
	implication: Stops are very tight (but not wrong-side); time stop behavior matches config (4h)

- timestamp: 2026-04-29T00:10:00.0000000+05:30
	checked: EC2 signals_log (latest 40)
	found: Supertrend_rsi signals still occur; most recent (BTCUSDT 06:45) was rejected due to COOLDOWN after 6 consecutive losses, while ema_adx_volume/liquidity_sweep_orderflow were approved
	implication: Global consecutive-loss cooldown is blocking supertrend signals even when conditions are met

- timestamp: 2026-04-29T00:25:00.0000000+05:30
	checked: Local pytest tests/test_risk.py
	found: Pytest failed to launch (Python executable not found)
	implication: Local verification is blocked; need user to run tests or configure Python path

- timestamp: 2026-04-29T00:40:00.0000000+05:30
	checked: Local pytest via py launcher
	found: py -m pytest tests/test_risk.py passed (13 tests)
	implication: Local risk logic changes validate; need EC2 verification

- timestamp: 2026-04-29T14:46:00.0000000+05:30
	checked: EC2 rebuild + in-container source/config inspection
	found: alphabot container rebuilt healthy with live code showing strategy-keyed cooldown (`cooldown_key = signal.strategy_name or "unknown"`) and `TIGHT STOP` rejection path; config includes `min_stop_distance_pct: 0.5`
	implication: The risk fix set is active in the running EC2 container

- timestamp: 2026-04-29T15:00:00.0000000+05:30
	checked: EC2 logs since restart
	found: REST fallback synced closed 15m candles for BTCUSDT, ETHUSDT, and SOLUSDT at 09:30 UTC and triggered `--- Entry eval: <symbol> 15m ---` for all three pairs
	implication: Candle-close evaluation remains healthy after the rebuild; lack of new trades is not caused by a stalled data feed

- timestamp: 2026-04-29T15:09:00.0000000+05:30
	checked: EC2 `signals_log` and `trades` after the 09:30 UTC cycle
	found: No `signals_log` rows or trades were written after restart, even though entry evaluation ran for all three symbols
	implication: The post-deploy cycle produced no candidate signals, so risk validation was not reached on that candle

- timestamp: 2026-04-29T15:09:30.0000000+05:30
	checked: In-container dry-run evaluation using fresh Binance REST candles
	found: BTCUSDT and ETHUSDT were `TRENDING_UP`, SOLUSDT was `TRENDING_DOWN`, and all four trend strategies (`supertrend_trail`, `supertrend_pullback`, `supertrend_rsi`, `ema_adx_volume`) returned `NO_SIGNAL` on each symbol
	implication: Current no-trade behavior is market-state/strategy-condition driven, not evidence that supertrend is still blocked by the previous global cooldown bug

## Resolution
<!-- OVERWRITE as understanding evolves -->

root_cause: Recent SL_HIT streaks were caused by tight ATR-based stops (0.34% to 0.60% of entry), and supertrend signals were being blocked by a global consecutive-loss cooldown triggered by other strategies' losses. After deployment, the remaining no-trade behavior on the latest cycle is because no trend strategy emitted a candidate signal on current market data.

fix: Add a minimum stop-distance guard (min_stop_distance_pct) to block ultra-tight SLs, and scope consecutive-loss cooldown per strategy so supertrend is not blocked by other strategies' losing streaks. Deploy those changes to EC2 and confirm candle-close evaluation still runs.

verification: Local `py -m pytest tests/test_risk.py` passed. EC2 container rebuilt healthy, live code/config confirmed, and 09:30 UTC entry evaluation ran for all three pairs. No post-deploy signals were produced because all current trend strategies returned `NO_SIGNAL` on fresh live data.

files_changed: ["alphabot/risk/risk_manager.py", "alphabot/positions/position_manager.py", "alphabot/config.py", "config.yaml", "tests/test_risk.py"]
