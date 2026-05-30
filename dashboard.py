"""Lightweight local dashboard for portfolio monitoring."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse

_log = logging.getLogger("Dashboard")

from config import CONFIG
from core.control_plane import enqueue_command, get_control_state, set_paused, update_control_state

PORTFOLIO_PATH = Path("data/portfolio.json")


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(_load_dashboard_state())
            return
        if path == "/api/control":
            self._send_json({"ok": True, "control": get_control_state()})
            return
        self._send_html(_render_html())

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/control":
            self._send_json({"ok": False, "error": "Unsupported endpoint"}, status=404)
            return

        body = self._read_json_body()
        action = str(body.get("action", "")).strip().lower()
        payload = body.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        response = _handle_control_action(action, payload)
        self._send_json(response, status=200 if response.get("ok") else 400)

    def log_message(self, format, *args):
        return

    def _send_json(self, payload: dict, status: int = 200):
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, html: str):
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json_body(self) -> dict:
        try:
            raw_len = self.headers.get("Content-Length", "0")
            content_length = int(raw_len)
        except Exception:
            content_length = 0

        if content_length <= 0:
            return {}

        try:
            payload = self.rfile.read(content_length)
            data = json.loads(payload.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def _load_state() -> dict:
    if not PORTFOLIO_PATH.exists():
        return {"balance": 0.0, "open_positions": {}, "closed_trades": []}
    try:
        return json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        _log.warning("Portfolio state load failed: %s", e)
        return {"balance": 0.0, "open_positions": {}, "closed_trades": []}


def _fetch_mark_prices(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}

    wanted = {str(s).upper() for s in symbols if s}
    urls = [
        "https://fapi.binance.com/fapi/v1/ticker/price",
        "https://testnet.binancefuture.com/fapi/v1/ticker/price",
    ]

    for url in urls:
        try:
            with urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            continue

        rows = payload if isinstance(payload, list) else [payload]
        prices: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "")).upper()
            if symbol not in wanted:
                continue

            price_raw = row.get("price")
            if price_raw is None:
                continue

            try:
                prices[symbol] = float(price_raw)
            except Exception:
                continue

        if prices:
            return prices

    return {}


def _enrich_open_positions_with_live_pnl(payload: dict) -> None:
    open_positions = payload.get("open_positions")
    if not isinstance(open_positions, dict) or not open_positions:
        return

    symbols = [str(s).upper() for s in open_positions.keys()]
    mark_prices = _fetch_mark_prices(symbols)

    for symbol, pos in open_positions.items():
        if not isinstance(pos, dict):
            continue

        order_ids_raw = pos.get("order_ids")
        order_ids = order_ids_raw if isinstance(order_ids_raw, dict) else {}
        has_exchange_protection = any(order_ids.get(k) for k in ("sl", "tp1", "tp2", "trail"))
        pos["protection_mode"] = "exchange" if has_exchange_protection else "client"

        mark = mark_prices.get(str(symbol).upper())
        if mark is None:
            continue

        try:
            entry = float(pos.get("entry_price", 0.0))
            qty = float(pos.get("quantity", 0.0))
            realized_partial = float(pos.get("pnl", 0.0))
        except Exception:
            continue

        direction = str(pos.get("direction", "")).upper()
        if direction == "LONG":
            unrealized = (mark - entry) * qty
        else:
            unrealized = (entry - mark) * qty

        pos["mark_price"] = mark
        pos["unrealized_pnl"] = unrealized
        pos["live_pnl"] = realized_partial + unrealized


def _to_float(value, min_value: float, max_value: float) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed < min_value or parsed > max_value:
        return None
    return parsed


def _to_int(value, min_value: int, max_value: int) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < min_value or parsed > max_value:
        return None
    return parsed


def _load_dashboard_state() -> dict:
    payload = _load_state()
    _enrich_open_positions_with_live_pnl(payload)
    payload["control"] = get_control_state()
    return payload


def _handle_control_action(action: str, payload: dict) -> dict:
    if action == "pause":
        state = set_paused(True)
        return {"ok": True, "control": state}

    if action == "resume":
        state = set_paused(False)
        return {"ok": True, "control": state}

    if action == "close_symbol":
        symbol = str(payload.get("symbol", "")).strip().upper()
        if not symbol:
            return {"ok": False, "error": "symbol is required"}
        command = enqueue_command("close_symbol", {"symbol": symbol})
        return {"ok": True, "queued": command, "control": get_control_state()}

    if action == "close_all":
        command = enqueue_command("close_all", {})
        return {"ok": True, "queued": command, "control": get_control_state()}

    if action == "set_overrides":
        overrides: dict = {}

        if "min_confidence" in payload:
            val = _to_float(payload.get("min_confidence"), 0.0, 1.0)
            if val is None:
                return {"ok": False, "error": "min_confidence must be between 0 and 1"}
            overrides["min_confidence"] = val

        if "correlation_threshold" in payload:
            val = _to_float(payload.get("correlation_threshold"), 0.0, 1.0)
            if val is None:
                return {"ok": False, "error": "correlation_threshold must be between 0 and 1"}
            overrides["correlation_threshold"] = val

        if "max_correlated_positions" in payload:
            val = _to_int(payload.get("max_correlated_positions"), 1, 20)
            if val is None:
                return {"ok": False, "error": "max_correlated_positions must be 1..20"}
            overrides["max_correlated_positions"] = val

        state = update_control_state(overrides=overrides)
        return {"ok": True, "control": state}

    return {"ok": False, "error": f"unsupported action: {action}"}


def _render_html() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>AlphaBot Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

    :root {
      --bg-top: #f6f1e9;
      --bg-bottom: #e7f1f8;
      --panel: rgba(255, 255, 255, 0.78);
      --panel-strong: rgba(255, 255, 255, 0.9);
      --text: #1f2a37;
      --muted: #55657a;
      --line: #d9e3ef;
      --accent: #05668d;
      --accent-soft: #7ec8e3;
      --profit: #0d9488;
      --loss: #b91c1c;
      --shadow: 0 14px 40px rgba(18, 46, 71, 0.12);
      --radius: 18px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--text);
      font-family: 'Space Grotesk', 'Trebuchet MS', sans-serif;
      background:
        radial-gradient(circle at 10% 15%, #ffe8cf 0%, transparent 35%),
        radial-gradient(circle at 85% 25%, #d7f0ff 0%, transparent 40%),
        linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
      min-height: 100vh;
      animation: pageFade 420ms ease-out;
    }

    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      padding: 22px 16px 30px;
    }

    .hero {
      background: linear-gradient(120deg, rgba(5, 102, 141, 0.1), rgba(126, 200, 227, 0.18));
      border: 1px solid rgba(5, 102, 141, 0.15);
      border-radius: var(--radius);
      padding: 18px 18px 16px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(6px);
      margin-bottom: 14px;
    }

    .title {
      margin: 0;
      font-size: clamp(1.25rem, 4.6vw, 2rem);
      letter-spacing: 0.01em;
      font-weight: 700;
      color: #15324a;
    }

    .sub {
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.95rem;
    }

    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
      animation: rise 440ms ease-out both;
    }

    .metric {
      color: var(--muted);
      font-size: 0.86rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }

    .value {
      display: block;
      font-size: 1.4rem;
      font-family: 'IBM Plex Mono', Consolas, monospace;
      margin-top: 6px;
      color: #11344f;
    }

    .panel {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 12px;
      margin-bottom: 14px;
      overflow: hidden;
    }

    .controls {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 12px;
      margin-bottom: 14px;
    }

    .control-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
    }

    .control-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr auto;
      gap: 8px;
      margin-top: 8px;
    }

    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      border: 1px solid var(--line);
      background: #f7fbff;
      color: #0e4362;
      margin-top: 6px;
    }

    .status-chip.paused {
      background: #fff7ed;
      color: #9a3412;
      border-color: #fed7aa;
    }

    button {
      border: 1px solid #c9d9e8;
      background: #fdfefe;
      color: #173f5f;
      border-radius: 10px;
      padding: 8px 12px;
      font-family: inherit;
      font-weight: 600;
      cursor: pointer;
    }

    button:hover { filter: brightness(0.98); }
    button.warn { color: #9a3412; border-color: #fecaca; background: #fff5f5; }
    button.strong { color: #ffffff; border-color: #05668d; background: #05668d; }

    input {
      width: 100%;
      border: 1px solid #c9d9e8;
      border-radius: 10px;
      padding: 8px 10px;
      font-family: inherit;
      color: #173f5f;
      background: #ffffff;
    }

    .inline-actions {
      display: inline-flex;
      gap: 6px;
    }

    .small-btn {
      padding: 5px 8px;
      font-size: 0.76rem;
      border-radius: 8px;
    }

    h2 {
      margin: 4px 6px 10px;
      font-size: 1rem;
      color: #15324a;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }

    th, td {
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }

    th {
      color: #365069;
      font-weight: 600;
      font-size: 0.8rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }

    tr:last-child td { border-bottom: none; }

    .pnl-pos { color: var(--profit); font-weight: 600; }
    .pnl-neg { color: var(--loss); font-weight: 600; }

    @media (max-width: 740px) {
      .wrap { padding: 14px 10px 18px; }
      .controls { grid-template-columns: 1fr; }
      .control-row { grid-template-columns: 1fr; }
      .panel { overflow-x: auto; }
      th, td { padding: 8px 7px; }
    }

    @keyframes pageFade {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <h1 class=\"title\">AlphaBot Portfolio Radar</h1>
      <div class=\"sub\">Live snapshot of open exposure and latest closed trades.</div>
    </div>

    <div class=\"cards\">
      <div class=\"card\"><div class=\"metric\">Balance</div><strong class=\"value\" id=\"balance\">-</strong></div>
      <div class=\"card\"><div class=\"metric\">Open Positions</div><strong class=\"value\" id=\"openCount\">-</strong></div>
      <div class=\"card\"><div class=\"metric\">Closed Trades</div><strong class=\"value\" id=\"closedCount\">-</strong></div>
    </div>

    <div class=\"controls\">
      <div class=\"panel\">
        <h2>Runtime Control</h2>
        <div id=\"botStatus\" class=\"status-chip\">Status: running</div>
        <div class=\"control-actions\">
          <button class=\"warn\" onclick=\"pauseBot()\">Pause New Entries</button>
          <button class=\"strong\" onclick=\"resumeBot()\">Resume</button>
          <button class=\"warn\" onclick=\"closeAllPositions()\">Close All Positions</button>
        </div>
      </div>

      <div class=\"panel\">
        <h2>Live Overrides</h2>
        <div class=\"control-row\">
          <input id=\"cfgMinConfidence\" placeholder=\"Min confidence (0-1)\" />
          <input id=\"cfgCorrThreshold\" placeholder=\"Corr threshold (0-1)\" />
          <input id=\"cfgCorrCount\" placeholder=\"Max correlated\" />
          <button class=\"strong\" onclick=\"applyOverrides()\">Apply</button>
        </div>
      </div>
    </div>

    <div class=\"panel\">
      <h2>Open Positions</h2>
      <table>
        <thead><tr><th>Symbol</th><th>Direction</th><th>Entry</th><th>SL</th><th>TP1</th><th>TP2</th><th>Mark</th><th>Qty</th><th>Guard</th><th>Live PnL</th><th>Action</th></tr></thead>
        <tbody id=\"openRows\"></tbody>
      </table>
    </div>

    <div class=\"panel\">
      <h2>Recent Closed Trades</h2>
      <table>
        <thead><tr><th>Symbol</th><th>Reason</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Close Time</th></tr></thead>
        <tbody id=\"closedRows\"></tbody>
      </table>
    </div>
  </div>

<script>
function pnlClass(v) {
  const n = Number(v || 0);
  return n >= 0 ? 'pnl-pos' : 'pnl-neg';
}

function fmtPrice(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '-';
  return n.toFixed(4);
}

async function apiControl(action, payload = {}) {
  const r = await fetch('/api/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, payload })
  });
  const data = await r.json();
  if (!data.ok) {
    alert(data.error || 'Control action failed');
  }
  return data;
}

async function pauseBot() {
  await apiControl('pause');
  await refresh();
}

async function resumeBot() {
  await apiControl('resume');
  await refresh();
}

async function closeSymbol(symbol) {
  await apiControl('close_symbol', { symbol });
  await refresh();
}

async function closeAllPositions() {
  await apiControl('close_all');
  await refresh();
}

async function applyOverrides() {
  const payload = {};
  const minConf = document.getElementById('cfgMinConfidence').value.trim();
  const corrThr = document.getElementById('cfgCorrThreshold').value.trim();
  const corrCnt = document.getElementById('cfgCorrCount').value.trim();

  if (minConf.length > 0) payload.min_confidence = Number(minConf);
  if (corrThr.length > 0) payload.correlation_threshold = Number(corrThr);
  if (corrCnt.length > 0) payload.max_correlated_positions = Number(corrCnt);

  await apiControl('set_overrides', payload);
  await refresh();
}

async function refresh() {
  const r = await fetch('/api/state');
  const s = await r.json();

  const open = Object.values(s.open_positions || {});
  const closed = (s.closed_trades || []).slice(-25).reverse();
  const control = s.control || { paused: false, overrides: {} };

  document.getElementById('balance').textContent = Number(s.balance || 0).toFixed(2);
  document.getElementById('openCount').textContent = open.length;
  document.getElementById('closedCount').textContent = closed.length;

  const statusEl = document.getElementById('botStatus');
  statusEl.textContent = control.paused ? 'Status: paused (new entries blocked)' : 'Status: running';
  statusEl.classList.toggle('paused', !!control.paused);

  const overrides = control.overrides || {};
  if (overrides.min_confidence !== undefined) {
    document.getElementById('cfgMinConfidence').value = overrides.min_confidence;
  }
  if (overrides.correlation_threshold !== undefined) {
    document.getElementById('cfgCorrThreshold').value = overrides.correlation_threshold;
  }
  if (overrides.max_correlated_positions !== undefined) {
    document.getElementById('cfgCorrCount').value = overrides.max_correlated_positions;
  }

  document.getElementById('openRows').innerHTML = open.map(p =>
    `<tr><td>${p.symbol}</td><td>${p.direction}</td><td>${fmtPrice(p.entry_price)}</td><td>${fmtPrice(p.stop_loss)}</td><td>${fmtPrice(p.take_profit_1)}</td><td>${fmtPrice(p.take_profit_2)}</td><td>${fmtPrice(p.mark_price)}</td><td>${Number(p.quantity||0).toFixed(4)}</td><td>${p.protection_mode || '-'}</td><td class="${pnlClass(p.live_pnl ?? p.pnl)}">${Number((p.live_pnl ?? p.pnl)||0).toFixed(2)}</td><td><span class="inline-actions"><button class="small-btn warn" onclick="closeSymbol('${p.symbol}')">Close</button></span></td></tr>`
  ).join('');

  document.getElementById('closedRows').innerHTML = closed.map(t =>
    `<tr><td>${t.symbol}</td><td>${t.status}</td><td>${Number(t.entry_price||0).toFixed(4)}</td><td>${Number(t.exit_price||0).toFixed(4)}</td><td class="${pnlClass(t.pnl)}">${Number(t.pnl||0).toFixed(2)}</td><td>${t.close_time||''}</td></tr>`
  ).join('');
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def main() -> None:
    host = CONFIG.dashboard.host
    port = CONFIG.dashboard.port
    server = HTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
