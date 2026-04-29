"""Query recent trades from DB for analysis."""
import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "/app/alphabot_data.db"
c = sqlite3.connect(db_path)

print("=== RECENT TRADES (last 30) ===")
rows = c.execute("""
    SELECT strategy_name, direction, entry_price, sl_price, tp1_price,
           exit_reason, net_pnl, open_timestamp, close_timestamp,
           round(cast(tp1_price-entry_price as float)/cast(entry_price-sl_price as float),2) as rr
    FROM trades ORDER BY close_timestamp DESC LIMIT 30
""").fetchall()
for r in rows:
    print(r)

print("\n=== SUMMARY BY EXIT REASON ===")
rows2 = c.execute("""
    SELECT exit_reason, count(*) as cnt, round(sum(net_pnl),2) as total_pnl,
           round(avg(net_pnl),2) as avg_pnl
    FROM trades GROUP BY exit_reason ORDER BY cnt DESC
""").fetchall()
for r in rows2:
    print(r)

print("\n=== SUMMARY BY STRATEGY ===")
rows3 = c.execute("""
    SELECT strategy_name, direction, count(*) as cnt, round(sum(net_pnl),2) as total_pnl,
           round(avg(net_pnl),2) as avg_pnl,
           sum(case when net_pnl > 0 then 1 else 0 end) as wins,
           sum(case when net_pnl <= 0 then 1 else 0 end) as losses
    FROM trades GROUP BY strategy_name, direction
""").fetchall()
for r in rows3:
    print(r)
