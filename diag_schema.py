import sqlite3
c = sqlite3.connect("/tmp/ab.db")
cols = [d[0] for d in c.execute("PRAGMA table_info(trades)").fetchall()]
print("TRADES COLUMNS:", cols)
rows = c.execute("SELECT * FROM trades ORDER BY rowid DESC LIMIT 5").fetchall()
print("SAMPLE ROWS:")
for r in rows:
    print(r)
summary = c.execute("""SELECT exit_reason, count(*) as cnt, round(sum(net_pnl),2) as total_pnl FROM trades GROUP BY exit_reason""").fetchall()
print("BY EXIT REASON:", summary)
strat = c.execute("""SELECT strategy_name, direction, count(*) cnt, round(sum(net_pnl),2) pnl FROM trades GROUP BY strategy_name, direction""").fetchall()
print("BY STRATEGY:", strat)
