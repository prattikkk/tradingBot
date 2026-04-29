import sqlite3
import os

db_path = '/home/ubuntu/tradingBot/alphabot_data.db'
if not os.path.exists(db_path):
    print(f'Error: {db_path} not found')
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print('--- TOTAL TRADES ---')
res = cursor.execute('SELECT COUNT(*) FROM trades').fetchone()
print(f'Total: {res[0]}')

print('\n--- EXIT REASON ANALYSIS ---')
res = cursor.execute('SELECT exit_reason, COUNT(*), SUM(net_pnl) FROM trades GROUP BY exit_reason').fetchall()
for row in res:
    print(f'{row[0]}: Count={row[1]}, NetPnL={row[2]:.2f}')

print('\n--- STRATEGY ANALYSIS ---')
res = cursor.execute('SELECT strategy_name, COUNT(*), SUM(net_pnl) FROM trades GROUP BY strategy_name').fetchall()
for row in res:
    print(f'{row[0]}: Count={row[1]}, NetPnL={row[2]:.2f}')

print('\n--- BREAKEVEN_STOP METRICS ---')
res = cursor.execute("SELECT AVG(gross_pnl), AVG(fees), AVG(net_pnl) FROM trades WHERE exit_reason=\"BREAKEVEN_STOP\"").fetchone()
if res and res[0] is not None:
    print(f'Avg Gross={res[0]:.4f}, Avg Fees={res[1]:.4f}, Avg Net={res[2]:.4f}')
else:
    print('No BREAKEVEN_STOP trades found.')

print('\n--- RECENT 20 TRADES ---')
res = cursor.execute('SELECT symbol, strategy_name, exit_reason, gross_pnl, fees, net_pnl FROM trades ORDER BY exit_time DESC LIMIT 20').fetchall()
for row in res:
    print(f"{row['symbol']} | {row['strategy_name']} | {row['exit_reason']} | G:{row['gross_pnl']:.2f} | F:{row['fees']:.2f} | N:{row['net_pnl']:.2f}")

conn.close()
