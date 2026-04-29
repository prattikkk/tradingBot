import sqlite3
from pathlib import Path


def main() -> None:
    db_path = Path("alphabot_data.db")
    print(f"DB exists: {db_path.exists()}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print("\n--- SIGNAL LOG COUNT ---")
    row = cur.execute("SELECT COUNT(*) AS cnt FROM signals_log").fetchone()
    print(f"Total signals: {row['cnt']}")

    print("\n--- RECENT 50 SIGNALS ---")
    rows = cur.execute(
        "SELECT timestamp, symbol, strategy_name, confidence, approved, rejection_reason "
        "FROM signals_log ORDER BY timestamp DESC LIMIT 50"
    ).fetchall()
    for r in rows:
        print(
            "{0} | {1} | {2} | conf={3:.1f} | approved={4} | {5}".format(
                r["timestamp"],
                r["symbol"],
                r["strategy_name"],
                r["confidence"],
                r["approved"],
                r["rejection_reason"] or "",
            )
        )

    print("\n--- REJECTION SUMMARY (LAST 500) ---")
    rows = cur.execute(
        "SELECT strategy_name, approved, rejection_reason, COUNT(*) AS cnt "
        "FROM (SELECT strategy_name, approved, rejection_reason "
        "FROM signals_log ORDER BY timestamp DESC LIMIT 500) "
        "GROUP BY strategy_name, approved, rejection_reason ORDER BY cnt DESC"
    ).fetchall()
    for r in rows:
        print(
            "{0} | approved={1} | {2} | {3}".format(
                r["strategy_name"],
                r["approved"],
                r["rejection_reason"] or "",
                r["cnt"],
            )
        )

    conn.close()


if __name__ == "__main__":
    main()
