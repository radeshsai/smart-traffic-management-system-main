import sqlite3
import os

conn = sqlite3.connect('database/traffic.db')
cur = conn.cursor()

print('--- Row count in simulation_metrics ---')
cur.execute("SELECT COUNT(*) FROM simulation_metrics")
print(cur.fetchone()[0])

print()
print('--- Most recent 5 rows (if any) ---')
cur.execute("SELECT * FROM simulation_metrics ORDER BY id DESC LIMIT 5")
rows = cur.fetchall()
if rows:
    cols = [d[0] for d in cur.description]
    print(cols)
    for r in rows:
        print(r)
else:
    print("No rows at all.")

print()
print('--- All session records (to see which mode ran when) ---')
try:
    cur.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 10")
    cols = [d[0] for d in cur.description]
    print(cols)
    for r in cur.fetchall():
        print(r)
except Exception as e:
    print("Could not read sessions table:", e)

conn.close()
