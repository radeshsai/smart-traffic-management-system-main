import sqlite3

conn = sqlite3.connect('database/traffic.db')
cur = conn.cursor()

print('--- per-direction current_count (latest row each) ---')
total_live = 0
for d in ['north', 'south', 'east', 'west']:
    cur.execute(
        'SELECT current_count, created_at FROM vehicle_counts WHERE direction=? ORDER BY id DESC LIMIT 1',
        (d,)
    )
    row = cur.fetchone()
    print(d, '->', row)
    if row and row[0] is not None:
        total_live += row[0]

print('SUM of live vehicles:', total_live)

cur.execute('SELECT COUNT(*) FROM detections')
det = cur.fetchone()[0]
print('Total detections:', det)

print('Computed throughput (detections - live):', max(0, det - total_live))

conn.close()
