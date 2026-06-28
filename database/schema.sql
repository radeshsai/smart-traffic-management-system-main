-- ============================================================
-- schema.sql — SQLite Database Schema
-- AI-Driven Smart Traffic Management System
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── 1. Detections ─────────────────────────────────────────────────
-- Raw vehicle detections from YOLOv8 per frame
CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
    direction       TEXT    NOT NULL CHECK(direction IN ('north','south','east','west')),
    frame_number    INTEGER NOT NULL,
    timestamp_sec   REAL    NOT NULL,
    class_id        INTEGER NOT NULL,
    class_name      TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    x1              INTEGER NOT NULL,
    y1              INTEGER NOT NULL,
    x2              INTEGER NOT NULL,
    y2              INTEGER NOT NULL,
    track_id        INTEGER
);

CREATE INDEX IF NOT EXISTS idx_detections_direction  ON detections(direction);
CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections(created_at);

-- ── 2. Vehicle Counts ──────────────────────────────────────────────
-- Aggregated vehicle count per frame per direction
CREATE TABLE IF NOT EXISTS vehicle_counts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
    direction       TEXT    NOT NULL,
    frame_number    INTEGER NOT NULL,
    timestamp_sec   REAL    NOT NULL,
    current_count   INTEGER NOT NULL DEFAULT 0,
    total_counted   INTEGER NOT NULL DEFAULT 0,
    count_car       INTEGER NOT NULL DEFAULT 0,
    count_motorcycle INTEGER NOT NULL DEFAULT 0,
    count_bus       INTEGER NOT NULL DEFAULT 0,
    count_truck     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_vehicle_counts_direction  ON vehicle_counts(direction);
CREATE INDEX IF NOT EXISTS idx_vehicle_counts_created_at ON vehicle_counts(created_at);

-- ── 3. Density Logs ────────────────────────────────────────────────
-- Classified traffic density readings per direction
CREATE TABLE IF NOT EXISTS density_logs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
    direction        TEXT    NOT NULL,
    vehicle_count    INTEGER NOT NULL,
    density_level    TEXT    NOT NULL CHECK(density_level IN ('LOW','MEDIUM','HIGH')),
    smoothed_count   REAL    NOT NULL,
    congestion_score REAL    NOT NULL,
    recommended_green INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_density_direction  ON density_logs(direction);
CREATE INDEX IF NOT EXISTS idx_density_created_at ON density_logs(created_at);

-- ── 4. Signal States ───────────────────────────────────────────────
-- Traffic signal state transitions
CREATE TABLE IF NOT EXISTS signal_states (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
    direction           TEXT    NOT NULL,
    phase               TEXT    NOT NULL CHECK(phase IN ('GREEN','YELLOW','RED','ALL_RED')),
    allocated_green     INTEGER NOT NULL DEFAULT 0,
    remaining_seconds   REAL    NOT NULL DEFAULT 0,
    cycle_number        INTEGER NOT NULL DEFAULT 0,
    density_level       TEXT    NOT NULL DEFAULT 'LOW'
);

CREATE INDEX IF NOT EXISTS idx_signal_direction  ON signal_states(direction);
CREATE INDEX IF NOT EXISTS idx_signal_created_at ON signal_states(created_at);

-- ── 5. Simulation Metrics ──────────────────────────────────────────
-- SUMO TraCI metrics per direction per step
CREATE TABLE IF NOT EXISTS simulation_metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
    sim_step         INTEGER NOT NULL,
    sim_time         REAL    NOT NULL,
    direction        TEXT    NOT NULL,
    waiting_time     REAL    NOT NULL DEFAULT 0,
    queue_length     INTEGER NOT NULL DEFAULT 0,
    mean_speed       REAL    NOT NULL DEFAULT 0,
    vehicle_count    INTEGER NOT NULL DEFAULT 0,
    throughput       INTEGER NOT NULL DEFAULT 0,
    congestion_score REAL    NOT NULL DEFAULT 0,
    tl_state         TEXT
);

CREATE INDEX IF NOT EXISTS idx_sim_step      ON simulation_metrics(sim_step);
CREATE INDEX IF NOT EXISTS idx_sim_direction ON simulation_metrics(direction);

-- ── 6. Session Log ────────────────────────────────────────────────
-- One row per application run (for audit / reporting)
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
    ended_at     TEXT,
    mode         TEXT NOT NULL DEFAULT 'full',
    video_north  TEXT,
    video_south  TEXT,
    video_east   TEXT,
    video_west   TEXT,
    total_frames INTEGER DEFAULT 0,
    notes        TEXT
);

-- ── Views ──────────────────────────────────────────────────────────

-- Latest density per direction
CREATE VIEW IF NOT EXISTS v_latest_density AS
SELECT d.direction,
       d.vehicle_count,
       d.density_level,
       d.congestion_score,
       d.recommended_green,
       d.created_at
FROM density_logs d
INNER JOIN (
    SELECT direction, MAX(id) AS max_id
    FROM density_logs
    GROUP BY direction
) latest ON d.direction = latest.direction AND d.id = latest.max_id;

-- Hourly vehicle count summary
CREATE VIEW IF NOT EXISTS v_hourly_counts AS
SELECT direction,
       strftime('%Y-%m-%d %H:00:00', created_at) AS hour,
       SUM(current_count)  AS total_vehicles,
       AVG(current_count)  AS avg_vehicles,
       MAX(current_count)  AS peak_vehicles,
       COUNT(*)            AS readings
FROM vehicle_counts
GROUP BY direction, strftime('%Y-%m-%d %H:00:00', created_at);

-- Signal cycle summary
CREATE VIEW IF NOT EXISTS v_signal_cycle_summary AS
SELECT direction,
       phase,
       COUNT(*)            AS phase_count,
       AVG(allocated_green) AS avg_green_time,
       MAX(allocated_green) AS max_green_time
FROM signal_states
WHERE phase = 'GREEN'
GROUP BY direction, phase;
