"""
src/database.py — SQLite Database Interface
=============================================
Handles all database operations: initialization, inserts,
queries, and reporting. Supports SQLite (default) and
PostgreSQL (via SQLAlchemy connection string).
"""

import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from loguru import logger

from config import Config
from src.detector import DetectionResult
from src.vehicle_counter import CountResult
from src.density_analyzer import DensityReading
from src.signal_controller import SignalState

def _local_ts() -> str:
    """Return current local time as YYYY-MM-DD HH:MM:SS string."""
    from datetime import datetime as _dt
    return _dt.now().strftime('%Y-%m-%d %H:%M:%S')


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """
    Thread-safe SQLite database manager.

    Provides methods to:
      - Initialize schema
      - Insert detections, counts, density readings, signal states, sim metrics
      - Query aggregated data for the dashboard
      - Generate CSV reports

    Usage:
        db = DatabaseManager()
        db.initialize()
        db.insert_detection_result(det_result)
        db.insert_count_result(count_result)
        db.insert_density_reading(density_reading)
        db.insert_signal_state("north", signal_state)
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._db_cfg  = self.config.database
        self._db_path = Path(self._db_cfg.sqlite_path)
        self._schema_path = self.config.paths.schema_path

        # Thread-local connections (one per thread)
        self._local = threading.local()
        self._lock  = threading.Lock()

        # Write-buffer for batched inserts (reduces I/O)
        self._detection_buffer: List[tuple] = []
        self._count_buffer:     List[tuple] = []
        self._density_buffer:   List[tuple] = []
        self._signal_buffer:    List[tuple] = []
        self._sim_buffer:       List[tuple] = []
        self._buffer_size = 100  # Flush every N records
        self._flush_interval = 2.0  # Also flush at least every N seconds,
                                     # so the dashboard's Raw Data Tables
                                     # don't lag behind low-traffic periods
                                     # where buffers take a while to fill.
        import time as _time_mod
        self._time_mod = _time_mod
        self._last_flush = self._time_mod.time()

        self._session_id: Optional[int] = None

    # ── Connection Management ─────────────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            logger.debug(f"New DB connection opened for thread: {threading.current_thread().name}")
        return self._local.conn

    @contextmanager
    def _cursor(self):
        """Context manager that yields a cursor and commits/rolls back."""
        conn = self._get_connection()
        cur  = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error(f"Database error: {exc}")
            raise
        finally:
            cur.close()

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Create database and apply schema.

        Returns:
            True if initialization succeeded.
        """
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

            if self._schema_path.exists():
                schema_sql = self._schema_path.read_text(encoding="utf-8")
            else:
                logger.warning("schema.sql not found — using inline fallback schema.")
                schema_sql = self._inline_schema()

            with self._cursor() as cur:
                cur.executescript(schema_sql)

            logger.success(f"✅ Database initialized at: {self._db_path}")
            return True

        except Exception as exc:
            logger.error(f"Database initialization failed: {exc}")
            return False

    def _inline_schema(self) -> str:
        """Minimal inline schema as fallback if schema.sql is missing."""
        return """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
            direction TEXT, frame_number INTEGER, timestamp_sec REAL,
            class_id INTEGER, class_name TEXT, confidence REAL,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER, track_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS vehicle_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
            direction TEXT, frame_number INTEGER, timestamp_sec REAL,
            current_count INTEGER, total_counted INTEGER,
            count_car INTEGER, count_motorcycle INTEGER,
            count_bus INTEGER, count_truck INTEGER
        );
        CREATE TABLE IF NOT EXISTS density_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
            direction TEXT, vehicle_count INTEGER, density_level TEXT,
            smoothed_count REAL, congestion_score REAL, recommended_green INTEGER
        );
        CREATE TABLE IF NOT EXISTS signal_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
            direction TEXT, phase TEXT, allocated_green INTEGER,
            remaining_seconds REAL, cycle_number INTEGER, density_level TEXT
        );
        CREATE TABLE IF NOT EXISTS simulation_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
            sim_step INTEGER, sim_time REAL, direction TEXT,
            waiting_time REAL, queue_length INTEGER, mean_speed REAL,
            vehicle_count INTEGER, throughput INTEGER, congestion_score REAL, tl_state TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT,
            ended_at TEXT, mode TEXT, video_north TEXT, video_south TEXT,
            video_east TEXT, video_west TEXT, total_frames INTEGER, notes TEXT
        );
        """

    # ── Session Management ────────────────────────────────────────────────────

    def start_session(self, mode: str = "full", notes: str = "") -> int:
        """
        Log the start of an application run.

        Returns:
            Session ID (integer).
        """
        paths = self.config.paths
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (mode, video_north, video_south,
                                      video_east, video_west, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    mode,
                    str(paths.video_paths.get("north", "")),
                    str(paths.video_paths.get("south", "")),
                    str(paths.video_paths.get("east", "")),
                    str(paths.video_paths.get("west", "")),
                    notes,
                ),
            )
            self._session_id = cur.lastrowid
        logger.info(f"Session {self._session_id} started (mode={mode}).")
        return self._session_id

    def end_session(self, total_frames: int = 0) -> None:
        """Log the end of an application run."""
        if self._session_id is None:
            return
        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE sessions
                SET ended_at = strftime('%Y-%m-%d %H:%M:%S','now','localtime'),
                    total_frames = ?
                WHERE id = ?
                """,
                (total_frames, self._session_id),
            )
        logger.info(f"Session {self._session_id} ended. Frames processed: {total_frames}")

    # ── Insert: Detections ────────────────────────────────────────────────────

    def insert_detection_result(self, result: DetectionResult) -> None:
        """Buffer all detections from one DetectionResult."""
        for det in result.detections:
            self._detection_buffer.append((
                result.direction,
                result.frame_number,
                round(result.timestamp, 4),
                det.class_id,
                det.class_name,
                round(det.confidence, 4),
                det.x1, det.y1, det.x2, det.y2,
                det.track_id,
            ))
        if len(self._detection_buffer) >= self._buffer_size or self._due_for_time_flush():
            self._flush_detections()

    def _flush_detections(self) -> None:
        if not self._detection_buffer:
            return
        with self._lock:
            buf = self._detection_buffer[:]
            self._detection_buffer.clear()
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO detections
                    (direction, frame_number, timestamp_sec, class_id, class_name,
                     confidence, x1, y1, x2, y2, track_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                buf,
            )
        logger.debug(f"Flushed {len(buf)} detection records.")

    # ── Insert: Vehicle Counts ────────────────────────────────────────────────

    def insert_count_result(self, result: CountResult) -> None:
        """Insert one vehicle count record using current_count from result."""
        by_cls = result.count_by_class
        self._count_buffer.append((_local_ts(),
            result.direction,
            result.frame_number,
            round(result.timestamp, 4),
            min(result.current_count, 60),  # live count, hard cap at 60
            result.total_counted,
            by_cls.get("car", 0),
            by_cls.get("motorcycle", 0),
            by_cls.get("bus", 0),
            by_cls.get("truck", 0),
        ))
        if len(self._count_buffer) >= self._buffer_size or self._due_for_time_flush():
            self._flush_counts()

    def _flush_counts(self) -> None:
        if not self._count_buffer:
            return
        with self._lock:
            buf = self._count_buffer[:]
            self._count_buffer.clear()
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO vehicle_counts
                    (created_at, direction, frame_number, timestamp_sec, current_count,
                     total_counted, count_car, count_motorcycle, count_bus, count_truck)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                buf,
            )
        logger.debug(f"Flushed {len(buf)} count records.")

    # ── Insert: Density ───────────────────────────────────────────────────────

    def insert_density_reading(self, reading: DensityReading) -> None:
        """Insert one density reading."""
        self._density_buffer.append((_local_ts(),
            reading.direction,
            reading.vehicle_count,
            reading.density_level,
            round(reading.smoothed_count, 4),
            round(reading.congestion_score, 4),
            reading.recommended_green,
        ))
        if len(self._density_buffer) >= self._buffer_size or self._due_for_time_flush():
            self._flush_density()

    def _flush_density(self) -> None:
        if not self._density_buffer:
            return
        with self._lock:
            buf = self._density_buffer[:]
            self._density_buffer.clear()
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO density_logs
                    (created_at, direction, vehicle_count, density_level,
                     smoothed_count, congestion_score, recommended_green)
                VALUES (?,?,?,?,?,?,?)
                """,
                buf,
            )

    # ── Insert: Signal States ─────────────────────────────────────────────────

    def insert_signal_state(self, direction: str, state: SignalState) -> None:
        """Insert a signal state record."""
        self._signal_buffer.append((_local_ts(),
            direction,
            state.phase.value,
            state.allocated_green,
            round(state.remaining_seconds, 2),
            state.cycle_number,
            state.last_density_level,
        ))
        if len(self._signal_buffer) >= self._buffer_size or self._due_for_time_flush():
            self._flush_signals()

    def _flush_signals(self) -> None:
        if not self._signal_buffer:
            return
        with self._lock:
            buf = self._signal_buffer[:]
            self._signal_buffer.clear()
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO signal_states
                    (created_at, direction, phase, allocated_green,
                     remaining_seconds, cycle_number, density_level)
                VALUES (?,?,?,?,?,?,?)
                """,
                buf,
            )

    # ── Insert: Simulation Metrics ────────────────────────────────────────────

    def insert_sim_metrics(self, snapshot) -> None:
        """Insert simulation metrics from a SimSnapshot."""
        for direction, metrics in snapshot.metrics.items():
            self._sim_buffer.append((_local_ts(),
                snapshot.step,
                round(snapshot.sim_time, 3),
                direction,
                round(metrics.waiting_time, 4),
                metrics.queue_length,
                round(metrics.mean_speed, 4),
                metrics.vehicle_count,
                metrics.throughput,
                round(metrics.congestion_score, 4),
                snapshot.tl_state,
            ))
        if len(self._sim_buffer) >= self._buffer_size or self._due_for_time_flush():
            self._flush_sim()

    def _flush_sim(self) -> None:
        if not self._sim_buffer:
            return
        with self._lock:
            buf = self._sim_buffer[:]
            self._sim_buffer.clear()
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO simulation_metrics
                    (created_at, sim_step, sim_time, direction, waiting_time, queue_length,
                     mean_speed, vehicle_count, throughput, congestion_score, tl_state)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                buf,
            )

    # ── Flush All Buffers ─────────────────────────────────────────────────────

    def flush_all(self) -> None:
        """Force-flush all pending write buffers to disk."""
        self._flush_detections()
        self._flush_counts()
        self._flush_density()
        self._flush_signals()
        self._flush_sim()
        self._last_flush = self._time_mod.time()
        logger.debug("All DB buffers flushed.")

    def _due_for_time_flush(self) -> bool:
        """
        True if it's been >= _flush_interval seconds since the last
        time-triggered flush. Resets the timer as a side effect so this
        can be called from each buffer's insert path without all five
        buffers re-triggering on every single call once due.
        """
        now = self._time_mod.time()
        if (now - self._last_flush) >= self._flush_interval:
            self._last_flush = now
            return True
        return False

    # ── Query: Dashboard Data ─────────────────────────────────────────────────

    def get_latest_counts(self) -> List[Dict]:
        """Return latest vehicle count per direction — fully compatible SQL."""
        results = []
        for direction in ["north", "south", "east", "west"]:
            with self._cursor() as cur:
                # Get the single latest row for this direction
                cur.execute(
                    "SELECT * FROM vehicle_counts WHERE direction=? ORDER BY id DESC LIMIT 1",
                    (direction,)
                )
                row = cur.fetchone()
                if row:
                    results.append(dict(row))
        return results

    def get_latest_density(self) -> List[Dict]:
        """Return the latest density reading per direction — simple LIMIT 1 per direction."""
        results = []
        for direction in ["north", "south", "east", "west"]:
            with self._cursor() as cur:
                cur.execute(
                    "SELECT * FROM density_logs WHERE direction=? ORDER BY id DESC LIMIT 1",
                    (direction,)
                )
                row = cur.fetchone()
                if row:
                    results.append(dict(row))
        return results

    def get_latest_signal_states(self) -> List[Dict]:
        """Return the latest signal state per direction — simple LIMIT 1 per direction."""
        results = []
        for direction in ["north", "south", "east", "west"]:
            with self._cursor() as cur:
                cur.execute(
                    "SELECT * FROM signal_states WHERE direction=? ORDER BY id DESC LIMIT 1",
                    (direction,)
                )
                row = cur.fetchone()
                if row:
                    results.append(dict(row))
        return results

    def get_density_timeseries(
        self,
        direction: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict]:
        """Return recent density readings for charting."""
        with self._cursor() as cur:
            if direction:
                cur.execute(
                    """
                    SELECT created_at, direction, vehicle_count,
                           density_level, congestion_score
                    FROM density_logs
                    WHERE direction = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (direction, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT created_at, direction, vehicle_count,
                           density_level, congestion_score
                    FROM density_logs
                    ORDER BY id DESC LIMIT ?
                    """,
                    (limit,),
                )
            rows = [dict(r) for r in cur.fetchall()]
            return list(reversed(rows))

    def get_signal_cycle_summary(self) -> List[Dict]:
        """Return green time allocation summary per direction."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT direction,
                       COUNT(*) AS total_cycles,
                       AVG(allocated_green) AS avg_green,
                       MAX(allocated_green) AS max_green,
                       MIN(allocated_green) AS min_green
                FROM signal_states
                WHERE phase = 'GREEN'
                GROUP BY direction
            """)
            return [dict(r) for r in cur.fetchall()]

    def get_sim_metrics_summary(self) -> List[Dict]:
        """Return aggregated simulation metrics per direction."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT direction,
                       AVG(waiting_time)     AS avg_waiting,
                       MAX(waiting_time)     AS max_waiting,
                       AVG(queue_length)     AS avg_queue,
                       MAX(queue_length)     AS max_queue,
                       AVG(mean_speed)       AS avg_speed,
                       MAX(throughput)       AS total_throughput,
                       AVG(congestion_score) AS avg_congestion
                FROM simulation_metrics
                GROUP BY direction
            """)
            return [dict(r) for r in cur.fetchall()]

    def get_hourly_summary(self) -> List[Dict]:
        """Return hourly vehicle count aggregates."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT direction,
                       strftime('%Y-%m-%d %H:00', created_at) AS hour,
                       SUM(current_count) AS total,
                       AVG(current_count) AS average,
                       MAX(current_count) AS peak
                FROM vehicle_counts
                GROUP BY direction, strftime('%Y-%m-%d %H:00', created_at)
                ORDER BY hour DESC
                LIMIT 96
            """)
            return [dict(r) for r in cur.fetchall()]

    def get_total_stats(self) -> Dict:
        """Return high-level totals for the dashboard header."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM detections")
            total_det = cur.fetchone()["n"]

            cur.execute("SELECT SUM(total_counted) AS n FROM (SELECT direction, MAX(total_counted) AS total_counted FROM vehicle_counts GROUP BY direction)")
            row = cur.fetchone()
            total_vehicles = row["n"] or 0

            cur.execute("SELECT COUNT(*) AS n FROM signal_states WHERE phase='GREEN'")
            total_cycles = cur.fetchone()["n"]

            # "Live Vehicles" on the dashboard = SUM(current_count) over the
            # latest row per direction (same definition analytics.py /
            # get_latest_counts() uses). We mirror that here so throughput
            # is computed against the exact number shown on screen.
            cur.execute("""
                SELECT SUM(current_count) AS n FROM (
                    SELECT direction, current_count,
                           ROW_NUMBER() OVER (PARTITION BY direction ORDER BY id DESC) AS rn
                    FROM vehicle_counts
                ) WHERE rn = 1
            """)
            row = cur.fetchone()
            live_vehicles = row["n"] or 0

            # Throughput = total detections logged so far minus vehicles
            # still live/in-frame right now. Floored at 0 so a cold start
            # (few detections, nonzero live count) never shows negative.
            throughput = max(0, total_det - live_vehicles)

        return {
            "total_detections":  total_det,
            "total_vehicles":    total_vehicles,
            "signal_cycles":     total_cycles,
            "sim_throughput":    throughput,
        }


    # ── Reporting ─────────────────────────────────────────────────────────────

    def export_csv(self, table: str, output_path: Path) -> bool:
        """
        Export a table to CSV.

        Args:
            table:       Table name (e.g. 'density_logs').
            output_path: Path for the CSV file.

        Returns:
            True if exported successfully.
        """
        import csv
        try:
            with self._cursor() as cur:
                cur.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 10000")
                rows = cur.fetchall()
                if not rows:
                    logger.warning(f"No data in table '{table}'.")
                    return False
                cols = [desc[0] for desc in cur.description]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                writer.writerows([dict(r) for r in rows])

            logger.success(f"Exported {len(rows)} rows → {output_path}")
            return True
        except Exception as exc:
            logger.error(f"CSV export failed for '{table}': {exc}")
            return False

    def export_all_reports(self) -> None:
        """Export all tables to CSV in outputs/reports/."""
        reports_dir = self.config.paths.reports_dir
        tables = [
            "detections", "vehicle_counts", "density_logs",
            "signal_states", "simulation_metrics",
        ]
        for table in tables:
            self.export_csv(table, reports_dir / f"{table}.csv")

    # ── Maintenance ───────────────────────────────────────────────────────────

    def prune_old_records(self) -> None:
        """Remove oldest rows when tables exceed max_rows_per_table."""
        max_rows = self._db_cfg.max_rows_per_table
        tables = [
            "detections", "vehicle_counts", "density_logs",
            "signal_states", "simulation_metrics",
        ]
        with self._cursor() as cur:
            for table in tables:
                cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
                count = cur.fetchone()["n"]
                if count > max_rows:
                    to_delete = count - max_rows
                    cur.execute(
                        f"DELETE FROM {table} WHERE id IN "
                        f"(SELECT id FROM {table} ORDER BY id ASC LIMIT ?)",
                        (to_delete,),
                    )
                    logger.info(f"Pruned {to_delete} old rows from '{table}'.")

    def close(self) -> None:
        """Flush buffers and close all connections."""
        self.flush_all()
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
        logger.info("Database connection closed.")
