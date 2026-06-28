"""tests/test_dashboard.py — Unit tests for Database and Analytics"""
import sys
import pytest
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from src.database import DatabaseManager
from src.detector import DetectionResult, Detection
from src.vehicle_counter import CountResult
from src.density_analyzer import DensityAnalyzer, DensityReading
from src.signal_controller import SignalController, SignalPhase, SignalState


@pytest.fixture
def tmp_db(tmp_path):
    """Return a DatabaseManager pointing to a temp SQLite file."""
    cfg = Config()
    cfg.database.sqlite_path = str(tmp_path / "test_traffic.db")
    cfg._paths = None   # Allow path re-init
    db = DatabaseManager(cfg)
    db.initialize()
    return db


class TestDatabaseInit:
    def test_initialize_creates_file(self, tmp_db, tmp_path):
        db_files = list(tmp_path.glob("*.db"))
        assert len(db_files) == 1

    def test_tables_exist(self, tmp_db):
        with tmp_db._cursor() as cur:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}
        expected = {"detections", "vehicle_counts", "density_logs",
                    "signal_states", "simulation_metrics", "sessions"}
        assert expected.issubset(tables)


class TestDatabaseInserts:
    def test_insert_detection_result(self, tmp_db):
        det = Detection(2, "car", 0.9, 0, 0, 100, 100)
        result = DetectionResult("north", 1, 0.1, [det])
        tmp_db.insert_detection_result(result)
        tmp_db.flush_all()
        with tmp_db._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM detections")
            assert cur.fetchone()[0] == 1

    def test_insert_count_result(self, tmp_db):
        result = CountResult(
            direction="south", frame_number=1, timestamp=0.1,
            current_count=5, total_counted=5, new_crossings=2,
            count_by_class={"car": 3, "motorcycle": 2, "bus": 0, "truck": 0},
        )
        tmp_db.insert_count_result(result)
        tmp_db.flush_all()
        with tmp_db._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM vehicle_counts")
            assert cur.fetchone()[0] == 1

    def test_insert_density_reading(self, tmp_db):
        cfg = Config()
        analyzer = DensityAnalyzer(cfg)
        reading = analyzer.analyze("east", 20)
        tmp_db.insert_density_reading(reading)
        tmp_db.flush_all()
        with tmp_db._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM density_logs")
            assert cur.fetchone()[0] == 1

    def test_session_lifecycle(self, tmp_db):
        sid = tmp_db.start_session(mode="test")
        assert sid is not None
        tmp_db.end_session(total_frames=100)
        with tmp_db._cursor() as cur:
            cur.execute("SELECT total_frames FROM sessions WHERE id=?", (sid,))
            row = cur.fetchone()
            assert row[0] == 100


class TestDatabaseQueries:
    def _populate(self, db):
        cfg = Config()
        analyzer = DensityAnalyzer(cfg)
        for direction in ["north", "south", "east", "west"]:
            reading = analyzer.analyze(direction, 10)
            db.insert_density_reading(reading)
            count_r = CountResult(
                direction=direction, frame_number=1, timestamp=0.1,
                current_count=10, total_counted=10, new_crossings=1,
                count_by_class={"car": 8, "motorcycle": 2, "bus": 0, "truck": 0},
            )
            db.insert_count_result(count_r)
        db.flush_all()

    def test_get_latest_density(self, tmp_db):
        self._populate(tmp_db)
        rows = tmp_db.get_latest_density()
        assert len(rows) == 4
        directions = {r["direction"] for r in rows}
        assert directions == {"north", "south", "east", "west"}

    def test_get_latest_counts(self, tmp_db):
        self._populate(tmp_db)
        rows = tmp_db.get_latest_counts()
        assert len(rows) == 4

    def test_density_timeseries(self, tmp_db):
        self._populate(tmp_db)
        rows = tmp_db.get_density_timeseries(direction="north")
        assert len(rows) >= 1

    def test_export_csv(self, tmp_db, tmp_path):
        self._populate(tmp_db)
        out = tmp_path / "density_logs.csv"
        result = tmp_db.export_csv("density_logs", out)
        assert result is True
        assert out.exists()
        content = out.read_text()
        assert "direction" in content
