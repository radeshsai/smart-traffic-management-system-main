"""tests/test_counter.py — Unit tests for VehicleCounter and DensityAnalyzer"""
import sys
import numpy as np
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from src.vehicle_counter import VehicleCounter, CountResult
from src.tracker import TrackingResult, Track
from src.density_analyzer import DensityAnalyzer


@pytest.fixture
def cfg():
    return Config()


def make_track(tid, x1, y1, x2, y2, cls="car"):
    return Track(track_id=tid, class_id=2, class_name=cls,
                 x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9)


class TestVehicleCounter:
    def test_init_direction(self, cfg):
        counter = VehicleCounter(cfg)
        counter.init_direction("north", frame_height=480)
        assert counter._line_y["north"] == int(480 * counter.line_ratio)

    def test_update_returns_count_result(self, cfg):
        counter = VehicleCounter(cfg)
        counter.init_direction("north", frame_height=480)
        tracks = [make_track(1, 100, 200, 150, 250)]
        tracking_result = TrackingResult(direction="north", frame_number=1,
                                         timestamp=0.1, tracks=tracks)
        result = counter.update(tracking_result)
        assert isinstance(result, CountResult)
        assert result.current_count == 1

    def test_no_double_counting(self, cfg):
        counter = VehicleCounter(cfg, line_ratio=0.5)
        counter.init_direction("north", frame_height=200)
        line_y = counter._line_y["north"]   # = 100

        # Vehicle above line
        tracking_result1 = TrackingResult(
            direction="north", frame_number=1, timestamp=0.1,
            tracks=[make_track(1, 50, line_y - 20, 100, line_y - 5)]
        )
        counter.update(tracking_result1)

        # Vehicle crosses line (now below)
        tracking_result2 = TrackingResult(
            direction="north", frame_number=2, timestamp=0.2,
            tracks=[make_track(1, 50, line_y + 5, 100, line_y + 20)]
        )
        result2 = counter.update(tracking_result2)
        assert result2.total_counted == 1

        # Same vehicle keeps moving — should NOT count again
        tracking_result3 = TrackingResult(
            direction="north", frame_number=3, timestamp=0.3,
            tracks=[make_track(1, 50, line_y + 30, 100, line_y + 50)]
        )
        result3 = counter.update(tracking_result3)
        assert result3.total_counted == 1

    def test_reset(self, cfg):
        counter = VehicleCounter(cfg)
        counter.init_direction("south")
        counter._states["south"].total_count = 42
        counter.reset_direction("south")
        assert counter.get_total_counted("south") == 0


class TestDensityAnalyzer:
    def test_classify_low(self, cfg):
        a = DensityAnalyzer(cfg)
        r = a.analyze("north", 5)
        assert r.density_level == "LOW"
        assert r.recommended_green == 20

    def test_classify_medium(self, cfg):
        a = DensityAnalyzer(cfg)
        r = a.analyze("south", 15)
        assert r.density_level == "MEDIUM"
        assert r.recommended_green == 40

    def test_classify_high(self, cfg):
        a = DensityAnalyzer(cfg)
        r = a.analyze("east", 25)
        assert r.density_level == "HIGH"
        assert r.recommended_green == 60

    def test_smoothing(self, cfg):
        a = DensityAnalyzer(cfg)
        # Feed alternating counts
        for count in [0, 30, 0, 30, 0]:
            a.analyze("west", count)
        latest = a.get_latest("west")
        # Smoothed should be somewhere between 0 and 30
        assert 0 < latest.smoothed_count < 30

    def test_congestion_score_range(self, cfg):
        a = DensityAnalyzer(cfg)
        for count in range(0, 35):
            r = a.analyze("north", count)
            assert 0.0 <= r.congestion_score <= 100.0

    def test_trend_detection(self, cfg):
        a = DensityAnalyzer(cfg)
        for count in [2, 4, 8, 12, 18, 24]:
            a.analyze("north", count)
        trend = a.get_trend("north")
        assert trend == "INCREASING"

    def test_summary_report_structure(self, cfg):
        a = DensityAnalyzer(cfg)
        a.analyze("north", 5)
        a.analyze("south", 15)
        report = a.summary_report()
        assert "directions" in report
        assert "intersection" in report
        assert "north" in report["directions"]
