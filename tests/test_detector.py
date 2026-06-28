"""tests/test_detector.py — Unit tests for VehicleDetector"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from src.detector import VehicleDetector, Detection, DetectionResult


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def detector(cfg):
    d = VehicleDetector(cfg)
    return d


class TestDetection:
    def test_detection_bbox(self):
        det = Detection(class_id=2, class_name="car", confidence=0.85,
                        x1=10, y1=20, x2=110, y2=120)
        assert det.bbox == (10, 20, 110, 120)
        assert det.center == (60, 70)
        assert det.area == 10000

    def test_detection_to_dict(self):
        det = Detection(class_id=2, class_name="car", confidence=0.85,
                        x1=10, y1=20, x2=110, y2=120)
        d = det.to_dict()
        assert d["class_name"] == "car"
        assert d["confidence"] == 0.85
        assert d["x1"] == 10

    def test_detection_with_track_id(self):
        det = Detection(class_id=3, class_name="motorcycle", confidence=0.72,
                        x1=5, y1=5, x2=50, y2=50, track_id=7)
        assert det.track_id == 7


class TestDetectionResult:
    def test_empty_result(self):
        result = DetectionResult(direction="north", frame_number=1, timestamp=0.1)
        assert result.count == 0
        assert result.count_by_class == {}

    def test_result_with_detections(self):
        dets = [
            Detection(2, "car", 0.9, 0, 0, 100, 100),
            Detection(2, "car", 0.8, 200, 0, 300, 100),
            Detection(3, "motorcycle", 0.7, 400, 0, 450, 80),
        ]
        result = DetectionResult(direction="south", frame_number=5, timestamp=0.5,
                                 detections=dets)
        assert result.count == 3
        assert result.count_by_class["car"] == 2
        assert result.count_by_class["motorcycle"] == 1


class TestVehicleDetector:
    def test_detector_not_loaded_initially(self, detector):
        assert not detector.is_loaded

    def test_detect_returns_result_when_not_loaded(self, detector, dummy_frame):
        result = detector.detect(dummy_frame, direction="north", frame_number=1)
        assert isinstance(result, DetectionResult)
        assert result.count == 0

    def test_annotate_does_not_crash_on_empty(self, detector, dummy_frame):
        result = DetectionResult(direction="north", frame_number=1, timestamp=0.0)
        annotated = detector.annotate(dummy_frame, result)
        assert annotated.shape == dummy_frame.shape

    def test_vehicle_classes_configured(self, cfg):
        assert 2 in cfg.detection.vehicle_classes   # car
        assert 3 in cfg.detection.vehicle_classes   # motorcycle
        assert 5 in cfg.detection.vehicle_classes   # bus
        assert 7 in cfg.detection.vehicle_classes   # truck
