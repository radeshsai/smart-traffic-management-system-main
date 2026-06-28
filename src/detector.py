"""
src/detector.py — YOLOv8 Vehicle Detector
==========================================
Wraps Ultralytics YOLOv8 to detect vehicles
(cars, motorcycles, buses, trucks) in video frames.

Detected classes (COCO IDs):
  2  → car
  3  → motorcycle
  5  → bus
  7  → truck
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from loguru import logger

from config import Config
from src.utils import draw_bounding_box, timer


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """Represents a single vehicle detection in one frame."""
    class_id: int           # COCO class ID (2, 3, 5, 7)
    class_name: str         # "car" | "motorcycle" | "bus" | "truck"
    confidence: float       # Model confidence score (0.0–1.0)
    x1: int                 # Bounding box top-left x
    y1: int                 # Bounding box top-left y
    x2: int                 # Bounding box bottom-right x
    y2: int                 # Bounding box bottom-right y
    track_id: Optional[int] = None   # Filled in by tracker

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounding box."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[int, int]:
        """Return bounding box centre (cx, cy)."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        """Return bounding box area in pixels²."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_dict(self) -> Dict:
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "track_id":   self.track_id,
        }


@dataclass
class DetectionResult:
    """All detections from one frame, for one direction."""
    direction: str
    frame_number: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def count_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    def __str__(self) -> str:
        return (
            f"DetectionResult({self.direction} | frame={self.frame_number} | "
            f"count={self.count} | {self.count_by_class} | "
            f"{self.inference_ms:.1f}ms)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# YOLOV8 DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class VehicleDetector:
    """
    YOLOv8-based vehicle detector.

    Loads the YOLOv8n model once and runs inference on frames.
    Filters detections to vehicle-only COCO classes.

    Usage:
        detector = VehicleDetector()
        detector.load()
        result = detector.detect(frame, direction="north", frame_number=1)
        annotated = detector.annotate(frame, result)
    """

    def __init__(self, config: Config = None):
        """
        Args:
            config: Project Config object (uses defaults if None).
        """
        self.config = config or Config()
        self._model = None
        self._loaded = False

        # Shorthand config references
        self._det_cfg  = self.config.detection
        self._model_path = str(self.config.paths.yolo_model)
        self._vehicle_classes = set(self._det_cfg.vehicle_classes)

    # ── Model Loading ─────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Load YOLOv8 model weights.

        Ultralytics auto-downloads yolov8n.pt if not found locally.

        Returns:
            True if loaded successfully.
        """
        try:
            from ultralytics import YOLO  # Lazy import (heavy)

            model_path = self._model_path
            if not Path(model_path).exists():
                logger.warning(
                    f"Model not found at {model_path}. "
                    f"Ultralytics will auto-download yolov8n.pt …"
                )
                model_path = self._det_cfg.model_name  # Use name for auto-download

            logger.info(f"Loading YOLOv8 model: {model_path} on device={self._det_cfg.device}")
            self._model = YOLO(model_path)
            self._model.to(self._det_cfg.device)
            self._loaded = True
            logger.success(f"✅ YOLOv8 model loaded. Classes: {self._det_cfg.class_names}")
            return True

        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as exc:
            logger.error(f"Failed to load YOLOv8 model: {exc}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    # ── Detection ─────────────────────────────────────────────────────────────

    @timer
    def detect(
        self,
        frame: np.ndarray,
        direction: str = "unknown",
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> DetectionResult:
        """
        Run YOLOv8 inference on a single frame.

        Args:
            frame:        BGR numpy array (H, W, 3).
            direction:    Camera direction label.
            frame_number: Frame index for logging.
            timestamp:    Time offset in seconds from video start.

        Returns:
            DetectionResult with all vehicle detections.
        """
        if not self.is_loaded:
            logger.error("Detector not loaded. Call load() first.")
            return DetectionResult(
                direction=direction,
                frame_number=frame_number,
                timestamp=timestamp,
            )

        import time
        t0 = time.perf_counter()

        try:
            results = self._model.predict(
                source=frame,
                conf=self._det_cfg.confidence,
                iou=self._det_cfg.iou_threshold,
                imgsz=self._det_cfg.imgsz,
                device=self._det_cfg.device,
                classes=list(self._vehicle_classes),
                verbose=False,
            )
        except Exception as exc:
            logger.error(f"[{direction}] YOLO inference failed at frame {frame_number}: {exc}")
            return DetectionResult(
                direction=direction,
                frame_number=frame_number,
                timestamp=timestamp,
            )

        inference_ms = (time.perf_counter() - t0) * 1000
        detections = self._parse_results(results)

        result = DetectionResult(
            direction=direction,
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            inference_ms=inference_ms,
        )

        logger.debug(f"[{direction}] {result}")
        return result

    def _parse_results(self, results) -> List[Detection]:
        """
        Parse raw Ultralytics results into Detection objects.

        Args:
            results: List of ultralytics.engine.results.Results.

        Returns:
            List of Detection objects (vehicle classes only).
        """
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())

                # Filter to vehicle classes only
                if cls_id not in self._vehicle_classes:
                    continue

                conf = float(boxes.conf[i].item())
                if conf < self._det_cfg.confidence:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

                # Sanity check: box must have positive area
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue

                class_name = self._det_cfg.class_names.get(cls_id, f"cls_{cls_id}")

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))

        return detections

    # ── Batch Detection ───────────────────────────────────────────────────────

    def detect_batch(
        self,
        frames: List[np.ndarray],
        direction: str = "unknown",
    ) -> List[DetectionResult]:
        """
        Run YOLOv8 inference on a batch of frames (more efficient on GPU).

        Args:
            frames:    List of BGR numpy arrays.
            direction: Camera direction label.

        Returns:
            List of DetectionResult (one per frame).
        """
        if not self.is_loaded:
            logger.error("Detector not loaded. Call load() first.")
            return []

        if not frames:
            return []

        import time
        t0 = time.perf_counter()

        try:
            results = self._model.predict(
                source=frames,
                conf=self._det_cfg.confidence,
                iou=self._det_cfg.iou_threshold,
                imgsz=self._det_cfg.imgsz,
                device=self._det_cfg.device,
                classes=list(self._vehicle_classes),
                verbose=False,
            )
        except Exception as exc:
            logger.error(f"[{direction}] Batch inference failed: {exc}")
            return [
                DetectionResult(direction=direction, frame_number=i, timestamp=float(i))
                for i in range(len(frames))
            ]

        inference_ms = (time.perf_counter() - t0) * 1000
        per_frame_ms = inference_ms / len(frames)

        return [
            DetectionResult(
                direction=direction,
                frame_number=i,
                timestamp=float(i),
                detections=self._parse_results([results[i]]),
                inference_ms=per_frame_ms,
            )
            for i in range(len(results))
        ]

    # ── Frame Annotation ──────────────────────────────────────────────────────

    def annotate(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a frame.

        Args:
            frame:           BGR image array (will NOT be modified in-place).
            result:          DetectionResult from detect().
            show_confidence: If True, append confidence score to label.

        Returns:
            New annotated frame as BGR numpy array.
        """
        annotated = frame.copy()

        for det in result.detections:
            color = self._det_cfg.class_colors.get(det.class_id, (200, 200, 200))
            label = det.class_name
            if show_confidence:
                label += f" {det.confidence:.0%}"

            draw_bounding_box(
                annotated,
                det.x1, det.y1, det.x2, det.y2,
                label=label,
                color=color,
                track_id=det.track_id,
            )

        # Vehicle count badge (bottom-left)
        h, w = annotated.shape[:2]
        badge_text = f"Vehicles: {result.count}"
        cv2.rectangle(annotated, (0, h - 32), (160, h), (20, 20, 20), -1)
        cv2.putText(
            annotated, badge_text,
            (6, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA
        )

        return annotated

    # ── Warmup ────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Run 3 dummy inferences to fully warm up the model.
        Eliminates first-frame latency spikes.
        """
        if not self.is_loaded:
            return
        dummy = np.zeros((416, 416, 3), dtype=np.uint8)
        for _ in range(3):
            self.detect(dummy, direction="warmup", frame_number=0)
        logger.debug("YOLOv8 warmup complete (3 passes).")