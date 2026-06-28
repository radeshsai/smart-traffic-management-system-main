"""
src/tracker.py — Multi-Object Vehicle Tracker
==============================================
Wraps YOLOv8's built-in ByteTrack tracker to assign
persistent track IDs to detected vehicles across frames.

ByteTrack is bundled with Ultralytics — no extra install needed.
Falls back to a lightweight IoU tracker if ByteTrack unavailable.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from config import Config
from src.detector import Detection, DetectionResult
from src.utils import timer


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Track:
    """Represents one tracked vehicle with persistent ID."""
    track_id: int
    class_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    age: int = 1                  # Frames this track has been active
    frames_missing: int = 0       # Consecutive frames without detection
    history: List[Tuple[int, int]] = field(default_factory=list)  # center point history

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict:
        return {
            "track_id":   self.track_id,
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "age": self.age,
        }


@dataclass
class TrackingResult:
    """All active tracks for one frame."""
    direction: str
    frame_number: int
    timestamp: float
    tracks: List[Track] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.tracks)

    @property
    def count_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in self.tracks:
            counts[t.class_name] = counts.get(t.class_name, 0) + 1
        return counts

    @property
    def track_ids(self) -> List[int]:
        return [t.track_id for t in self.tracks]

    def __str__(self) -> str:
        return (
            f"TrackingResult({self.direction} | frame={self.frame_number} | "
            f"tracks={self.count} | {self.count_by_class})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK: LIGHTWEIGHT IoU TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class IoUTracker:
    """
    Minimal IoU-based tracker as fallback when ByteTrack is unavailable.
    Assigns track IDs by matching detections frame-to-frame using IoU.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 3):
        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1
        self._iou_thresh = iou_threshold
        self._max_missing = max_missing

    def _iou(
        self,
        b1: Tuple[int, int, int, int],
        b2: Tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection-over-Union between two bounding boxes."""
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Match detections to existing tracks and return updated track list.

        Args:
            detections: Detections from current frame.

        Returns:
            List of active Track objects with assigned IDs.
        """
        matched_det_ids = set()
        matched_trk_ids = set()

        # Match detections to existing tracks by IoU
        for trk_id, track in self._tracks.items():
            best_iou = self._iou_thresh
            best_det = None
            for det in detections:
                iou = self._iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det is not None:
                # Update existing track
                cx, cy = track.center
                track.x1, track.y1 = best_det.x1, best_det.y1
                track.x2, track.y2 = best_det.x2, best_det.y2
                track.confidence = best_det.confidence
                track.age += 1
                track.frames_missing = 0
                track.history.append((cx, cy))
                if len(track.history) > 30:
                    track.history.pop(0)
                matched_det_ids.add(id(best_det))
                matched_trk_ids.add(trk_id)
            else:
                track.frames_missing += 1

        # Create new tracks for unmatched detections
        for det in detections:
            if id(det) not in matched_det_ids:
                new_track = Track(
                    track_id=self._next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                    confidence=det.confidence,
                    history=[det.center],
                )
                self._tracks[self._next_id] = new_track
                self._next_id += 1

        # Remove stale tracks
        stale = [tid for tid, t in self._tracks.items()
                 if t.frames_missing > self._max_missing]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRACKER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class VehicleTracker:
    """
    Multi-object vehicle tracker using YOLOv8 ByteTrack.

    Maintains one tracker instance per direction for independent tracking.
    Falls back to IoUTracker if ByteTrack/ultralytics is unavailable.

    Usage:
        tracker = VehicleTracker()
        tracker.init_direction("north")
        tracking_result = tracker.update(detection_result)
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._trk_cfg = self.config.tracking
        self._det_cfg = self.config.detection
        self._vehicle_classes = set(self._det_cfg.vehicle_classes)

        # One tracker per direction (ByteTrack or IoU fallback)
        self._yolo_models: Dict[str, object] = {}       # YOLO models for ByteTrack
        self._iou_trackers: Dict[str, IoUTracker] = {}  # Fallback trackers
        self._use_bytetrack: Dict[str, bool] = {}        # Per-direction: was ByteTrack
                                                            # successfully loaded for THIS
                                                            # direction. Previously this was
                                                            # a single shared bool, so one
                                                            # direction's load failure (or
                                                            # success) could silently affect
                                                            # every other direction's choice
                                                            # of tracker.
        self._initialized_directions: List[str] = []

        self._total_unique_ids: Dict[str, int] = {}     # Max track ID seen per direction

    # ── Initialization ────────────────────────────────────────────────────────

    def _det_cfg_model_name(self) -> str:
        """
        Fallback model name for auto-download when the configured weights
        file isn't found on disk yet.

        NOTE: the original code used self._trk_cfg.tracker_type here, which
        is "bytetrack" — the tracking ALGORITHM name, not a YOLO model
        filename. Passing that to YOLO(...) would try to load a file
        literally named "bytetrack.pt", which doesn't exist, so this
        fallback path was already broken before ByteTrack was wired in
        below. self.config.detection.model_name ("yolov8n.pt") mirrors
        what detector.py's own load() method already uses correctly.
        """
        return self.config.detection.model_name

    def _build_tracker_yaml(self) -> str:
        """
        Write a ByteTrack config YAML using this project's own
        TrackingConfig values, and return its path.

        NOTE: without this, model.track(tracker="bytetrack.yaml") uses
        Ultralytics' BUNDLED default tracker config, not this project's
        TrackingConfig — track_high_thresh/new_track_thresh in that bundled
        file are commonly ~0.5/0.6, which sit well above the detection
        confidence floor (0.25) we deliberately lowered elsewhere in this
        project to catch small/distant vehicles. The result: YOLO detects
        the vehicle, but ByteTrack's OWN internal threshold throws it away
        before a track is ever created — the same "too-strict confidence
        gate" bug we already fixed once, just hiding one layer deeper.
        This writes our own tracker yaml so TrackingConfig's values (and a
        track_high_thresh consistent with our 0.25 detection floor) are
        what's actually used.
        """
        import pathlib
        cfg_dir = pathlib.Path(self.config.paths.yolo_model).parent
        yaml_path = cfg_dir / "_project_bytetrack.yaml"

        # track_high_thresh governs which detections are confident enough to
        # be matched/extend a track at all. It should not sit above our
        # detection confidence floor, or qualifying detections get dropped
        # by the tracker after already surviving YOLO's own threshold.
        # new_track_thresh (which detections are confident enough to START
        # a brand new track) can reasonably stay a bit higher than
        # track_high_thresh, to avoid spawning new tracks on every noisy
        # low-confidence blip, while still sitting at/below the configured
        # values rather than the stock file's higher defaults.
        track_high_thresh = min(self._trk_cfg.track_high_thresh, self._det_cfg.confidence)
        new_track_thresh  = min(self._trk_cfg.new_track_thresh, max(self._det_cfg.confidence, track_high_thresh))

        yaml_content = (
            f"tracker_type: bytetrack\n"
            f"track_high_thresh: {track_high_thresh}\n"
            f"track_low_thresh: {self._trk_cfg.track_low_thresh}\n"
            f"new_track_thresh: {new_track_thresh}\n"
            f"track_buffer: {self._trk_cfg.track_buffer}\n"
            f"match_thresh: {self._trk_cfg.match_thresh}\n"
            f"fuse_score: True\n"
        )
        yaml_path.write_text(yaml_content)
        return str(yaml_path)

    def init_direction(self, direction: str) -> None:
        """
        Initialize a tracker for the given direction.

        Args:
            direction: Camera direction (north/south/east/west).
        """
        if direction in self._initialized_directions:
            return

        # Always initialize IoU tracker as fallback.
        #
        # NOTE: this previously reused self._trk_cfg.match_thresh (0.8) as
        # this tracker's IoU threshold. 0.8 is a sensible value for
        # ByteTrack's own internal matching, but far too strict for this
        # simple frame-to-frame IoU tracker — normal detection box jitter
        # between consecutive frames routinely drops below 80% overlap even
        # for a vehicle that hasn't moved much, causing the match to fail
        # and a brand-new track ID to be assigned to the same real vehicle.
        # 0.3 (this class's own constructor default) is a more realistic
        # threshold for basic IoU tracking.
        #
        # max_missing was also lowered from 30 -> 3, which kills a track
        # after just 3 consecutive frames without a matching detection.
        # Combined with the 0.8 threshold above, almost any brief detection
        # gap (occlusion, a confidence dip, motion blur) was enough to
        # delete the track entirely, so the same vehicle got a new ID the
        # moment it was detected again. 10 is a middle ground: forgiving
        # enough to survive a brief miss, without letting tracks for
        # vehicles that have actually left the frame linger as long as the
        # original 30.
        self._iou_trackers[direction] = IoUTracker(
            iou_threshold=0.3,
            max_missing=10,
        )

        try:
            from ultralytics import YOLO
            model_path = str(self.config.paths.yolo_model)
            if not __import__('pathlib').Path(model_path).exists():
                model_path = self._det_cfg_model_name()
            self._yolo_models[direction] = YOLO(model_path)
            if not hasattr(self, "_tracker_yaml_path"):
                self._tracker_yaml_path = self._build_tracker_yaml()
            self._use_bytetrack[direction] = True
            logger.debug(f"[{direction}] ByteTrack tracker initialized.")
        except Exception as e:
            logger.warning(f"[{direction}] ByteTrack unavailable ({e}). Using IoU tracker.")
            self._use_bytetrack[direction] = False

        self._initialized_directions.append(direction)
        self._total_unique_ids[direction] = 0
        logger.info(f"[{direction}] Tracker initialized (ByteTrack={self._use_bytetrack[direction]})")

    def init_all(self) -> None:
        """Initialize trackers for all four directions."""
        for d in self.config.video.directions:
            self.init_direction(d)

    # ── Update ────────────────────────────────────────────────────────────────

    @timer
    def update(
        self,
        raw_frame: np.ndarray,
        direction: str,
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> Tuple[TrackingResult, DetectionResult]:
        """
        Run detection + ByteTrack tracking on a frame and return both the
        tracking result and a DetectionResult view of the same boxes (with
        track_id already populated on each Detection).

        NOTE: this now runs YOLO inference itself via model.track(), rather
        than accepting a pre-computed DetectionResult from VehicleDetector.
        ByteTrack needs to run its own detection pass internally (Ultralytics'
        model.track() doesn't accept externally-supplied boxes), so calling
        VehicleDetector.detect() AND this method on the same frame would run
        inference twice for no benefit. Returning a DetectionResult here too
        means callers that previously used detector.detect()'s output
        (DB writes, live preview annotation) keep working against the same
        shape, now backed by real ByteTrack IDs instead of a separate,
        unlinked detection pass.

        Args:
            raw_frame:    BGR frame to run detection+tracking on.
            direction:    Camera direction label.
            frame_number: Frame index for logging/metadata.
            timestamp:    Time offset in seconds from video start.

        Returns:
            (TrackingResult, DetectionResult) — same underlying detections,
            two views for convenience.
        """
        if direction not in self._initialized_directions:
            self.init_direction(direction)

        if self._use_bytetrack.get(direction, False):
            try:
                tracks = self._update_bytetrack(direction, raw_frame)
            except Exception as e:
                logger.warning(
                    f"[{direction}] ByteTrack inference failed ({e}); "
                    f"falling back to IoU tracker for this frame."
                )
                tracks = []
        else:
            # ByteTrack unavailable for this direction — fall back to a
            # plain YOLO detect pass + the IoU tracker, so tracking still
            # works (with the known IoU limitations) instead of going dark.
            tracks = self._update_iou_from_frame(direction, raw_frame)

        # Update max unique ID counter
        for t in tracks:
            if t.track_id > self._total_unique_ids.get(direction, 0):
                self._total_unique_ids[direction] = t.track_id

        tracking_result = TrackingResult(
            direction=direction,
            frame_number=frame_number,
            timestamp=timestamp,
            tracks=tracks,
        )

        # Build a DetectionResult view of the same boxes, with track_id
        # already populated, for callers that previously relied on
        # VehicleDetector.detect()'s output shape (DB writes, preview
        # annotation in main.py).
        detections = [
            Detection(
                class_id=t.class_id,
                class_name=t.class_name,
                confidence=t.confidence,
                x1=t.x1, y1=t.y1, x2=t.x2, y2=t.y2,
                track_id=t.track_id,
            )
            for t in tracks
        ]
        detection_result = DetectionResult(
            direction=direction,
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
        )

        logger.debug(f"[{direction}] {tracking_result}")
        return tracking_result, detection_result

    def _update_bytetrack(self, direction: str, raw_frame: np.ndarray) -> List[Track]:
        """
        Run YOLOv8 + ByteTrack via Ultralytics' model.track(), returning
        Track objects with the real persistent ByteTrack IDs.

        persist=True is required across calls so the tracker keeps its
        internal state (motion model, ID assignment) between frames for
        this direction's model instance, instead of resetting every call.
        """
        model = self._yolo_models[direction]
        det_cfg = self._det_cfg

        results = model.track(
            source=raw_frame,
            persist=True,
            tracker=getattr(self, "_tracker_yaml_path", "bytetrack.yaml"),
            conf=det_cfg.confidence,
            iou=det_cfg.iou_threshold,
            imgsz=det_cfg.imgsz,
            device=det_cfg.device,
            classes=list(self._vehicle_classes),
            verbose=False,
        )

        tracks: List[Track] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                # boxes.id is None on frames where ByteTrack hasn't
                # assigned any IDs yet (e.g. the very first frame, or a
                # frame with zero detections) — not an error.
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in self._vehicle_classes:
                    continue

                conf = float(boxes.conf[i].item())
                track_id = int(boxes.id[i].item())

                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue

                class_name = det_cfg.class_names.get(cls_id, f"cls_{cls_id}")

                # Reuse the existing track's history/age if we've seen this
                # ID before for this direction, so age/history stay
                # meaningful (Ultralytics itself doesn't give us this).
                existing = self._iou_trackers.get(direction, IoUTracker())._tracks.get(track_id)
                age = existing.age + 1 if existing else 1
                history = (existing.history if existing else [])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                history = history + [(cx, cy)]
                if len(history) > 30:
                    history = history[-30:]

                track = Track(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=class_name,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    age=age,
                    frames_missing=0,
                    history=history,
                )
                tracks.append(track)
                # Stash into the IoU tracker's dict purely as lightweight
                # per-ID memory for age/history continuity above — the IoU
                # tracker's own matching logic is NOT used on this path.
                self._iou_trackers.setdefault(direction, IoUTracker())._tracks[track_id] = track

        return tracks

    def _update_iou_from_frame(self, direction: str, raw_frame: np.ndarray) -> List[Track]:
        """
        Fallback path when ByteTrack isn't available for this direction:
        run a plain YOLO detection pass, then hand boxes to the IoU tracker.
        """
        from src.detector import VehicleDetector  # lazy import, avoid cycle at module load

        if not hasattr(self, "_fallback_detector"):
            self._fallback_detector = VehicleDetector(self.config)
            self._fallback_detector.load()

        det_result = self._fallback_detector.detect(raw_frame, direction=direction)
        return self._iou_trackers[direction].update(det_result.detections)

    # ── Statistics ────────────────────────────────────────────────────────────

    def unique_vehicle_count(self, direction: str) -> int:
        """
        Return total number of unique vehicles ever detected in this direction.
        (Equivalent to the highest track ID assigned.)
        """
        return self._total_unique_ids.get(direction, 0)

    def reset_direction(self, direction: str) -> None:
        """Reset tracker state for a direction (e.g., between video clips)."""
        if direction in self._iou_trackers:
            self._iou_trackers[direction].reset()
            self._total_unique_ids[direction] = 0
            logger.info(f"[{direction}] Tracker state reset.")

    def reset_all(self) -> None:
        """Reset all direction trackers."""
        for d in self._initialized_directions:
            self.reset_direction(d)