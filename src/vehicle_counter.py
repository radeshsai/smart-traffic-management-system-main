"""
src/vehicle_counter.py — Per-Direction Vehicle Counter
========================================================
Counts vehicles crossing a virtual counting line in each
directional camera feed using track ID history.

Counting logic:
  - Draw a horizontal line at ~60% of frame height.
  - A vehicle is counted when its centre crosses the line
    (tracked by direction of movement).
  - Uses track ID to prevent double-counting the same vehicle.
"""

import numpy as np
import cv2
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from config import Config
from src.tracker import TrackingResult, Track
from src.utils import draw_counting_line


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CountState:
    """Counting state for a single direction."""
    direction: str
    total_count: int = 0
    count_by_class: Dict[str, int] = field(default_factory=lambda: {
        "car": 0, "motorcycle": 0, "bus": 0, "truck": 0
    })
    counted_ids: Set[int] = field(default_factory=set)         # Track IDs already counted
    frame_counts: List[int] = field(default_factory=list)      # Count per frame (history)
    current_frame_count: int = 0                               # Vehicles IN current frame

    @property
    def unique_vehicles(self) -> int:
        """Total unique vehicles counted (crossing events)."""
        return len(self.counted_ids)


@dataclass
class CountResult:
    """Result from one counting update cycle."""
    direction: str
    frame_number: int
    timestamp: float
    current_count: int          # Vehicles visible in this frame
    total_counted: int          # Cumulative unique vehicles counted
    new_crossings: int          # Vehicles that just crossed the line this frame
    count_by_class: Dict[str, int] = field(default_factory=dict)
    line_y: int = 0             # Y-coordinate of counting line

    def to_dict(self) -> Dict:
        return {
            "direction":     self.direction,
            "frame_number":  self.frame_number,
            "timestamp":     round(self.timestamp, 3),
            "current_count": self.current_count,
            "total_counted": self.total_counted,
            "new_crossings": self.new_crossings,
            "count_by_class": self.count_by_class,
        }

    def __str__(self) -> str:
        return (
            f"Count({self.direction} | frame={self.frame_number} | "
            f"visible={self.current_count} | total={self.total_counted} | "
            f"new={self.new_crossings} | {self.count_by_class})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VEHICLE COUNTER
# ═══════════════════════════════════════════════════════════════════════════════

class VehicleCounter:
    """
    Counts vehicles using a virtual counting line per direction.

    The counting line is placed at a configurable fraction of the
    frame height (default: 60%). When a tracked vehicle's centre
    crosses this line (moving downward = entering intersection),
    it is counted once.

    Usage:
        counter = VehicleCounter()
        counter.init_direction("north", frame_height=480)
        count_result = counter.update(tracking_result, frame)
        annotated = counter.annotate(frame, count_result)
    """

    def __init__(
        self,
        config: Config = None,
        line_ratio: float = 0.60,
    ):
        """
        Args:
            config:     Project Config object.
            line_ratio: Counting line position as fraction of frame height.
                        0.6 = 60% down the frame.
        """
        self.config = config or Config()
        self.line_ratio = line_ratio

        # Per-direction state
        self._states: Dict[str, CountState] = {}
        self._line_y: Dict[str, int] = {}           # Pixel Y of counting line per direction
        self._prev_centers: Dict[str, Dict[int, Tuple[int, int]]] = defaultdict(dict)

    # ── Initialization ────────────────────────────────────────────────────────

    def init_direction(self, direction: str, frame_height: int = 480) -> None:
        """
        Set up counting state for a direction.

        Args:
            direction:    Camera direction label.
            frame_height: Height of video frames in pixels.
        """
        if direction not in self._states:
            self._states[direction] = CountState(direction=direction)

        line_y = int(frame_height * self.line_ratio)
        self._line_y[direction] = line_y
        logger.info(
            f"[{direction}] Counter initialized. Counting line at y={line_y} "
            f"({int(self.line_ratio * 100)}% of {frame_height}px)"
        )

    def init_all(self, frame_height: int = 480) -> None:
        """Initialize counters for all configured directions."""
        for d in self.config.video.directions:
            self.init_direction(d, frame_height)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        tracking_result: TrackingResult,
        frame: Optional[np.ndarray] = None,
    ) -> CountResult:
        """
        Update counts based on current tracked vehicles.

        Counts vehicles by:
          1. Recording visible vehicle count in current frame.
          2. Detecting line-crossing events using previous frame centres.
          3. Marking crossed track IDs to prevent double-counting.

        Args:
            tracking_result: Output from VehicleTracker.update().
            frame:           Current video frame (used to get frame height if not initialized).

        Returns:
            CountResult with updated counts.
        """
        direction = tracking_result.direction
        fn = tracking_result.frame_number
        ts = tracking_result.timestamp

        # Auto-initialize if needed
        if direction not in self._states:
            h = frame.shape[0] if frame is not None else 480
            self.init_direction(direction, h)

        state = self._states[direction]
        line_y = self._line_y.get(direction, 288)
        prev_centers = self._prev_centers[direction]

        new_crossings = 0
        # current_count = active tracks in THIS frame only
        current_count = len(tracking_result.tracks)
        # Cap at a realistic max to avoid tracker ghost accumulation
        current_count = min(current_count, 50)
        state.current_frame_count = current_count
        state.frame_counts.append(current_count)

        for track in tracking_result.tracks:
            cx, cy = track.center
            prev_center = prev_centers.get(track.track_id)
            prev_cy = prev_center[1] if prev_center is not None else None

            # Check line crossing: vehicle moves downward across line
            if prev_cy is not None:
                crossed_down = prev_cy < line_y <= cy
                crossed_up   = prev_cy > line_y >= cy

                if (crossed_down or crossed_up) and track.track_id not in state.counted_ids:
                    # Count this vehicle
                    state.counted_ids.add(track.track_id)
                    state.total_count += 1
                    state.count_by_class[track.class_name] = (
                        state.count_by_class.get(track.class_name, 0) + 1
                    )
                    new_crossings += 1
                    logger.debug(
                        f"[{direction}] Vehicle #{track.track_id} ({track.class_name}) "
                        f"crossed line at frame {fn}. Total: {state.total_count}"
                    )

            # Update previous centres
            prev_centers[track.track_id] = (cx, cy)

        # Prune prev_centers for tracks no longer active
        active_ids = {t.track_id for t in tracking_result.tracks}
        stale_ids = set(prev_centers.keys()) - active_ids
        for sid in stale_ids:
            del prev_centers[sid]

        # Build per-frame class breakdown from active tracks (not cumulative)
        frame_by_class = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        for track in tracking_result.tracks:
            cls = track.class_name
            if cls in frame_by_class:
                frame_by_class[cls] += 1

        result = CountResult(
            direction=direction,
            frame_number=fn,
            timestamp=ts,
            current_count=current_count,
            total_counted=state.total_count,
            new_crossings=new_crossings,
            count_by_class=frame_by_class,   # ← per-frame not cumulative
            line_y=line_y,
        )

        if new_crossings > 0:
            logger.info(f"[{direction}] {result}")

        return result

    # ── Annotation ────────────────────────────────────────────────────────────

    def annotate(
        self,
        frame: np.ndarray,
        count_result: CountResult,
    ) -> np.ndarray:
        """Draw counting line on frame."""
        annotated = frame.copy()
        line_y = count_result.line_y

        # Draw counting line — show current frame count not cumulative
        annotated = draw_counting_line(
            annotated,
            line_y=line_y,
            color=(0, 220, 220),
            label=f"COUNT LINE  |  Now: {count_result.current_count}",
        )
        return annotated

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_state(self, direction: str) -> Optional[CountState]:
        """Return raw CountState for a direction."""
        return self._states.get(direction)

    def get_current_count(self, direction: str) -> int:
        """
        Return the number of vehicles currently visible in the latest frame.
        This is the primary input for density analysis.
        """
        state = self._states.get(direction)
        return state.current_frame_count if state else 0

    def get_total_counted(self, direction: str) -> int:
        """Return cumulative unique vehicles counted in this direction."""
        state = self._states.get(direction)
        return state.total_count if state else 0

    def get_all_current_counts(self) -> Dict[str, int]:
        """Return dict of direction → current frame vehicle count."""
        return {d: self.get_current_count(d) for d in self._states}

    def get_summary(self) -> Dict[str, Dict]:
        """Return full summary for all directions."""
        summary = {}
        for direction, state in self._states.items():
            summary[direction] = {
                "current":      state.current_frame_count,
                "total":        state.total_count,
                "by_class":     state.count_by_class,
                "unique_ids":   state.unique_vehicles,
            }
        return summary

    def reset_direction(self, direction: str) -> None:
        """Reset counter state for a specific direction."""
        if direction in self._states:
            self._states[direction] = CountState(direction=direction)
            self._prev_centers[direction] = {}
            logger.info(f"[{direction}] Counter reset.")

    def reset_all(self) -> None:
        """Reset all direction counters."""
        for d in list(self._states.keys()):
            self.reset_direction(d)