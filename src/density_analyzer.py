"""
src/density_analyzer.py — Traffic Density Analyzer
====================================================
Classifies traffic density per direction based on vehicle counts,
applies smoothing, and generates congestion scores.

Density Rules (from config):
  0  – 10  vehicles → LOW    (green time: 20s)
  11 – 20  vehicles → MEDIUM (green time: 40s)
  21+      vehicles → HIGH   (green time: 60s)
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from config import Config
from src.utils import moving_average, timestamp


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DensityReading:
    """Single density measurement for one direction at one point in time."""
    direction: str
    timestamp: str
    vehicle_count: int
    density_level: str          # LOW | MEDIUM | HIGH
    smoothed_count: float       # Moving-average smoothed count
    congestion_score: float     # 0–100 composite score
    recommended_green: int      # Signal green time (seconds)

    def to_dict(self) -> Dict:
        return {
            "direction":         self.direction,
            "timestamp":         self.timestamp,
            "vehicle_count":     self.vehicle_count,
            "density_level":     self.density_level,
            "smoothed_count":    round(self.smoothed_count, 2),
            "congestion_score":  round(self.congestion_score, 2),
            "recommended_green": self.recommended_green,
        }

    def __str__(self) -> str:
        return (
            f"Density({self.direction} | count={self.vehicle_count} | "
            f"level={self.density_level} | score={self.congestion_score:.1f} | "
            f"green={self.recommended_green}s)"
        )


@dataclass
class IntersectionDensity:
    """Aggregated density snapshot across all four directions."""
    timestamp: str
    readings: Dict[str, DensityReading] = field(default_factory=dict)

    @property
    def highest_density_direction(self) -> Optional[str]:
        """Return the direction with highest congestion score."""
        if not self.readings:
            return None
        return max(self.readings, key=lambda d: self.readings[d].congestion_score)

    @property
    def total_vehicles(self) -> int:
        return sum(r.vehicle_count for r in self.readings.values())

    @property
    def average_congestion(self) -> float:
        if not self.readings:
            return 0.0
        return sum(r.congestion_score for r in self.readings.values()) / len(self.readings)

    @property
    def overall_level(self) -> str:
        """Overall intersection density level."""
        avg = self.average_congestion
        if avg < 33:
            return "LOW"
        elif avg < 66:
            return "MEDIUM"
        return "HIGH"

    def to_dict(self) -> Dict:
        return {
            "timestamp":                  self.timestamp,
            "total_vehicles":             self.total_vehicles,
            "average_congestion":         round(self.average_congestion, 2),
            "overall_level":              self.overall_level,
            "highest_density_direction":  self.highest_density_direction,
            "per_direction":              {d: r.to_dict() for d, r in self.readings.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DENSITY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class DensityAnalyzer:
    """
    Analyzes per-direction vehicle counts to classify traffic density.

    Features:
      - Smoothed count via configurable rolling window
      - Congestion score (0–100) based on count + trend
      - Historical data storage per direction
      - Intersection-level aggregated snapshot

    Usage:
        analyzer = DensityAnalyzer()
        reading = analyzer.analyze("north", vehicle_count=15)
        snapshot = analyzer.get_intersection_snapshot()
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._den_cfg = self.config.density
        self._sig_cfg = self.config.signal

        # Rolling window of raw counts per direction
        window = self._den_cfg.smoothing_window
        self._count_history: Dict[str, deque] = {
            d: deque(maxlen=window)
            for d in self.config.video.directions
        }

        # Full reading history (for charting / DB)
        self._reading_history: Dict[str, List[DensityReading]] = {
            d: [] for d in self.config.video.directions
        }

        # Latest reading per direction
        self._latest: Dict[str, Optional[DensityReading]] = {
            d: None for d in self.config.video.directions
        }

        logger.info(
            f"DensityAnalyzer initialized. "
            f"Thresholds: LOW≤{self._den_cfg.low_max} | "
            f"MEDIUM≤{self._den_cfg.medium_max} | HIGH>{self._den_cfg.medium_max}"
        )

    # ── Core Analysis ─────────────────────────────────────────────────────────

    def analyze(
        self,
        direction: str,
        vehicle_count: int,
    ) -> DensityReading:
        """
        Classify traffic density for a direction given a vehicle count.

        Args:
            direction:     Camera direction (north/south/east/west).
            vehicle_count: Raw vehicle count from VehicleCounter.

        Returns:
            DensityReading with level, score, and recommended green time.
        """
        vehicle_count = max(0, vehicle_count)

        # Update rolling history
        if direction not in self._count_history:
            self._count_history[direction] = deque(maxlen=self._den_cfg.smoothing_window)
        self._count_history[direction].append(vehicle_count)

        # Smoothed count (moving average)
        history_list = list(self._count_history[direction])
        smoothed = moving_average(history_list, window=self._den_cfg.smoothing_window)

        # Classify using smoothed count
        density_level = self._den_cfg.classify(int(round(smoothed)))

        # Congestion score 0–100
        congestion_score = self._compute_congestion_score(
            direction=direction,
            current_count=vehicle_count,
            smoothed_count=smoothed,
            history=history_list,
        )

        # Recommended green time
        green_time = self._sig_cfg.get_green_time(density_level)

        reading = DensityReading(
            direction=direction,
            timestamp=timestamp(),
            vehicle_count=vehicle_count,
            density_level=density_level,
            smoothed_count=smoothed,
            congestion_score=congestion_score,
            recommended_green=green_time,
        )

        # Store
        if direction not in self._reading_history:
            self._reading_history[direction] = []
        self._reading_history[direction].append(reading)
        self._latest[direction] = reading

        logger.debug(f"[{direction}] {reading}")
        return reading

    def analyze_all(
        self,
        counts: Dict[str, int],
    ) -> Dict[str, DensityReading]:
        """
        Analyze density for all directions at once.

        Args:
            counts: Dict of direction → vehicle_count.

        Returns:
            Dict of direction → DensityReading.
        """
        results = {}
        for direction, count in counts.items():
            results[direction] = self.analyze(direction, count)
        return results

    # ── Congestion Scoring ────────────────────────────────────────────────────

    def _compute_congestion_score(
        self,
        direction: str,
        current_count: int,
        smoothed_count: float,
        history: List[int],
    ) -> float:
        """
        Compute a 0–100 congestion score.

        Score components:
          - Base score: proportional to smoothed count (max reference = 30 vehicles)
          - Trend bonus: increasing trend adds up to 15 points
          - Spike penalty: sudden spike above smoothed adds up to 10 points

        Args:
            direction:     Camera direction label.
            current_count: Raw vehicle count this frame.
            smoothed_count: Moving-average count.
            history:       Recent count history list.

        Returns:
            Float score between 0.0 and 100.0.
        """
        MAX_REFERENCE = 30.0   # Vehicle count that maps to 100% base score

        # Base: smoothed count normalised to 0–85
        base = min(smoothed_count / MAX_REFERENCE, 1.0) * 85.0

        # Trend: compare latest half of window to earlier half
        trend_bonus = 0.0
        if len(history) >= 4:
            mid = len(history) // 2
            early_avg = sum(history[:mid]) / mid
            late_avg  = sum(history[mid:]) / (len(history) - mid)
            delta = late_avg - early_avg
            trend_bonus = min(max(delta / MAX_REFERENCE, 0.0), 1.0) * 15.0

        # Spike: current count well above smoothed
        spike_penalty = 0.0
        if smoothed_count > 0:
            spike_ratio = (current_count - smoothed_count) / MAX_REFERENCE
            spike_penalty = min(max(spike_ratio, 0.0), 1.0) * 10.0

        score = base + trend_bonus + spike_penalty
        return round(min(score, 100.0), 2)

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def get_intersection_snapshot(self) -> IntersectionDensity:
        """
        Return a combined snapshot of all four directions.

        Returns:
            IntersectionDensity with per-direction readings and aggregates.
        """
        snapshot = IntersectionDensity(timestamp=timestamp())
        for direction, reading in self._latest.items():
            if reading is not None:
                snapshot.readings[direction] = reading
        return snapshot

    def get_latest(self, direction: str) -> Optional[DensityReading]:
        """Return the most recent DensityReading for a direction."""
        return self._latest.get(direction)

    def get_all_latest(self) -> Dict[str, Optional[DensityReading]]:
        """Return latest readings for all directions."""
        return dict(self._latest)

    # ── History & Trend ───────────────────────────────────────────────────────

    def get_history(
        self,
        direction: str,
        last_n: Optional[int] = None,
    ) -> List[DensityReading]:
        """
        Return reading history for a direction.

        Args:
            direction: Camera direction.
            last_n:    Return only last N readings (None = all).

        Returns:
            List of DensityReading objects.
        """
        history = self._reading_history.get(direction, [])
        if last_n is not None:
            return history[-last_n:]
        return history

    def get_trend(self, direction: str, window: int = 10) -> str:
        """
        Return trend string for a direction.

        Returns:
            "INCREASING" | "DECREASING" | "STABLE"
        """
        history = self._reading_history.get(direction, [])
        if len(history) < 3:
            return "STABLE"

        recent = [r.vehicle_count for r in history[-window:]]
        if len(recent) < 2:
            return "STABLE"

        # Linear regression slope
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])

        if slope > 0.5:
            return "INCREASING"
        elif slope < -0.5:
            return "DECREASING"
        return "STABLE"

    def get_peak_count(self, direction: str) -> int:
        """Return the highest vehicle count ever recorded for a direction."""
        history = self._reading_history.get(direction, [])
        if not history:
            return 0
        return max(r.vehicle_count for r in history)

    def get_average_count(self, direction: str) -> float:
        """Return the overall average vehicle count for a direction."""
        history = self._reading_history.get(direction, [])
        if not history:
            return 0.0
        return sum(r.vehicle_count for r in history) / len(history)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary_report(self) -> Dict:
        """
        Generate a summary statistics report for all directions.

        Returns:
            Dict with per-direction stats and intersection totals.
        """
        report = {
            "generated_at": timestamp(),
            "directions": {},
        }

        for direction in self.config.video.directions:
            history = self._reading_history.get(direction, [])
            latest  = self._latest.get(direction)
            report["directions"][direction] = {
                "total_readings":     len(history),
                "latest_count":       latest.vehicle_count if latest else 0,
                "latest_level":       latest.density_level if latest else "N/A",
                "latest_score":       latest.congestion_score if latest else 0,
                "peak_count":         self.get_peak_count(direction),
                "average_count":      round(self.get_average_count(direction), 2),
                "trend":              self.get_trend(direction),
                "recommended_green":  latest.recommended_green if latest else 20,
            }

        snapshot = self.get_intersection_snapshot()
        report["intersection"] = {
            "total_vehicles":        snapshot.total_vehicles,
            "average_congestion":    round(snapshot.average_congestion, 2),
            "overall_level":         snapshot.overall_level,
            "busiest_direction":     snapshot.highest_density_direction,
        }

        return report

    def reset(self) -> None:
        """Clear all history and reset to initial state."""
        for direction in self.config.video.directions:
            self._count_history[direction].clear()
            self._reading_history[direction] = []
            self._latest[direction] = None
        logger.info("DensityAnalyzer reset.")
