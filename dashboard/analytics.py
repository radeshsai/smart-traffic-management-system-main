"""
dashboard/analytics.py — Data Aggregation for Dashboard
=========================================================
Reads from the SQLite database and in-memory state to
produce data structures ready for dashboard rendering.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from loguru import logger

from config import Config
from src.database import DatabaseManager


class DashboardAnalytics:
    """
    Aggregates and prepares data for the Streamlit dashboard.

    Reads from:
      - DatabaseManager (historical records)
      - In-memory state (optional live overrides)
    """

    def __init__(self, db: DatabaseManager, config: Config = None):
        self.db     = db
        self.config = config or Config()
        self._live_state: Dict[str, Any] = {}

    # ── Live State Injection ──────────────────────────────────────────────────

    def update_live_state(self, state: Dict[str, Any]) -> None:
        """Inject live state from the running pipeline."""
        self._live_state.update(state)

    # ── Header Stats ──────────────────────────────────────────────────────────

    def get_header_stats(self) -> Dict[str, Any]:
        """Return top-level KPI numbers for the dashboard header."""
        try:
            db_stats = self.db.get_total_stats()
        except Exception as e:
            logger.warning(f"DB header stats error: {e}")
            db_stats = {
                "total_detections": 0,
                "total_vehicles":   0,
                "signal_cycles":    0,
                "sim_throughput":   0,
            }
        return db_stats

    # ── Current Counts ────────────────────────────────────────────────────────

    def get_current_counts(self) -> Dict[str, int]:
        """
        Return current vehicle count per direction.
        Prefers live state, falls back to latest DB record.
        """
        live = self._live_state.get("current_counts")
        if live:
            return live

        try:
            rows = self.db.get_latest_counts()
            return {r["direction"]: r["current_count"] for r in rows}
        except Exception as e:
            logger.warning(f"get_current_counts DB error: {e}")
            return {d: 0 for d in self.config.video.directions}

    # ── Density ───────────────────────────────────────────────────────────────

    def get_current_density(self) -> Dict[str, Dict]:
        """Return latest density info per direction."""
        live = self._live_state.get("density")
        if live:
            return live

        try:
            rows = self.db.get_latest_density()
            return {
                r["direction"]: {
                    "level":             r.get("density_level", "LOW"),
                    "score":             r.get("congestion_score", 0),
                    "vehicle_count":     r.get("vehicle_count", 0),
                    "recommended_green": r.get("recommended_green", 20),
                }
                for r in rows
            }
        except Exception as e:
            logger.warning(f"get_current_density DB error: {e}")
            return {
                d: {"level": "LOW", "score": 0, "vehicle_count": 0, "recommended_green": 20}
                for d in self.config.video.directions
            }

    # ── Signal States ─────────────────────────────────────────────────────────

    def get_signal_states(self) -> Dict[str, Dict]:
        """Return latest signal state per direction — reads last 2 seconds of data."""
        live = self._live_state.get("signal_states")
        if live:
            return live

        try:
            # Get latest row per direction from last 5 seconds
            # to catch fast-moving YELLOW phases
            rows = self.db.get_latest_signal_states()
            result = {}
            for r in rows:
                d = r["direction"]
                result[d] = {
                    "phase":             r.get("phase", "RED"),
                    "remaining_seconds": r.get("remaining_seconds", 0),
                    "allocated_green":   r.get("allocated_green", 0),
                    "cycle_number":      r.get("cycle_number", 0),
                    "density_level":     r.get("density_level", "LOW"),
                }
            return result
        except Exception as e:
            logger.warning(f"get_signal_states DB error: {e}")
            return {
                d: {"phase": "RED", "remaining_seconds": 0,
                    "allocated_green": 0, "cycle_number": 0, "density_level": "LOW"}
                for d in self.config.video.directions
            }

    # ── Density Time Series ───────────────────────────────────────────────────

    def get_density_timeseries_df(
        self,
        direction: Optional[str] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Return density history as a DataFrame for charting."""
        try:
            rows = self.db.get_density_timeseries(direction=direction, limit=limit)
            if not rows:
                return pd.DataFrame(columns=["created_at","direction","vehicle_count","congestion_score"])
            df = pd.DataFrame(rows)
            df["created_at"] = pd.to_datetime(df["created_at"])
            return df
        except Exception as e:
            logger.warning(f"get_density_timeseries_df error: {e}")
            return pd.DataFrame()

    # ── Class Distribution ────────────────────────────────────────────────────

    def get_class_distribution(self) -> Dict[str, int]:
        """Return total vehicle count by class across all directions."""
        try:
            rows = self.db.get_latest_counts()
            totals: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
            for r in rows:
                totals["car"]        += r.get("count_car", 0)
                totals["motorcycle"] += r.get("count_motorcycle", 0)
                totals["bus"]        += r.get("count_bus", 0)
                totals["truck"]      += r.get("count_truck", 0)
            return totals
        except Exception as e:
            logger.warning(f"get_class_distribution error: {e}")
            return {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

    # ── Simulation Metrics ────────────────────────────────────────────────────

    def get_sim_metrics(self) -> List[Dict]:
        """Return aggregated simulation metrics per direction."""
        live = self._live_state.get("sim_metrics")
        if live:
            return live

        try:
            return self.db.get_sim_metrics_summary()
        except Exception as e:
            logger.warning(f"get_sim_metrics error: {e}")
            return []

    def get_sim_summary(self) -> Dict:
        """Return latest simulation summary from live state."""
        return self._live_state.get("sim_result", {})

    # ── Cycle Logs ────────────────────────────────────────────────────────────

    def get_cycle_logs(self) -> List[Dict]:
        """Return signal cycle log records."""
        live = self._live_state.get("cycle_logs", [])
        if live:
            return [log.to_dict() if hasattr(log, "to_dict") else log for log in live]

        try:
            rows = self.db.get_signal_cycle_summary()
            return rows
        except Exception as e:
            logger.warning(f"get_cycle_logs error: {e}")
            return []

    # ── Congestion History ────────────────────────────────────────────────────

    def get_congestion_history(self, limit: int = 300) -> List[Dict]:
        """Return congestion score history for heatmap."""
        try:
            rows = self.db.get_density_timeseries(limit=limit)
            return rows
        except Exception as e:
            logger.warning(f"get_congestion_history error: {e}")
            return []

    # ── Active Signal Info ────────────────────────────────────────────────────

    def get_active_signal_info(self) -> Dict:
        """Return info about the currently active direction (any phase)."""
        live = self._live_state.get("active_signal")
        if live:
            return live

        states = self.get_signal_states()

        # Priority: GREEN first, then YELLOW, then ALL_RED
        for priority_phase in ["GREEN", "YELLOW", "ALL_RED"]:
            for direction, state in states.items():
                if state.get("phase") == priority_phase:
                    return {
                        "direction":       direction,
                        "phase":           priority_phase,
                        "remaining":       state.get("remaining_seconds", 0),
                        "allocated_green": state.get("allocated_green", 0),
                        "density_level":   state.get("density_level", "LOW"),
                    }
        return {
            "direction":       "N/A",
            "phase":           "RED",
            "remaining":       0,
            "allocated_green": 0,
            "density_level":   "LOW",
        }