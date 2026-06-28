"""
src/simulation_engine.py — SUMO Simulation Engine Bridge
==========================================================
Integrates the TraCI controller with the signal controller
and density analyzer to run a fully coupled simulation.

If SUMO is not installed, falls back to a pure-Python
statistical simulator that generates realistic metrics.
"""

import time
import math
import random
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger

from config import Config
from src.signal_controller import SignalController, SignalState, SignalPhase
from src.density_analyzer import DensityAnalyzer, DensityReading
from src.database import DatabaseManager


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK: STATISTICAL SIMULATOR (No SUMO needed)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FallbackMetrics:
    """Simulated metrics when SUMO is unavailable."""
    direction: str
    step: int
    waiting_time: float
    queue_length: int
    mean_speed: float
    vehicle_count: int
    throughput: int
    congestion_score: float

    def to_dict(self) -> Dict:
        return {
            "direction":        self.direction,
            "step":             self.step,
            "waiting_time":     round(self.waiting_time, 3),
            "queue_length":     self.queue_length,
            "mean_speed":       round(self.mean_speed, 3),
            "vehicle_count":    self.vehicle_count,
            "throughput":       self.throughput,
            "congestion_score": round(self.congestion_score, 2),
        }


class StatisticalSimulator:
    """
    Pure-Python traffic simulator that generates realistic
    metrics without requiring SUMO installation.

    Models:
      - Queue buildup during RED phase
      - Queue drainage during GREEN phase
      - Waiting time as function of queue length
      - Speed as inverse function of density
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._step = 0
        self._throughput: Dict[str, int] = {d: 0 for d in config.video.directions}
        self._queues:     Dict[str, float] = {d: 0.0 for d in config.video.directions}
        self._waited:     Dict[str, float] = {d: 0.0 for d in config.video.directions}
        self._rng = random.Random(42)

    def update(
        self,
        signal_states: Dict[str, SignalState],
        density_readings: Dict[str, Optional[DensityReading]],
    ) -> Dict[str, FallbackMetrics]:
        """
        Advance simulation one step and return metrics.

        Args:
            signal_states:    Current signal phase per direction.
            density_readings: Latest density per direction.

        Returns:
            Dict of direction → FallbackMetrics.
        """
        self._step += 1
        metrics = {}

        for direction in self.config.video.directions:
            state   = signal_states.get(direction)
            reading = density_readings.get(direction)

            # Arrival rate from density
            count = reading.vehicle_count if reading else 3
            arrival_rate = count / 30.0    # vehicles per second

            is_green = state is not None and state.is_green

            # Queue model: vehicles arrive each step, drain when green
            if is_green:
                drain_rate = min(self._queues[direction], 1.8)
                self._queues[direction] = max(
                    0.0, self._queues[direction] - drain_rate + arrival_rate
                )
                self._throughput[direction] += int(drain_rate)
                speed = self._rng.uniform(8.0, 13.0)   # Moving traffic
                # Reset wait when green clears the queue
                self._waited[direction] = max(0.0, self._waited[direction] - 4.0)
            else:
                self._queues[direction] += arrival_rate
                self._queues[direction] = min(self._queues[direction], 20.0)
                speed = self._rng.uniform(0.0, 1.5)    # Nearly stopped
                # Waiting time grows with queue — capped at 90s (realistic)
                if int(self._queues[direction]) > 0:
                    self._waited[direction] = min(
                        self._waited[direction] + 0.8,
                        90.0
                    )

            queue_len = int(self._queues[direction])
            wait_time = max(0.0, self._waited[direction] + self._rng.uniform(-1, 1))

            # Congestion score 0-100
            max_q = 20.0; max_w = 90.0
            score = min((queue_len / max_q) * 60 + (wait_time / max_w) * 40, 100.0)
            score += self._rng.uniform(-3, 3)
            score = max(0.0, min(100.0, score))

            metrics[direction] = FallbackMetrics(
                direction=direction,
                step=self._step,
                waiting_time=round(wait_time, 3),
                queue_length=queue_len,
                mean_speed=round(speed, 3),
                vehicle_count=count,
                throughput=self._throughput[direction],
                congestion_score=round(score, 2),
            )

        return metrics

    def get_throughput(self) -> int:
        return sum(self._throughput.values())


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE (MAIN CLASS)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """Full result from one simulation run."""
    mode: str                      # "sumo" | "statistical"
    total_steps: int
    total_sim_time: float
    throughput: int
    avg_waiting_time: float
    max_waiting_time: float
    avg_queue_length: float
    max_queue_length: int
    avg_congestion_score: float
    per_direction: Dict[str, Dict] = field(default_factory=dict)
    snapshots: List[Dict]          = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "mode":               self.mode,
            "total_steps":        self.total_steps,
            "total_sim_time":     round(self.total_sim_time, 2),
            "throughput":         self.throughput,
            "avg_waiting_time":   round(self.avg_waiting_time, 3),
            "max_waiting_time":   round(self.max_waiting_time, 3),
            "avg_queue_length":   round(self.avg_queue_length, 2),
            "max_queue_length":   self.max_queue_length,
            "avg_congestion":     round(self.avg_congestion_score, 2),
            "per_direction":      self.per_direction,
        }


class SimulationEngine:
    """
    High-level simulation engine that integrates:
      - SignalController (dynamic green times)
      - DensityAnalyzer (current congestion)
      - TraCIController (SUMO, if available) OR
      - StatisticalSimulator (fallback)

    Usage:
        engine = SimulationEngine(signal_controller, density_analyzer, db)
        result = engine.run(max_steps=1800)
    """

    def __init__(
        self,
        signal_controller: SignalController,
        density_analyzer: DensityAnalyzer,
        db: Optional[DatabaseManager] = None,
        config: Config = None,
    ):
        self.config           = config or Config()
        self.signal_controller = signal_controller
        self.density_analyzer  = density_analyzer
        self.db                = db

        self._sim_cfg = self.config.simulation
        self._use_sumo = False
        self._traci_ctrl = None
        self._stat_sim   = StatisticalSimulator(self.config)

        # Metrics accumulation
        self._all_snapshots: List[Dict] = []
        self._all_waiting:   List[float] = []
        self._all_queues:    List[int]   = []
        self._all_scores:    List[float] = []

        # Callbacks
        self._on_step: Optional[Callable] = None    # Called each step with metrics

    # ── SUMO Integration ──────────────────────────────────────────────────────

    def try_connect_sumo(self) -> bool:
        """
        Attempt to launch and connect to SUMO.
        If SUMO is not installed, falls back silently.

        Returns:
            True if SUMO connected successfully.
        """
        try:
            from simulation.traci_controller import TraCIController
            ctrl = TraCIController(self.config)
            if ctrl.start():
                self._traci_ctrl = ctrl
                self._use_sumo   = True
                logger.success("✅ SUMO simulation engine connected.")
                return True
            else:
                logger.warning("SUMO connection failed. Using statistical simulator.")
                return False
        except Exception as exc:
            logger.warning(f"SUMO not available ({exc}). Using statistical simulator.")
            return False

    # ── Main Run Loop ─────────────────────────────────────────────────────────

    def run(
        self,
        max_steps: Optional[int] = None,
        step_callback: Optional[Callable] = None,
    ) -> SimulationResult:
        """
        Run the full simulation and return results.

        Args:
            max_steps:      Override config max_steps.
            step_callback:  Called each step: callback(step, metrics_dict)

        Returns:
            SimulationResult with aggregated statistics.
        """
        max_steps = max_steps or self._sim_cfg.max_steps
        self._on_step = step_callback
        mode = "sumo" if self._use_sumo else "statistical"

        logger.info(
            f"Starting simulation engine ({mode} mode) | "
            f"max_steps={max_steps}"
        )

        start_wall = time.time()

        if self._use_sumo:
            result = self._run_sumo(max_steps)
        else:
            result = self._run_statistical(max_steps)

        elapsed = time.time() - start_wall
        logger.success(
            f"Simulation complete ({mode}) in {elapsed:.1f}s | "
            f"Steps={result.total_steps} | "
            f"Throughput={result.throughput} | "
            f"Avg wait={result.avg_waiting_time:.2f}s"
        )
        return result

    # ── Statistical Simulation ────────────────────────────────────────────────

    def _run_statistical(self, max_steps: int) -> SimulationResult:
        """Run using the built-in statistical simulator."""
        self.signal_controller.start()
        step_length = self._sim_cfg.step_length
        warmup      = self._sim_cfg.warmup_steps
        interval    = self._sim_cfg.metrics_interval

        for step in range(max_steps):
            # Advance signal controller
            self.signal_controller.tick(delta_seconds=step_length)

            # Get current states
            signal_states   = self.signal_controller.get_all_states()
            density_readings = self.density_analyzer.get_all_latest()

            # Update statistical simulator
            metrics = self._stat_sim.update(signal_states, density_readings)

            # Collect after warmup at interval
            if step >= warmup and step % interval == 0:
                snap = {
                    "step":       step,
                    "sim_time":   step * step_length,
                    "directions": {d: m.to_dict() for d, m in metrics.items()},
                }
                self._all_snapshots.append(snap)

                for m in metrics.values():
                    self._all_waiting.append(m.waiting_time)
                    self._all_queues.append(m.queue_length)
                    self._all_scores.append(m.congestion_score)

                # Write to DB
                if self.db:
                    self._write_sim_metrics_to_db(step, step * step_length, metrics)

                # External callback
                if self._on_step:
                    self._on_step(step, snap)

        self.signal_controller.stop()
        return self._build_result("statistical", max_steps)

    # ── SUMO Simulation ───────────────────────────────────────────────────────

    def _run_sumo(self, max_steps: int) -> SimulationResult:
        """Run using SUMO via TraCI."""
        ctrl       = self._traci_ctrl
        step_len   = self._sim_cfg.step_length
        warmup     = self._sim_cfg.warmup_steps
        interval   = self._sim_cfg.metrics_interval

        self.signal_controller.start()

        try:
            for step in range(max_steps):
                # Sync signal controller → SUMO
                self.signal_controller.tick(delta_seconds=step_len)
                signal_states = self.signal_controller.get_all_states()

                active_dir = self.signal_controller.get_active_direction()
                active_state = signal_states.get(active_dir)
                if active_state and active_state.is_green:
                    ctrl.set_green(active_dir, int(active_state.remaining_seconds))

                ctrl.step()

                if step >= warmup and step % interval == 0:
                    snapshot = ctrl.collect_metrics(step)
                    snap_dict = snapshot.to_dict()
                    self._all_snapshots.append(snap_dict)

                    for m in snapshot.metrics.values():
                        self._all_waiting.append(m.waiting_time)
                        self._all_queues.append(m.queue_length)
                        self._all_scores.append(m.congestion_score)

                    if self.db:
                        self.db.insert_sim_metrics(snapshot)

                    if self._on_step:
                        self._on_step(step, snap_dict)

        finally:
            ctrl.stop()
            self.signal_controller.stop()

        return self._build_result("sumo", max_steps)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _write_sim_metrics_to_db(
        self,
        step: int,
        sim_time: float,
        metrics: Dict[str, FallbackMetrics],
    ) -> None:
        """Write fallback metrics to DB in SimSnapshot-compatible format."""
        if not self.db:
            return

        # Create a lightweight snapshot-like object
        class _Snap:
            pass

        snap = _Snap()
        snap.step     = step
        snap.sim_time = sim_time
        snap.tl_state = self.signal_controller.get_active_direction()[0].upper() + "rrr"

        class _M:
            pass

        snap.metrics = {}
        for direction, fm in metrics.items():
            m = _M()
            m.waiting_time     = fm.waiting_time
            m.queue_length     = fm.queue_length
            m.mean_speed       = fm.mean_speed
            m.vehicle_count    = fm.vehicle_count
            m.throughput       = fm.throughput
            m.congestion_score = fm.congestion_score
            snap.metrics[direction] = m

        self.db.insert_sim_metrics(snap)

    def _build_result(self, mode: str, steps: int) -> SimulationResult:
        """Build SimulationResult from accumulated metrics."""
        throughput = (
            self._traci_ctrl.get_throughput()
            if self._use_sumo and self._traci_ctrl
            else self._stat_sim.get_throughput()
        )

        # Per-direction aggregates from snapshots
        per_dir: Dict[str, Dict] = {}
        for direction in self.config.video.directions:
            dir_waiting = []
            dir_queues  = []
            dir_scores  = []
            for snap in self._all_snapshots:
                dirs = snap.get("directions", {})
                m    = dirs.get(direction, {})
                if m:
                    dir_waiting.append(m.get("waiting_time", 0))
                    dir_queues.append(m.get("queue_length", 0))
                    dir_scores.append(m.get("congestion_score", 0))

            per_dir[direction] = {
                "avg_waiting":   round(sum(dir_waiting) / max(len(dir_waiting), 1), 3),
                "max_waiting":   round(max(dir_waiting, default=0), 3),
                "avg_queue":     round(sum(dir_queues)  / max(len(dir_queues),  1), 2),
                "max_queue":     max(dir_queues, default=0),
                "avg_congestion": round(sum(dir_scores) / max(len(dir_scores), 1), 2),
            }

        return SimulationResult(
            mode=mode,
            total_steps=steps,
            total_sim_time=steps * self._sim_cfg.step_length,
            throughput=throughput,
            avg_waiting_time=round(sum(self._all_waiting) / max(len(self._all_waiting), 1), 3),
            max_waiting_time=round(max(self._all_waiting, default=0), 3),
            avg_queue_length=round(sum(self._all_queues) / max(len(self._all_queues), 1), 2),
            max_queue_length=max(self._all_queues, default=0),
            avg_congestion_score=round(sum(self._all_scores) / max(len(self._all_scores), 1), 2),
            per_direction=per_dir,
            snapshots=self._all_snapshots[-100:],   # Keep last 100 for dashboard
        )

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_latest_snapshot(self) -> Optional[Dict]:
        return self._all_snapshots[-1] if self._all_snapshots else None

    def is_using_sumo(self) -> bool:
        return self._use_sumo

    def stop(self) -> None:
        """Clean shutdown of simulation engine."""
        if self._traci_ctrl:
            self._traci_ctrl.stop()
        self.signal_controller.stop()
        logger.info("SimulationEngine stopped.")