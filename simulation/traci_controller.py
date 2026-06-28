"""
simulation/traci_controller.py — SUMO TraCI Session Manager
=============================================================
Launches SUMO, connects via TraCI, controls traffic lights
dynamically, and collects simulation metrics.

SUMO phase index map (matches signals.add.xml):
  0  → NORTH  green
  1  → NORTH  yellow
  2  → ALL_RED
  3  → SOUTH  green
  4  → SOUTH  yellow
  5  → ALL_RED
  6  → EAST   green
  7  → EAST   yellow
  8  → ALL_RED
  9  → WEST   green
  10 → WEST   yellow
  11 → ALL_RED
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from config import Config


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimMetrics:
    """Metrics collected from SUMO at one simulation step."""
    step: int
    timestamp: float
    direction: str

    # Per-lane metrics
    waiting_time: float = 0.0       # Average waiting time (seconds)
    queue_length: int   = 0         # Number of halted vehicles
    mean_speed: float   = 0.0       # Average vehicle speed (m/s)
    vehicle_count: int  = 0         # Vehicles currently on approach lane

    # Intersection-wide
    throughput: int     = 0         # Vehicles that have left the intersection
    congestion_score: float = 0.0   # 0–100 composite

    def to_dict(self) -> Dict:
        return {
            "step":             self.step,
            "timestamp":        round(self.timestamp, 2),
            "direction":        self.direction,
            "waiting_time":     round(self.waiting_time, 3),
            "queue_length":     self.queue_length,
            "mean_speed":       round(self.mean_speed, 3),
            "vehicle_count":    self.vehicle_count,
            "throughput":       self.throughput,
            "congestion_score": round(self.congestion_score, 2),
        }


@dataclass
class SimSnapshot:
    """Complete simulation snapshot across all directions."""
    step: int
    sim_time: float
    metrics: Dict[str, SimMetrics] = field(default_factory=dict)
    active_phase: int = 0
    tl_state: str = ""

    @property
    def total_waiting(self) -> float:
        return sum(m.waiting_time for m in self.metrics.values())

    @property
    def total_queue(self) -> int:
        return sum(m.queue_length for m in self.metrics.values())

    @property
    def total_throughput(self) -> int:
        return sum(m.throughput for m in self.metrics.values())

    def to_dict(self) -> Dict:
        return {
            "step":           self.step,
            "sim_time":       round(self.sim_time, 2),
            "active_phase":   self.active_phase,
            "tl_state":       self.tl_state,
            "total_waiting":  round(self.total_waiting, 3),
            "total_queue":    self.total_queue,
            "total_throughput": self.total_throughput,
            "directions":     {d: m.to_dict() for d, m in self.metrics.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE MAP
# ═══════════════════════════════════════════════════════════════════════════════

# Maps direction → (green_phase_index, yellow_phase_index, all_red_index)
DIRECTION_PHASES: Dict[str, Tuple[int, int, int]] = {
    "north": (0,  1,  2),
    "south": (3,  4,  5),
    "east":  (6,  7,  8),
    "west":  (9, 10, 11),
}

# Maps direction → approach lane ID (from intersection.net.xml)
DIRECTION_LANES: Dict[str, str] = {
    "north": "north_in_0",
    "south": "south_in_0",
    "east":  "east_in_0",
    "west":  "west_in_0",
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRACI CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class TraCIController:
    """
    Manages a SUMO simulation session via TraCI.

    Lifecycle:
        ctrl = TraCIController()
        ctrl.start()
        for step in range(max_steps):
            ctrl.set_green("north", green_time=40)
            snapshot = ctrl.collect_metrics(step)
            ctrl.step()
        ctrl.stop()
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._sim_cfg = self.config.simulation
        self._paths   = self.config.paths

        self._traci = None          # traci module (imported lazily)
        self._sumo_process = None   # subprocess.Popen handle
        self._connected = False
        self._step_count = 0
        self._departed_vehicles = set()   # Track IDs that have departed
        self._arrived_vehicles  = set()   # Track IDs that have arrived (throughput)

        # Metrics history
        self._snapshots: List[SimSnapshot] = []

        # Current direction being served
        self._current_direction: str = "north"
        self._current_phase_idx: int = 0

    # ── SUMO / TraCI Lifecycle ────────────────────────────────────────────────

    def _import_traci(self) -> bool:
        """Lazily import traci and set SUMO_HOME."""
        try:
            sumo_home = self._sim_cfg.sumo_home
            if sumo_home and Path(sumo_home).exists():
                if sumo_home not in sys.path:
                    sys.path.append(os.path.join(sumo_home, "tools"))
            import traci as _traci
            self._traci = _traci
            logger.debug("traci imported successfully.")
            return True
        except ImportError:
            logger.error(
                "traci not found. Install SUMO and ensure SUMO_HOME is set.\n"
                "  Ubuntu: sudo apt install sumo sumo-tools\n"
                "  then:   export SUMO_HOME=/usr/share/sumo"
            )
            return False

    def _build_sumo_cmd(self) -> List[str]:
        """Build the SUMO launch command."""
        net    = str(self._paths.sumo_net)
        routes = str(self._paths.sumo_routes)
        add    = str(self._paths.sumo_signals)

        trip_out  = str(self._paths.sim_results_dir / "tripinfo.xml")
        queue_out = str(self._paths.sim_results_dir / "queue.xml")
        summ_out  = str(self._paths.sim_results_dir / "summary.xml")

        cmd = [
            self._sim_cfg.sumo_binary,
            "--net-file",        net,
            "--route-files",     routes,
            "--additional-files", add,
            "--step-length",     str(self._sim_cfg.step_length),
            "--no-warnings",     "true",
            "--no-step-log",     "true",
            "--tripinfo-output", trip_out,
            "--queue-output",    queue_out,
            "--summary-output",  summ_out,
            "--remote-port",     str(self._sim_cfg.traci_port),
            "--begin",           "0",
            "--end",             str(self._sim_cfg.max_steps),
        ]
        return cmd

    def start(self) -> bool:
        """
        Launch SUMO as a subprocess and connect via TraCI.

        Returns:
            True if successfully connected.
        """
        if not self._import_traci():
            return False

        cmd = self._build_sumo_cmd()
        logger.info(f"Launching SUMO: {' '.join(cmd)}")

        try:
            self._sumo_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(1.5)   # Allow SUMO to start listening

            self._traci.init(
                port=self._sim_cfg.traci_port,
                host=self._sim_cfg.traci_host,
            )
            self._connected = True
            logger.success("✅ TraCI connected to SUMO.")
            return True

        except FileNotFoundError:
            logger.error(
                f"SUMO binary '{self._sim_cfg.sumo_binary}' not found. "
                "Install SUMO or set sumo_binary='sumo-gui' in config."
            )
            return False
        except ConnectionRefusedError:
            logger.error(
                "TraCI connection refused. SUMO may not have started. "
                "Check SUMO installation and port availability."
            )
            return False
        except Exception as exc:
            logger.error(f"TraCI start failed: {exc}")
            return False

    def stop(self) -> None:
        """Close TraCI connection and terminate SUMO."""
        if self._connected and self._traci:
            try:
                self._traci.close()
            except Exception:
                pass
            self._connected = False

        if self._sumo_process:
            self._sumo_process.terminate()
            try:
                self._sumo_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._sumo_process.kill()
            self._sumo_process = None

        logger.info(f"SUMO stopped after {self._step_count} steps.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ── Simulation Step ───────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance simulation by one step (step_length seconds)."""
        if not self._connected:
            return
        try:
            self._traci.simulationStep()
            self._step_count += 1

            # Track departed / arrived for throughput
            departed = self._traci.simulation.getDepartedIDList()
            arrived  = self._traci.simulation.getArrivedIDList()
            self._departed_vehicles.update(departed)
            self._arrived_vehicles.update(arrived)
        except Exception as exc:
            logger.warning(f"Simulation step error at step {self._step_count}: {exc}")

    @property
    def sim_time(self) -> float:
        """Current simulation time in seconds."""
        if not self._connected:
            return 0.0
        try:
            return self._traci.simulation.getTime()
        except Exception:
            return float(self._step_count)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Traffic Light Control ─────────────────────────────────────────────────

    def set_green(self, direction: str, green_time: int) -> None:
        """
        Set the given direction to GREEN for green_time seconds in SUMO.

        Args:
            direction:  north / south / east / west
            green_time: Duration of green phase in seconds.
        """
        if not self._connected:
            return

        phases = DIRECTION_PHASES.get(direction)
        if phases is None:
            logger.warning(f"Unknown direction: {direction}")
            return

        green_phase = phases[0]
        tl_id = self._sim_cfg.tl_id

        try:
            self._traci.trafficlight.setPhase(tl_id, green_phase)
            self._traci.trafficlight.setPhaseDuration(tl_id, green_time)
            self._current_direction = direction
            self._current_phase_idx = green_phase
            logger.debug(
                f"[TraCI] Set {direction} GREEN for {green_time}s "
                f"(phase={green_phase})"
            )
        except Exception as exc:
            logger.error(f"[TraCI] set_green failed: {exc}")

    def set_yellow(self, direction: str) -> None:
        """Set the given direction to YELLOW."""
        if not self._connected:
            return
        phases = DIRECTION_PHASES.get(direction)
        if phases is None:
            return
        try:
            self._traci.trafficlight.setPhase(self._sim_cfg.tl_id, phases[1])
            self._traci.trafficlight.setPhaseDuration(
                self._sim_cfg.tl_id, self.config.signal.yellow_time
            )
        except Exception as exc:
            logger.error(f"[TraCI] set_yellow failed: {exc}")

    def set_all_red(self) -> None:
        """Set all directions to RED (clearance phase)."""
        if not self._connected:
            return
        try:
            # Use phase 2 (all_red_1) as representative all-red state
            self._traci.trafficlight.setPhase(self._sim_cfg.tl_id, 2)
            self._traci.trafficlight.setPhaseDuration(
                self._sim_cfg.tl_id, self.config.signal.all_red_time
            )
        except Exception as exc:
            logger.error(f"[TraCI] set_all_red failed: {exc}")

    def get_tl_state(self) -> str:
        """Return current traffic light state string (e.g. 'Grrr')."""
        if not self._connected:
            return "rrrr"
        try:
            return self._traci.trafficlight.getRedYellowGreenState(
                self._sim_cfg.tl_id
            )
        except Exception:
            return "rrrr"

    # ── Metrics Collection ────────────────────────────────────────────────────

    def collect_metrics(self, step: Optional[int] = None) -> SimSnapshot:
        """
        Collect per-direction and intersection-wide metrics from SUMO.

        Args:
            step: Simulation step number (uses internal counter if None).

        Returns:
            SimSnapshot with metrics for all directions.
        """
        step = step if step is not None else self._step_count
        snapshot = SimSnapshot(
            step=step,
            sim_time=self.sim_time,
            active_phase=self._current_phase_idx,
            tl_state=self.get_tl_state(),
        )

        for direction, lane_id in DIRECTION_LANES.items():
            metrics = self._collect_lane_metrics(step, direction, lane_id)
            metrics.throughput = len(self._arrived_vehicles)
            snapshot.metrics[direction] = metrics

        self._snapshots.append(snapshot)
        return snapshot

    def _collect_lane_metrics(
        self,
        step: int,
        direction: str,
        lane_id: str,
    ) -> SimMetrics:
        """Collect metrics for a single approach lane."""
        metrics = SimMetrics(
            step=step,
            timestamp=self.sim_time,
            direction=direction,
        )

        if not self._connected:
            return metrics

        try:
            lane = self._traci.lane
            metrics.waiting_time  = lane.getWaitingTime(lane_id)
            metrics.vehicle_count = lane.getLastStepVehicleNumber(lane_id)
            metrics.queue_length  = lane.getLastStepHaltingNumber(lane_id)
            metrics.mean_speed    = lane.getLastStepMeanSpeed(lane_id)

            # Congestion score: weighted sum of queue and waiting time
            max_queue = 20.0
            max_wait  = 120.0
            q_score = min(metrics.queue_length / max_queue, 1.0) * 60.0
            w_score = min(metrics.waiting_time  / max_wait,  1.0) * 40.0
            metrics.congestion_score = round(q_score + w_score, 2)

        except Exception as exc:
            logger.debug(f"[TraCI] Metrics collection error for {direction}: {exc}")

        return metrics

    # ── Run Full Simulation ───────────────────────────────────────────────────

    def run_simulation(
        self,
        signal_states_callback=None,
        metrics_callback=None,
    ) -> List[SimSnapshot]:
        """
        Run a full simulation loop with optional callbacks.

        Args:
            signal_states_callback: Called each step with current SignalController
                                    states to sync SUMO signals.
                                    Signature: callback(step) → Dict[str, SignalState]
            metrics_callback:       Called every metrics_interval steps
                                    with the latest SimSnapshot.
                                    Signature: callback(snapshot)

        Returns:
            List of all SimSnapshot objects collected.
        """
        if not self._connected:
            logger.error("Not connected to SUMO. Call start() first.")
            return []

        logger.info(
            f"Starting simulation run: max_steps={self._sim_cfg.max_steps}, "
            f"warmup={self._sim_cfg.warmup_steps}"
        )

        cycle_order = self.config.signal.cycle_order
        dir_idx = 0
        phase_timer = 0.0
        current_sub_phase = "green"    # green | yellow | all_red

        # Default green time when no external controller connected
        default_green = self.config.signal.green_times["LOW"]
        current_green = default_green

        for step in range(self._sim_cfg.max_steps):
            # Sync signal state with external controller (if provided)
            if signal_states_callback:
                states = signal_states_callback(step)
                if states:
                    for direction, state in states.items():
                        if state.is_green:
                            self.set_green(direction, int(state.remaining_seconds))

            # Collect metrics at interval (after warmup)
            if (step % self._sim_cfg.metrics_interval == 0 and
                    step >= self._sim_cfg.warmup_steps):
                snapshot = self.collect_metrics(step)
                if metrics_callback:
                    metrics_callback(snapshot)

            self.step()

            # Check if simulation ended
            if not self._connected:
                break

            try:
                if self._traci.simulation.getMinExpectedNumber() == 0:
                    logger.info(f"All vehicles departed/arrived at step {step}. Ending.")
                    break
            except Exception:
                break

        logger.success(
            f"Simulation complete. "
            f"Steps={self._step_count} | "
            f"Throughput={len(self._arrived_vehicles)} vehicles"
        )
        return self._snapshots

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_snapshots(self) -> List[SimSnapshot]:
        """Return all collected simulation snapshots."""
        return list(self._snapshots)

    def get_throughput(self) -> int:
        """Return total vehicles that completed their trip."""
        return len(self._arrived_vehicles)

    def get_summary(self) -> Dict:
        """Return high-level simulation summary."""
        if not self._snapshots:
            return {"status": "no data"}

        all_waiting = [s.total_waiting for s in self._snapshots]
        all_queues  = [s.total_queue   for s in self._snapshots]

        return {
            "total_steps":       self._step_count,
            "total_sim_time":    self.sim_time,
            "throughput":        self.get_throughput(),
            "avg_waiting_time":  round(sum(all_waiting) / len(all_waiting), 3),
            "max_waiting_time":  round(max(all_waiting), 3),
            "avg_queue_length":  round(sum(all_queues)  / len(all_queues),  2),
            "max_queue_length":  max(all_queues),
            "snapshots_count":   len(self._snapshots),
        }
