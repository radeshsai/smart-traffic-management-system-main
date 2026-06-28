"""
src/signal_controller.py
Signal sequence: GREEN → YELLOW(3s) → NEXT GREEN
No ALL_RED phase. No delay between transitions.
YELLOW_DURATION = 3 seconds (fixed constant everywhere).
"""
import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from loguru import logger

from config import Config
from src.density_analyzer import DensityReading

YELLOW_DURATION = 3          # Fixed constant — never changes
MAX_WAIT_CYCLES = 4          # Starvation guard


class SignalPhase(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    RED    = "RED"


@dataclass
class SignalState:
    direction: str
    phase: SignalPhase = SignalPhase.RED
    remaining_seconds: float = 0.0
    allocated_green: int = 0
    cycle_number: int = 0
    total_green_time: float = 0.0
    total_wait_time: float = 0.0
    last_density_level: str = "LOW"

    @property
    def is_green(self): return self.phase == SignalPhase.GREEN
    @property
    def is_red(self):   return self.phase == SignalPhase.RED

    def to_dict(self):
        return {
            "direction":         self.direction,
            "phase":             self.phase.value,
            "remaining_seconds": round(self.remaining_seconds, 1),
            "allocated_green":   self.allocated_green,
            "cycle_number":      self.cycle_number,
            "total_green_time":  round(self.total_green_time, 1),
            "total_wait_time":   round(self.total_wait_time, 1),
            "last_density":      self.last_density_level,
        }


@dataclass
class CycleLog:
    direction: str
    cycle_number: int
    density_level: str
    vehicle_count: int
    green_time_allocated: int
    green_time_used: float
    timestamp: str

    def to_dict(self):
        return {
            "direction":            self.direction,
            "cycle_number":         self.cycle_number,
            "density_level":        self.density_level,
            "vehicle_count":        self.vehicle_count,
            "green_time_allocated": self.green_time_allocated,
            "green_time_used":      round(self.green_time_used, 2),
            "timestamp":            self.timestamp,
        }


class SignalController:
    """
    2-phase signal controller:
        GREEN(dynamic) → YELLOW(3s) → [next dir] GREEN

    No ALL_RED. No delay. Immediate switch after YELLOW.
    """

    def __init__(self, config: Config = None):
        self.config   = config or Config()
        self._sig_cfg = self.config.signal
        self._dirs: List[str] = list(self._sig_cfg.cycle_order)

        self._states: Dict[str, SignalState] = {
            d: SignalState(direction=d) for d in self._dirs
        }
        self._density: Dict[str, Optional[DensityReading]] = {
            d: None for d in self._dirs
        }

        # State machine — only 2 phases: GREEN and YELLOW
        self._active:      str         = self._dirs[0]
        self._phase:       SignalPhase = SignalPhase.GREEN
        self._remaining:   float       = float(self._sig_cfg.green_times["LOW"])
        self._cycle_count: int         = 0
        self._skip_counts: Dict[str,int] = {d: 0 for d in self._dirs}

        self._cycle_logs: List[CycleLog] = []
        self._lock     = threading.Lock()
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._on_phase_change: Optional[Callable] = None

        # Set initial GREEN
        self._apply_state()
        logger.info(
            f"SignalController ready. "
            f"Sequence: GREEN → YELLOW({YELLOW_DURATION}s) → NEXT GREEN"
        )

    # ── State sync ────────────────────────────────────────────────────────────

    def _apply_state(self):
        """Push internal state to SignalState objects."""
        for d in self._dirs:
            if d == self._active:
                self._states[d].phase             = self._phase
                self._states[d].remaining_seconds = max(self._remaining, 0.0)
            else:
                self._states[d].phase             = SignalPhase.RED
                self._states[d].remaining_seconds = 0.0

    # ── Density ───────────────────────────────────────────────────────────────

    def update_density(self, direction: str, reading: DensityReading) -> None:
        with self._lock:
            self._density[direction] = reading
            self._states[direction].last_density_level = reading.density_level

    def update_all_densities(self, readings: Dict[str, DensityReading]) -> None:
        for d, r in readings.items():
            self.update_density(d, r)

    # ── Tick ──────────────────────────────────────────────────────────────────

    def tick(self, delta_seconds: float = 1.0) -> Dict[str, SignalState]:
        with self._lock:
            self._remaining -= delta_seconds

            # Accumulate wait time for RED directions
            for d in self._dirs:
                if d != self._active:
                    self._states[d].total_wait_time += delta_seconds

            # Transition when timer expires
            if self._remaining <= 0:
                if self._phase == SignalPhase.GREEN:
                    self._start_yellow()
                elif self._phase == SignalPhase.YELLOW:
                    self._start_next_green()   # Immediate — no ALL_RED

            self._apply_state()

        return dict(self._states)

    # ── Transitions ───────────────────────────────────────────────────────────

    def _start_yellow(self):
        """GREEN → YELLOW (same direction, fixed 3s)."""
        self._states[self._active].total_green_time += self._states[self._active].allocated_green
        self._phase     = SignalPhase.YELLOW
        self._remaining = float(YELLOW_DURATION)   # Always exactly 3s
        logger.info(f"[Signal] {self._active.upper()} YELLOW {YELLOW_DURATION}s")

    def _start_next_green(self):
        """YELLOW ends → immediately start next direction GREEN. No delay."""
        cur = self._active
        dn  = self._density.get(cur)

        # Log completed cycle
        self._states[cur].cycle_number += 1
        self._cycle_count += 1
        self._cycle_logs.append(CycleLog(
            direction=cur,
            cycle_number=self._states[cur].cycle_number,
            density_level=dn.density_level if dn else "LOW",
            vehicle_count=dn.vehicle_count if dn else 0,
            green_time_allocated=self._states[cur].allocated_green,
            green_time_used=float(self._states[cur].allocated_green),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        ))

        # Set current dir to RED
        self._states[cur].phase             = SignalPhase.RED
        self._states[cur].remaining_seconds = 0.0

        # Pick next direction (fair adaptive scheduler)
        next_dir = self._pick_next(cur)

        # Update skip counters
        for d in self._dirs:
            if d == next_dir:
                self._skip_counts[d] = 0
            elif d != cur:
                self._skip_counts[d] += 1

        # Determine green time
        dn_next = self._density.get(next_dir)
        if dn_next:
            gt    = dn_next.recommended_green
            level = dn_next.density_level
        else:
            gt    = self._sig_cfg.green_times["LOW"]
            level = "LOW"
        gt = max(self._sig_cfg.min_green_time, min(gt, self._sig_cfg.max_green_time))

        # Immediately activate next direction GREEN
        self._active    = next_dir
        self._phase     = SignalPhase.GREEN
        self._remaining = float(gt)

        self._states[next_dir].allocated_green    = gt
        self._states[next_dir].last_density_level = level

        logger.info(
            f"[Signal] {next_dir.upper()} GREEN {gt}s "
            f"(density={level}, skips={dict(self._skip_counts)})"
        )
        if self._on_phase_change:
            self._on_phase_change(next_dir, SignalPhase.GREEN, gt)

    def _pick_next(self, exclude: str) -> str:
        """Fair adaptive scheduler with starvation guard."""
        candidates = [d for d in self._dirs if d != exclude]

        # Starvation guard
        starved = [d for d in candidates if self._skip_counts[d] >= MAX_WAIT_CYCLES]
        if starved:
            chosen = max(starved, key=lambda d: (
                self._density[d].congestion_score if self._density.get(d) else 0
            ))
            logger.info(f"[Scheduler] Starvation guard → {chosen.upper()}")
            return chosen

        def priority(d):
            dn   = self._density.get(d)
            cong = dn.congestion_score if dn else 0
            wait = self._skip_counts[d]
            return cong * 0.7 + (wait / MAX_WAIT_CYCLES) * 0.3 * 100

        return max(candidates, key=priority)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._is_running = True
        logger.info("SignalController started.")

    def start_threaded(self, tick_interval: float = 0.25):
        self._is_running = True
        self._thread = threading.Thread(
            target=self._run_loop, args=(tick_interval,),
            daemon=True, name="SignalThread")
        self._thread.start()

    def _run_loop(self, interval: float):
        """Precise wall-clock based loop — never drifts."""
        last = time.perf_counter()
        while self._is_running:
            now     = time.perf_counter()
            elapsed = now - last
            last    = now
            self.tick(delta_seconds=elapsed)
            # Sleep exactly interval seconds
            next_tick = last + interval
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

    def stop(self):
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("SignalController stopped.")

    def get_state(self, direction: str) -> Optional[SignalState]:
        return self._states.get(direction)

    def get_all_states(self) -> Dict[str, SignalState]:
        return dict(self._states)

    def get_active_direction(self) -> str:
        return self._active

    def get_active_phase(self) -> SignalPhase:
        return self._phase

    def get_cycle_count(self) -> int:
        return self._cycle_count

    def get_cycle_logs(self) -> List[CycleLog]:
        return list(self._cycle_logs)

    def get_summary(self) -> Dict:
        return {
            "active_direction": self._active,
            "active_phase":     self._phase.value,
            "phase_remaining":  round(self._remaining, 1),
            "cycle_count":      self._cycle_count,
            "skip_counts":      dict(self._skip_counts),
            "states":           {d: s.to_dict() for d, s in self._states.items()},
        }

    def on_phase_change(self, callback: Callable):
        self._on_phase_change = callback
