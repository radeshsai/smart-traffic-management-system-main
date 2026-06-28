"""tests/test_signal.py — Unit tests for SignalController"""
import sys
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from src.signal_controller import SignalController, SignalPhase
from src.density_analyzer import DensityAnalyzer


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def controller(cfg):
    ctrl = SignalController(cfg)
    ctrl.start()
    return ctrl


class TestSignalController:
    def test_initial_state(self, controller):
        # First direction should start on GREEN
        active = controller.get_active_direction()
        state = controller.get_state(active)
        assert state is not None
        assert state.phase == SignalPhase.GREEN

    def test_all_states_exist(self, controller, cfg):
        states = controller.get_all_states()
        for direction in cfg.signal.cycle_order:
            assert direction in states

    def test_only_one_green_at_a_time(self, controller):
        states = controller.get_all_states()
        green_count = sum(1 for s in states.values() if s.phase == SignalPhase.GREEN)
        assert green_count == 1

    def test_tick_decreases_remaining(self, controller):
        active = controller.get_active_direction()
        state_before = controller.get_state(active)
        before = state_before.remaining_seconds
        controller.tick(delta_seconds=5.0)
        state_after = controller.get_state(active)
        assert state_after.remaining_seconds <= before

    def test_density_update_changes_green_time(self, cfg):
        ctrl = SignalController(cfg)
        ctrl.start()
        analyzer = DensityAnalyzer(cfg)
        # HIGH density → 60s green
        reading = analyzer.analyze("north", 25)
        ctrl.update_density("north", reading)
        assert reading.recommended_green == 60

    def test_phase_transitions(self, cfg):
        ctrl = SignalController(cfg)
        ctrl.start()
        active = ctrl.get_active_direction()
        initial_state = ctrl.get_state(active)
        green_time = initial_state.allocated_green

        # Exhaust green phase
        ctrl.tick(delta_seconds=float(green_time) + 0.1)
        state_after_green = ctrl.get_state(active)
        assert state_after_green.phase in (SignalPhase.YELLOW, SignalPhase.ALL_RED, SignalPhase.RED)

    def test_get_summary_structure(self, controller):
        summary = controller.get_summary()
        assert "active_direction" in summary
        assert "active_phase" in summary
        assert "cycle_count" in summary
        assert "states" in summary

    def test_density_levels_map_to_correct_green(self, cfg):
        ctrl = SignalController(cfg)
        assert ctrl.config.signal.get_green_time("LOW")    == 20
        assert ctrl.config.signal.get_green_time("MEDIUM") == 40
        assert ctrl.config.signal.get_green_time("HIGH")   == 60

    def test_cycle_count_increments(self, cfg):
        ctrl = SignalController(cfg)
        ctrl.start()
        initial_count = ctrl.get_cycle_count()
        active = ctrl.get_active_direction()
        state = ctrl.get_state(active)
        green = state.allocated_green
        # GREEN → YELLOW → ALL_RED → next GREEN
        ctrl.tick(green + 0.1)                    # finish green
        ctrl.tick(cfg.signal.yellow_time + 0.1)   # finish yellow
        ctrl.tick(cfg.signal.all_red_time + 0.1)  # finish all-red
        assert ctrl.get_cycle_count() > initial_count


class TestConfig:
    def test_density_classify(self, cfg):
        assert cfg.density.classify(0)  == "LOW"
        assert cfg.density.classify(10) == "LOW"
        assert cfg.density.classify(11) == "MEDIUM"
        assert cfg.density.classify(20) == "MEDIUM"
        assert cfg.density.classify(21) == "HIGH"
        assert cfg.density.classify(50) == "HIGH"

    def test_signal_green_times(self, cfg):
        assert cfg.signal.green_times["LOW"]    == 20
        assert cfg.signal.green_times["MEDIUM"] == 40
        assert cfg.signal.green_times["HIGH"]   == 60
