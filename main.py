"""
main.py — CLI Entry Point
==========================
AI-Driven Smart Traffic Management System

Usage:
    python main.py --mode full              # Full pipeline
    python main.py --mode detect            # Detection only (loops videos)
    python main.py --mode simulate          # Simulation only
    python main.py --mode dashboard         # Launch dashboard
    python main.py --generate-test-videos   # Generate synthetic videos
    python main.py --export-reports         # Export CSV reports
    python main.py --help
"""

import sys
import time
import click
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loguru import logger
from config import Config
from src.utils import setup_logger, generate_all_test_videos


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION PIPELINE  (videos loop forever — Ctrl+C to stop)
# ═══════════════════════════════════════════════════════════════════════════════

def run_detection_pipeline(cfg: Config, directions=None, max_frames=None) -> None:
    """
    Run YOLOv8 detection + tracking + counting + density + signal.
    Videos loop continuously so the dashboard always has fresh data.
    Press Ctrl+C to stop.
    """
    import cv2
    from src.video_loader import VideoStream
    from src.detector import VehicleDetector
    from src.tracker import VehicleTracker
    from src.vehicle_counter import VehicleCounter
    from src.density_analyzer import DensityAnalyzer
    from src.signal_controller import SignalController
    from src.database import DatabaseManager
    from src.utils import draw_hud

    logger.info("=" * 60)
    logger.info("  Detection Pipeline  (videos loop — Ctrl+C to stop)")
    logger.info("=" * 60)

    db = DatabaseManager(cfg)
    db.initialize()
    session_id = db.start_session(mode="detect")

    detector = VehicleDetector(cfg)
    tracker  = VehicleTracker(cfg)
    counter  = VehicleCounter(cfg)
    analyzer = DensityAnalyzer(cfg)
    signal   = SignalController(cfg)

    if not detector.load():
        logger.error("Failed to load YOLO model. Exiting.")
        sys.exit(1)

    detector.warmup()
    tracker.init_all()
    # Write initial signal state so dashboard shows live signal immediately
    for _d_init in cfg.video.directions:
        _s_init = signal.get_state(_d_init)
        if _s_init:
            db.insert_signal_state(_d_init, _s_init)
    db.flush_all()

    target_dirs = directions or cfg.video.directions

    # ── Open one VideoStream per direction ───────────────────────────────────
    streams = {}
    for d in target_dirs:
        path = cfg.paths.video_paths.get(d)
        if path and Path(str(path)).exists():
            s = VideoStream(d, str(path), cfg)
            if s.open():
                streams[d] = s
                logger.info(f"[{d}] Stream opened: {path}")
            else:
                logger.warning(f"[{d}] Could not open stream.")
        else:
            logger.warning(
                f"[{d}] Video not found: {path}. "
                "Run: python main.py --generate-test-videos"
            )

    if not streams:
        logger.error("No video streams opened. Exiting.")
        db.end_session(0)
        sys.exit(1)

    # ── Counting line: per-direction, from each video's REAL native height ──
    # Previously this used one shared cfg.video.frame_height (640) for every
    # direction, which only made sense because every frame was being
    # force-resized to 640x640 below. Now that frames are left at native
    # resolution (see resize_wh below), a single shared line_y would be
    # wrong for 3 of these 4 feeds — e.g. way too high on the 4K east/west
    # footage, cutting off most of the road.
    for d, s in streams.items():
        native_h = s.info.height if s.info else cfg.video.frame_height
        counter.init_direction(d, frame_height=native_h)

    # ── No forced resize before detection ─────────────────────────────────
    # Previously every frame was squeezed to a single fixed size
    # (cfg.video.frame_width/height = 640x640) via cv2.resize() in
    # video_loader.py, regardless of each video's real native resolution.
    # Source videos here range from 480p (south) to 4K (east/west) — forcing
    # all of them through the same 640x640 target meant the 4K feeds lost
    # vastly more real detail per vehicle than the already-small 480p feed,
    # which is why small/distant vehicles were detected far worse on
    # east/west than on north/south despite clearer source footage.
    # YOLO's own model.track()/model.predict() already resizes whatever
    # frame it's given to `imgsz` internally (with its own letterboxing),
    # so this manual pre-resize was a redundant, lossy step that destroyed
    # detail before YOLO ever got a chance to use it. Passing resize=None
    # here means VideoStream.frames()'s `if resize:` guard never fires, so
    # frames now reach the detector at full native resolution, and YOLO's
    # own resize-to-imgsz happens from the best possible source detail.
    resize_wh    = None
    skip         = cfg.video.skip_frames
    log_interval = cfg.database.log_every_n_frames
    save_interval= cfg.video.annotated_frame_interval
    show_preview = cfg.video.show_preview

    frame_count = 0
    loop_count  = {d: 0 for d in streams}
    fps_times   = []
    t_start     = time.time()

    # Build per-direction frame generators (loop=True → infinite loop)
    generators = {
        d: s.frames(skip=skip, resize=resize_wh, loop=True)
        for d, s in streams.items()
    }

    dir_list = list(streams.keys())
    dir_idx  = 0      # round-robin index

    # ── Preview window sizing for a 1366x760 laptop screen ──────────────────
    # Previously cv2.imshow() created default-sized windows matching each
    # frame's own resolution. Since frames are no longer force-resized
    # before detection (now at native resolution — up to 3840x2160 for
    # east/west), the preview windows would open far larger than a laptop
    # screen, badly overlapping and unusable for a live demo. This creates
    # each window once upfront, sized and positioned into a 2x2 grid that
    # fits a 1366x760 display, independent of each video's real resolution
    # (the displayed image is scaled to the window; detection itself still
    # runs on the full native frame — this only affects what's shown).
    #
    # Reserve a small margin for window title bars (~30px) and the taskbar
    # (~50px) so the bottom row isn't clipped.
    _SCREEN_W, _SCREEN_H = 1366, 760
    _MARGIN_BOTTOM = 50      # taskbar
    _TITLEBAR_H    = 30      # per-window title bar
    _usable_w  = _SCREEN_W // 2
    _usable_h  = (_SCREEN_H - _MARGIN_BOTTOM) // 2 - _TITLEBAR_H

    _grid_positions = {
        0: (0,             0),
        1: (_usable_w,     0),
        2: (0,             _usable_h + _TITLEBAR_H),
        3: (_usable_w,     _usable_h + _TITLEBAR_H),
    }

    if cfg.video.show_preview:
        for _i, _d in enumerate(dir_list):
            _title = f"Traffic - {_d.upper()}"
            cv2.namedWindow(_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_title, _usable_w, _usable_h)
            _pos = _grid_positions.get(_i, (0, 0))
            cv2.moveWindow(_title, _pos[0], _pos[1])

    logger.info(f"Processing directions: {dir_list}  |  loop=True")
    logger.info("Press Ctrl+C to stop.\n")

    # Start signal controller in background thread — ticks every 1s reliably
    signal.start_threaded(tick_interval=0.25)  # 0.25s ticks = fast transitions
    _sig_ticked = True   # flag that threaded mode is on



    try:
        while True:
            if max_frames and frame_count >= max_frames:
                logger.info(f"Reached max_frames={max_frames}. Stopping.")
                break

            direction = dir_list[dir_idx % len(dir_list)]
            dir_idx  += 1

            try:
                frame_data = next(generators[direction])
            except StopIteration:
                # Should not happen with loop=True, but handle gracefully
                loop_count[direction] += 1
                logger.debug(f"[{direction}] Loop {loop_count[direction]} restarting.")
                generators[direction] = streams[direction].frames(
                    skip=skip, resize=resize_wh, loop=True
                )
                continue

            frame_count += 1
            t0    = time.time()
            frame = frame_data.frame

            # ── Detection + Tracking (single inference pass via ByteTrack) ───
            track_result, det_result = tracker.update(
                frame,
                direction=direction,
                frame_number=frame_data.frame_number,
                timestamp=frame_data.timestamp,
            )

            # ── Counting ─────────────────────────────────────────────────────
            count_result = counter.update(track_result, frame)

            # ── Live vehicle count = YOLO detections in this frame ──────────
            current_count = det_result.count   # direct from YOLO — always live
            density_reading = analyzer.analyze(direction, current_count)

            # ── Signal ───────────────────────────────────────────────────────
            signal.update_density(direction, density_reading)
            # Signal ticks in background thread (started above)
            # Update density into controller every frame
            pass
            sig_state = signal.get_state(direction)

            # ── FPS ──────────────────────────────────────────────────────────
            fps_times.append(time.time() - t0)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0



            # ── DB write ─────────────────────────────────────────────────────
            if frame_count % 5 == 0:
                count_result.current_count = current_count
                db.insert_count_result(count_result)
                db.insert_density_reading(density_reading)
            if frame_count % 20 == 0:
                db.insert_detection_result(det_result)
            # Write signal states every 3 frames for fast dashboard updates
            if frame_count % 3 == 0:
                for _d in dir_list:
                    _s = signal.get_state(_d)
                    if _s:
                        db.insert_signal_state(_d, _s)

            # ── Annotate every frame with boxes + track IDs + HUD ───────────
            annotated = frame.copy()

            # Draw bounding boxes with class + track ID
            CLASS_COLORS = {
                "car":        (0, 200, 0),
                "motorcycle": (0, 210, 255),
                "bus":        (255, 80, 0),
                "truck":      (200, 0, 200),
            }
            import cv2 as _cv2
            for det in det_result.detections:
                color = CLASS_COLORS.get(det.class_name, (180,180,180))
                _cv2.rectangle(annotated, (det.x1,det.y1), (det.x2,det.y2), color, 2)
                tid   = f"#{det.track_id}" if det.track_id else "DET"
                label = f"{det.class_name} {tid}"
                (tw,th),_ = _cv2.getTextSize(label, _cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                _cv2.rectangle(annotated, (det.x1, det.y1-th-4), (det.x1+tw+4, det.y1), color, -1)
                _cv2.putText(annotated, label, (det.x1+2, det.y1-3),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, _cv2.LINE_AA)

            # Draw tracking trails
            for track in track_result.tracks:
                if len(track.history) > 1:
                    for j in range(1, len(track.history)):
                        _cv2.line(annotated, track.history[j-1], track.history[j], (0,255,255), 1)

            # Draw counting line
            annotated = counter.annotate(annotated, count_result)

            # Draw HUD overlay
            if sig_state:
                annotated = draw_hud(
                    annotated, direction,
                    vehicle_count=current_count,
                    density_level=density_reading.density_level,
                    signal_state=sig_state.phase.value,
                    green_time=sig_state.allocated_green,
                    fps=fps,
                )

            # ── Live preview window (OpenCV) ─────────────────────────────────
            if show_preview:
                win_title = f"Traffic - {direction.upper()}"
                cv2.imshow(win_title, annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("'q' pressed — stopping.")
                    break

            # ── Save annotated frame snapshot ────────────────────────────────
            if frame_count % save_interval == 0:
                out_path = (
                    cfg.paths.frames_dir
                    / f"{direction}_frame{frame_data.frame_number:06d}.jpg"
                )
                import cv2 as _cv2
                _cv2.imwrite(str(out_path), annotated)

            # ── Console log ──────────────────────────────────────────────────
            if frame_count % 50 == 0:
                active     = signal.get_active_direction()
                act_state  = signal.get_state(active)
                remaining  = act_state.remaining_seconds if act_state else 0
                logger.info(
                    f"Frame {frame_count:5d} | {direction:<5} | "
                    f"Count={current_count:2d} | "
                    f"Density={density_reading.density_level:<6} | "
                    f"FPS={fps:5.1f} | "
                    f"Signal={active.upper()} {act_state.phase.value if act_state else '?'} "
                    f"({remaining:.0f}s left)"
                )

    except KeyboardInterrupt:
        logger.info("\nDetection pipeline stopped by user (Ctrl+C).")
    finally:
        signal.stop()
        for s in streams.values():
            s.close()
        if show_preview:
            import cv2 as _cv2
            _cv2.destroyAllWindows()
        db.flush_all()
        db.end_session(frame_count)
        elapsed = time.time() - t_start
        logger.success(
            f"\n✅ Detection complete.\n"
            f"   Frames processed : {frame_count}\n"
            f"   Time elapsed     : {elapsed:.1f}s\n"
            f"   Avg FPS          : {frame_count / max(elapsed,1):.1f}\n"
            f"   Session ID       : {session_id}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation_pipeline(cfg: Config, max_steps=None) -> None:
    """Run the SUMO / statistical simulation engine."""
    from src.density_analyzer import DensityAnalyzer
    from src.signal_controller import SignalController
    from src.simulation_engine import SimulationEngine
    from src.database import DatabaseManager

    logger.info("=" * 60)
    logger.info("  Simulation Pipeline Starting")
    logger.info("=" * 60)

    db = DatabaseManager(cfg)
    db.initialize()
    db.start_session(mode="simulate")

    analyzer = DensityAnalyzer(cfg)
    signal   = SignalController(cfg)

    baseline = {"north": 8, "south": 15, "east": 25, "west": 12}
    for direction, count in baseline.items():
        reading = analyzer.analyze(direction, count)
        signal.update_density(direction, reading)

    engine = SimulationEngine(signal, analyzer, db, cfg)
    engine.try_connect_sumo()

    steps = max_steps or cfg.simulation.max_steps

    def on_step(step, metrics):
        if step % 100 == 0:
            logger.info(
                f"[Sim] Step {step:4d} | "
                f"Active: {signal.get_active_direction().upper():<5} | "
                f"Cycle: {signal.get_cycle_count()}"
            )

    result = engine.run(max_steps=steps, step_callback=on_step)
    db.flush_all()
    db.end_session()

    logger.success(
        f"\n✅ Simulation complete.\n"
        f"   Mode          : {result.mode}\n"
        f"   Steps         : {result.total_steps}\n"
        f"   Throughput    : {result.throughput} vehicles\n"
        f"   Avg wait      : {result.avg_waiting_time:.2f}s\n"
        f"   Avg queue     : {result.avg_queue_length:.1f} vehicles\n"
        f"   Avg congestion: {result.avg_congestion_score:.1f}/100"
    )


def run_dashboard() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = ROOT / "dashboard" / "streamlit_app.py"
    logger.info(f"Launching dashboard: {dashboard_path}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
    )


def run_full_pipeline(cfg: Config, max_frames=None, sim_steps=None) -> None:
    """Run detection + simulation concurrently in separate threads."""
    import threading

    logger.info("=" * 60)
    logger.info("  Full Pipeline  (Detection + Simulation)")
    logger.info("=" * 60)

    det_thread = threading.Thread(
        target=run_detection_pipeline,
        args=(cfg,),
        kwargs={"max_frames": max_frames},
        name="DetectionThread",
        daemon=True,
    )
    sim_thread = threading.Thread(
        target=run_simulation_pipeline,
        args=(cfg,),
        kwargs={"max_steps": sim_steps},
        name="SimulationThread",
        daemon=True,
    )

    det_thread.start()
    time.sleep(2)
    sim_thread.start()

    try:
        det_thread.join()
        sim_thread.join()
    except KeyboardInterrupt:
        logger.info("Full pipeline interrupted by user.")

    logger.success("✅ Full pipeline complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

@click.command()
@click.option(
    "--mode",
    type=click.Choice(["full", "detect", "simulate", "dashboard"]),
    default="full",
    show_default=True,
    help="Pipeline mode to run.",
)
@click.option("--directions", "-d", multiple=True, default=None,
              help="Directions to process (e.g. -d north -d east).")
@click.option("--max-frames", type=int, default=None,
              help="Stop detection after N frames.")
@click.option("--sim-steps",  type=int, default=None,
              help="Override max simulation steps.")
@click.option("--generate-test-videos", is_flag=True, default=False,
              help="Generate synthetic test videos and exit.")
@click.option("--export-reports", is_flag=True, default=False,
              help="Export all DB tables to CSV and exit.")
@click.option("--show-preview", is_flag=True, default=False,
              help="Show live OpenCV preview windows while detecting.")
@click.option("--log-level",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", show_default=True)
def main(mode, directions, max_frames, sim_steps,
         generate_test_videos, export_reports, show_preview, log_level):
    """
    🚦 AI-Driven Smart Traffic Management System

    \b
    Examples:
      python main.py --mode detect                     # Loop videos forever
      python main.py --mode detect --show-preview      # + live OpenCV window
      python main.py --mode detect --max-frames 500    # Stop after 500 frames
      python main.py --mode simulate --sim-steps 3600
      python main.py --mode full
      python main.py --mode dashboard
      python main.py --generate-test-videos
      python main.py --export-reports
    """
    cfg = Config()
    setup_logger(log_level=log_level, log_dir=cfg.paths.logs_dir)

    # Apply show_preview flag to config
    cfg.video.show_preview = show_preview

    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    console.print(Panel.fit(
        "[bold cyan]🚦 AI-Driven Smart Traffic Management System[/bold cyan]\n"
        "[dim]YOLOv8 · ByteTrack · SUMO · Streamlit · SQLite[/dim]",
        border_style="cyan",
    ))

    if generate_test_videos:
        logger.info("Generating synthetic test videos …")
        generate_all_test_videos(cfg.paths.input_dir, duration=60)
        logger.success("Done. Videos saved to data/input/")
        return

    if export_reports:
        from src.database import DatabaseManager
        db = DatabaseManager(cfg)
        db.initialize()
        db.export_all_reports()
        logger.success(f"Reports exported to {cfg.paths.reports_dir}")
        return

    dirs = list(directions) if directions else None

    if mode == "detect":
        run_detection_pipeline(cfg, directions=dirs, max_frames=max_frames)
    elif mode == "simulate":
        run_simulation_pipeline(cfg, max_steps=sim_steps)
    elif mode == "dashboard":
        run_dashboard()
    elif mode == "full":
        run_full_pipeline(cfg, max_frames=max_frames, sim_steps=sim_steps)


if __name__ == "__main__":
    main()
