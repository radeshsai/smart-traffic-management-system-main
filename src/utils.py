"""
src/utils.py — Shared Utilities
=================================
Logging setup, helper functions, frame annotation tools,
synthetic video generator, and common decorators.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Tuple, Dict, List, Optional

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logger(log_level: str = "INFO", log_dir: Path = None) -> None:
    """
    Configure Loguru logger with console + rotating file output.

    Args:
        log_level: Logging level string (DEBUG/INFO/WARNING/ERROR).
        log_dir:   Directory for log files. Defaults to outputs/logs/.
    """
    logger.remove()  # Remove default handler

    # Console handler with colour
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
    )

    # File handler with rotation
    if log_dir is None:
        log_dir = Path(__file__).resolve().parents[1] / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        str(log_file),
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
    )

    logger.info(f"Logger initialised → {log_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def timer(func):
    """Decorator: log execution time of any function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"⏱  {func.__qualname__} finished in {elapsed:.3f}s")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator: retry a function on exception."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for "
                        f"{func.__qualname__}: {exc}"
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise RuntimeError(
                f"{func.__qualname__} failed after {max_attempts} attempts."
            )
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME ANNOTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Direction header colours (BGR)
DIRECTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "north": (255, 100, 100),
    "south": (100, 255, 100),
    "east":  (100, 100, 255),
    "west":  (255, 255, 100),
}

# Density badge colours
DENSITY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "LOW":    (0, 200, 0),
    "MEDIUM": (0, 165, 255),
    "HIGH":   (0, 0, 220),
}


def draw_bounding_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    track_id: Optional[int] = None,
) -> np.ndarray:
    """Draw a labelled bounding box on a frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label}" + (f" #{track_id}" if track_id is not None else "")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, text,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return frame


def draw_hud(
    frame: np.ndarray,
    direction: str,
    vehicle_count: int,
    density_level: str,
    signal_state: str,
    green_time: int,
    fps: float = 0.0,
) -> np.ndarray:
    """
    Draw a heads-up display overlay on a video frame.

    Args:
        frame:         BGR image array.
        direction:     One of north/south/east/west.
        vehicle_count: Total vehicles in current frame.
        density_level: LOW / MEDIUM / HIGH.
        signal_state:  GREEN / YELLOW / RED.
        green_time:    Allocated green seconds.
        fps:           Current processing FPS.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    bar_h = 56
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    dir_color = DIRECTION_COLORS.get(direction, (200, 200, 200))
    density_color = DENSITY_COLORS.get(density_level, (200, 200, 200))

    signal_color_map = {
        "GREEN": (0, 220, 0),
        "YELLOW": (0, 200, 220),
        "RED": (0, 0, 220),
    }
    sig_color = signal_color_map.get(signal_state, (200, 200, 200))

    # Direction label
    cv2.putText(frame, direction.upper(), (10, 36),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, dir_color, 2, cv2.LINE_AA)

    # Vehicle count
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (170, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Density level
    cv2.putText(frame, f"Density: {density_level}", (170, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, density_color, 1, cv2.LINE_AA)

    # Signal state
    cv2.putText(frame, f"Signal: {signal_state} ({green_time}s)", (340, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, sig_color, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (340, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def draw_counting_line(
    frame: np.ndarray,
    line_y: int,
    color: Tuple[int, int, int] = (0, 255, 255),
    label: str = "COUNT LINE",
) -> np.ndarray:
    """Draw a horizontal counting line across the frame."""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), color, 2)
    cv2.putText(frame, label, (10, line_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC VIDEO GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_video(
    output_path: Path,
    duration_seconds: int = 30,
    fps: int = 10,
    width: int = 640,
    height: int = 480,
    num_vehicles: int = 15,
    direction: str = "north",
) -> None:
    """
    Generate a synthetic traffic video for testing.
    Draws simple coloured rectangles moving across a grey road.

    Args:
        output_path:      Where to save the .mp4 file.
        duration_seconds: Length of video in seconds.
        fps:              Frames per second.
        width:            Frame width (pixels).
        height:           Frame height (pixels).
        num_vehicles:     Number of moving 'vehicle' rectangles.
        direction:        Label shown on the video.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = duration_seconds * fps

    # Initialise vehicle states: (x, y, w, h, speed, color)
    rng = np.random.default_rng(seed=42)
    vehicles = []
    for _ in range(num_vehicles):
        vw = rng.integers(30, 70)
        vh = rng.integers(20, 45)
        x  = rng.integers(0, width - vw)
        y  = rng.integers(-height, 0)           # Start above frame
        speed = rng.integers(3, 10)
        color = tuple(int(c) for c in rng.integers(80, 255, 3))
        vehicles.append([x, int(y), int(vw), int(vh), int(speed), color])

    for f in range(total_frames):
        # Road background
        frame = np.full((height, width, 3), 60, dtype=np.uint8)

        # Lane markings
        for lane_x in [width // 4, width // 2, 3 * width // 4]:
            for seg_y in range(0, height, 40):
                cv2.rectangle(frame, (lane_x - 2, seg_y), (lane_x + 2, seg_y + 20),
                              (200, 200, 200), -1)

        # Move and draw vehicles
        for v in vehicles:
            x, y, vw, vh, speed, color = v
            y += speed
            if y > height:
                y = rng.integers(-80, -vh)
                x = rng.integers(0, width - vw)
            v[1] = y

            cv2.rectangle(frame, (x, y), (x + vw, y + vh), color, -1)
            cv2.rectangle(frame, (x, y), (x + vw, y + vh), (255, 255, 255), 1)

        # Direction watermark
        cv2.putText(frame, f"[SYNTHETIC] {direction.upper()}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame {f + 1}/{total_frames}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    logger.info(f"Synthetic video saved → {output_path} ({total_frames} frames)")


def generate_all_test_videos(input_dir: Path, duration: int = 30) -> None:
    """Generate synthetic test videos for all four directions."""
    directions_vehicles = {
        "north": 8,    # LOW density
        "south": 15,   # MEDIUM density
        "east":  25,   # HIGH density
        "west":  12,   # MEDIUM density
    }
    for direction, n_vehicles in directions_vehicles.items():
        out_path = input_dir / f"{direction}.mp4"
        logger.info(f"Generating {direction} video with {n_vehicles} vehicles …")
        generate_synthetic_video(
            output_path=out_path,
            duration_seconds=duration,
            num_vehicles=n_vehicles,
            direction=direction,
        )
    logger.success("✅ All 4 synthetic test videos generated.")


# ═══════════════════════════════════════════════════════════════════════════════
# MISC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def timestamp() -> str:
    """Return ISO-formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ms_to_time(ms: float) -> str:
    """Convert milliseconds to HH:MM:SS.mmm string."""
    total_s = ms / 1000
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))


def moving_average(values: List[float], window: int = 5) -> float:
    """Calculate moving average over the last `window` values."""
    if not values:
        return 0.0
    window_vals = values[-window:]
    return sum(window_vals) / len(window_vals)


def format_table(data: Dict, title: str = "") -> str:
    """Format a dict as an ASCII table string for logging."""
    lines = [f"\n{'=' * 40}", f"  {title}", f"{'=' * 40}"]
    for k, v in data.items():
        lines.append(f"  {str(k):<20} {str(v)}")
    lines.append(f"{'=' * 40}\n")
    return "\n".join(lines)
