"""
config.py — Central Configuration Module
=========================================
All tunable parameters, paths, and constants for the
AI-Driven Smart Traffic Management System.

Import this module wherever configuration is needed:
    from config import Config
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class PathConfig:
    """All filesystem paths used across the project."""

    # Root
    root: Path = ROOT_DIR

    # Data
    data_dir: Path = ROOT_DIR / "data"
    input_dir: Path = ROOT_DIR / "data" / "input"
    processed_dir: Path = ROOT_DIR / "data" / "processed"

    # Models
    models_dir: Path = ROOT_DIR / "models"
    yolo_model: Path = ROOT_DIR / "models" / "yolov8s.pt"
                                    # Switched from yolov8n.pt -> yolov8s.pt.
                                    # After fixing resolution (640), confidence
                                    # (0.25), NMS IoU (0.3), per-direction
                                    # native-resolution frames, and the
                                    # ByteTrack tracker-config confidence gate,
                                    # remaining missed detections are believed
                                    # to be yolov8n's real small-object accuracy
                                    # ceiling rather than a config/code bug.
                                    # yolov8s trades CPU speed for materially
                                    # better small-object recall — this is the
                                    # next experiment now that other causes are
                                    # ruled out. If FPS drops too far on CPU,
                                    # revert to yolov8n.pt (and update
                                    # detection.model_name to match below).

    # Database
    database_dir: Path = ROOT_DIR / "database"
    db_path: Path = ROOT_DIR / "database" / "traffic.db"
    schema_path: Path = ROOT_DIR / "database" / "schema.sql"

    # Outputs
    outputs_dir: Path = ROOT_DIR / "outputs"
    frames_dir: Path = ROOT_DIR / "outputs" / "frames"
    reports_dir: Path = ROOT_DIR / "outputs" / "reports"
    logs_dir: Path = ROOT_DIR / "outputs" / "logs"
    sim_results_dir: Path = ROOT_DIR / "outputs" / "simulation_results"

    # Simulation
    simulation_dir: Path = ROOT_DIR / "simulation"
    sumo_config_dir: Path = ROOT_DIR / "simulation" / "sumo_config"
    sumo_net: Path = ROOT_DIR / "simulation" / "sumo_config" / "intersection.net.xml"
    sumo_routes: Path = ROOT_DIR / "simulation" / "sumo_config" / "routes.rou.xml"
    sumo_signals: Path = ROOT_DIR / "simulation" / "sumo_config" / "signals.add.xml"

    # Video inputs per direction
    @property
    def video_paths(self) -> Dict[str, Path]:
        return {
            "north": self.input_dir / "north.mp4",
            "south": self.input_dir / "south.mp4",
            "east":  self.input_dir / "east.mp4",
            "west":  self.input_dir / "west.mp4",
        }

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        dirs = [
            self.processed_dir, self.models_dir, self.database_dir,
            self.frames_dir, self.reports_dir, self.logs_dir, self.sim_results_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class VideoConfig:
    """Video capture and processing parameters."""

    # Directions to process (order matters for signal cycling)
    directions: List[str] = field(default_factory=lambda: ["north", "south", "east", "west"])

    # Frame processing
    frame_width: int = 640          # Resize width for processing
    frame_height: int = 640         # Resize height for processing
                                     # Raised from 416 -> 640, matching
                                     # DetectionConfig.imgsz below. main.py
                                     # resizes every camera frame to this
                                     # size BEFORE it reaches the detector
                                     # (see read_one_frame_per_direction /
                                     # VideoStream.frames in video_loader.py).
                                     # At 416, small/distant vehicles were
                                     # already shrunk past the point of
                                     # recovery before YOLO even ran, so
                                     # raising imgsz alone wouldn't have
                                     # fixed missed detections — the detail
                                     # was thrown away one step earlier.
                                     # Keep this equal to imgsz: there's no
                                     # benefit to YOLO upsampling a smaller
                                     # frame back up, and no benefit to
                                     # capturing more detail than imgsz can
                                     # use.
    fps_target: int = 10            # Target FPS for processing (skip frames for speed)
    skip_frames: int = 3            # Process every Nth frame (1 = every frame)

    # Display
    show_preview: bool = False      # Show OpenCV preview window (disable on server)
    save_annotated_frames: bool = False   # Save bounding-box annotated frames
    annotated_frame_interval: int = 60   # Save every Nth processed frame


# ═══════════════════════════════════════════════════════════════════════════════
# YOLO DETECTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class DetectionConfig:
    """YOLOv8 detection parameters."""

    # Model
    model_name: str = "yolov8s.pt"    # nano=fast, small=balanced, medium=accurate
                                       # Matches paths.yolo_model above — used as
                                       # the auto-download fallback name if the
                                       # configured weights file isn't found locally.
    device: str = "cpu"               # "cpu" | "cuda" | "mps"
    imgsz: int = 960                  # Inference image size (must be multiple of 32)
                                       # Raised again from 640 -> 960. The remaining missed
                                       # detections (a distant vehicle cluster near the
                                       # horizon/overpass) are now small enough in pixel
                                       # terms that YOLO's internal feature maps at 640
                                       # likely never form a strong signal for them in the
                                       # first place — confidence threshold tuning can't
                                       # recover a detection that was never confidently
                                       # produced. 960 gives the model more spatial
                                       # resolution to work with for small objects.
                                       # NOTE: unlike the earlier 416->640 change, this does
                                       # NOT need a matching frame_width/frame_height update
                                       # below — main.py no longer pre-resizes frames before
                                       # detection (resize_wh=None), so imgsz is the only
                                       # thing controlling YOLO's internal resize now.
                                       # Real cost: substantially more CPU time per frame
                                       # on top of yolov8s's own cost — expect FPS to drop
                                       # further from wherever it sits today. If unworkable,
                                       # 768 is a smaller step up from 640 to try first.

    # Thresholds
    confidence: float = 0.15          # Minimum detection confidence (0–1)
                                       # Lowered again from 0.25 -> 0.15 as an experiment
                                       # to mitigate a specific, confirmed limitation:
                                       # COCO-pretrained YOLO models detect REAR views of
                                       # vehicles noticeably less confidently than front
                                       # views, because COCO's training photos are
                                       # front/side-heavy. On a divided road where one
                                       # carriageway drives away from the camera, this
                                       # shows up as a consistent one-side miss that has
                                       # nothing to do with distance, size, or resolution.
                                       # There is no clean way to threshold by viewing
                                       # angle with a stock YOLO confidence parameter, so
                                       # this lowers the bar globally as a blunt
                                       # mitigation — expect more false positives
                                       # (shadows, signage, lane markings) as a real
                                       # tradeoff, not a free improvement. If false
                                       # positives become a problem, move this back
                                       # toward 0.25 and treat the rear-view gap as a
                                       # documented model limitation instead.
    iou_threshold: float = 0.30       # NMS IoU threshold
                                       # Lowered from 0.45 -> 0.30. NMS discards a detected
                                       # box if it overlaps an already-kept box (same class)
                                       # by more than this fraction, on the assumption it's
                                       # a duplicate detection of the same object. In dense,
                                       # tightly-packed traffic (e.g. a jam viewed from an
                                       # elevated angle), two genuinely separate vehicles can
                                       # easily overlap >45% on screen, so the higher threshold
                                       # was causing real, distinct vehicles to be thrown away
                                       # as "duplicates." 0.30 requires heavier overlap before
                                       # suppressing a box, keeping more separate-but-close
                                       # vehicles. Tradeoff: if you start seeing the SAME
                                       # vehicle double-boxed, raise this back up slightly
                                       # (try 0.35) rather than reverting all the way to 0.45.

    # Vehicle classes in COCO dataset (YOLOv8 default)
    # 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])

    # Human-readable class names mapping
    class_names: Dict[int, str] = field(default_factory=lambda: {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    })

    # Bounding box colors per class (BGR for OpenCV)
    class_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: {
        2: (0, 255, 0),      # car       → green
        3: (255, 165, 0),    # motorcycle → orange
        5: (0, 0, 255),      # bus       → red
        7: (255, 0, 255),    # truck     → magenta
    })


# ═══════════════════════════════════════════════════════════════════════════════
# TRACKING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class TrackingConfig:
    """ByteTrack / multi-object tracking parameters."""

    tracker_type: str = "bytetrack"   # "bytetrack" | "botsort"
    track_high_thresh: float = 0.5    # High confidence detection threshold
    track_low_thresh: float = 0.1     # Low confidence detection threshold
    new_track_thresh: float = 0.6     # New track creation threshold
    track_buffer: int = 30            # Frames to keep a lost track alive
    match_thresh: float = 0.8         # IoU threshold for track-detection matching
    min_box_area: float = 10.0        # Minimum bounding box area (pixels²)


# ═══════════════════════════════════════════════════════════════════════════════
# DENSITY ANALYSIS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class DensityConfig:
    """Traffic density classification thresholds."""

    # Density bands (vehicle count → level)
    low_max: int = 10       # 0–10   → LOW
    medium_max: int = 20    # 11–20  → MEDIUM
                            # 21+    → HIGH

    # Density level labels
    levels: Dict[str, str] = field(default_factory=lambda: {
        "LOW":    "🟢 Low",
        "MEDIUM": "🟡 Medium",
        "HIGH":   "🔴 High",
    })

    # Rolling window for smoothing density calculation (in frames)
    smoothing_window: int = 5

    def classify(self, count: int) -> str:
        """Return density level string for a given vehicle count."""
        if count <= self.low_max:
            return "LOW"
        elif count <= self.medium_max:
            return "MEDIUM"
        else:
            return "HIGH"


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL CONTROL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SignalConfig:
    """Traffic signal timing and cycling parameters."""

    # Green time per density level (seconds)
    green_times: Dict[str, int] = field(default_factory=lambda: {
        "LOW":    15,   # 0–10 vehicles
        "MEDIUM": 25,   # 11–20 vehicles
        "HIGH":   40,   # 21+ vehicles
    })

    # Fixed phase durations (seconds)
    yellow_time: int = 3      # Yellow / amber phase
    all_red_time: int = 1     # All-red clearance phase between directions
    min_green_time: int = 5   # Absolute minimum green (safety)
    max_green_time: int = 90  # Absolute maximum green (fairness cap)

    # Signal phase names
    phases: List[str] = field(default_factory=lambda: ["GREEN", "YELLOW", "RED"])

    # Cycling order (must match directions in VideoConfig)
    cycle_order: List[str] = field(default_factory=lambda: ["north", "south", "east", "west"])

    def get_green_time(self, density_level: str) -> int:
        """Return green time (seconds) for a given density level."""
        return self.green_times.get(density_level, self.green_times["LOW"])


# ═══════════════════════════════════════════════════════════════════════════════
# SUMO SIMULATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SimulationConfig:
    """SUMO and TraCI simulation parameters."""

    # SUMO binary
    sumo_binary: str = "sumo"          # "sumo" (headless) | "sumo-gui" (visual)
    sumo_home: str = os.environ.get("SUMO_HOME", "/usr/share/sumo")

    # Simulation timing
    step_length: float = 1.0           # Simulation step = 1 second real time
    max_steps: int = 3600              # Max steps per run (3600 = 1 simulated hour)
    warmup_steps: int = 60             # Steps before metrics collection begins

    # TraCI connection
    traci_port: int = 8813             # Port for TraCI connection
    traci_host: str = "localhost"

    # SUMO output files
    tripinfo_output: str = "outputs/simulation_results/tripinfo.xml"
    summary_output: str = "outputs/simulation_results/summary.xml"
    queue_output: str = "outputs/simulation_results/queue.xml"

    # Intersection junction ID (must match intersection.net.xml)
    junction_id: str = "center"

    # Traffic light ID (must match signals.add.xml)
    tl_id: str = "center_tl"

    # Metrics collection interval (every N steps)
    metrics_interval: int = 10

    # Vehicle type IDs in SUMO routes
    vehicle_types: List[str] = field(default_factory=lambda: [
        "car", "motorcycle", "bus", "truck"
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class DatabaseConfig:
    """Database connection and retention settings."""

    # Backend: "sqlite" | "postgresql"
    backend: str = "sqlite"

    # SQLite (default)
    sqlite_path: str = str(ROOT_DIR / "database" / "traffic.db")

    # PostgreSQL (optional — set via environment variables)
    pg_host: str = os.environ.get("PG_HOST", "localhost")
    pg_port: int = int(os.environ.get("PG_PORT", "5432"))
    pg_user: str = os.environ.get("PG_USER", "traffic_user")
    pg_password: str = os.environ.get("PG_PASSWORD", "")
    pg_database: str = os.environ.get("PG_DATABASE", "traffic_db")

    # Data retention
    max_rows_per_table: int = 100_000   # Auto-prune oldest rows beyond this
    log_every_n_frames: int = 10         # Insert DB record every N frames

    @property
    def connection_string(self) -> str:
        """Return SQLAlchemy-compatible connection string."""
        if self.backend == "postgresql":
            return (
                f"postgresql://{self.pg_user}:{self.pg_password}"
                f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
            )
        return f"sqlite:///{self.sqlite_path}"


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class LoggingConfig:
    """Loguru logging settings."""

    log_level: str = "INFO"                        # DEBUG | INFO | WARNING | ERROR
    log_file: str = str(ROOT_DIR / "outputs" / "logs" / "traffic_{time}.log")
    rotation: str = "10 MB"                        # Rotate when file hits 10 MB
    retention: str = "7 days"                      # Keep logs for 7 days
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    colorize: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class DashboardConfig:
    """Streamlit dashboard settings."""

    page_title: str = "🚦 Smart Traffic Management"
    page_icon: str = "🚦"
    layout: str = "wide"                  # "wide" | "centered"
    refresh_interval: int = 2             # Auto-refresh every N seconds
    max_chart_points: int = 100           # Max data points shown on live charts
    theme: str = "dark"                   # Dashboard color theme


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER CONFIG CLASS
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    """
    Master configuration object.

    Usage:
        from config import Config
        cfg = Config()
        print(cfg.detection.confidence)
        print(cfg.paths.yolo_model)
    """

    def __init__(self):
        self.paths      = PathConfig()
        self.video      = VideoConfig()
        self.detection  = DetectionConfig()
        self.tracking   = TrackingConfig()
        self.density    = DensityConfig()
        self.signal     = SignalConfig()
        self.simulation = SimulationConfig()
        self.database   = DatabaseConfig()
        self.logging    = LoggingConfig()
        self.dashboard  = DashboardConfig()

        # Ensure all output directories exist on instantiation
        self.paths.ensure_dirs()

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  yolo_model={self.paths.yolo_model},\n"
            f"  confidence={self.detection.confidence},\n"
            f"  db={self.database.connection_string},\n"
            f"  sumo_binary={self.simulation.sumo_binary}\n"
            f")"
        )


# ── Module-level singleton (import and reuse) ─────────────────────────────────
config = Config()


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    cfg = Config()

    table = Table(title="📋 Active Configuration", show_header=True)
    table.add_column("Section",   style="cyan",  no_wrap=True)
    table.add_column("Parameter", style="white")
    table.add_column("Value",     style="green")

    table.add_row("Paths",      "YOLO model",         str(cfg.paths.yolo_model))
    table.add_row("Paths",      "Database",           str(cfg.paths.db_path))
    table.add_row("Paths",      "Logs",               str(cfg.paths.logs_dir))
    table.add_row("Detection",  "Confidence",         str(cfg.detection.confidence))
    table.add_row("Detection",  "Device",             cfg.detection.device)
    table.add_row("Detection",  "Vehicle classes",    str(cfg.detection.vehicle_classes))
    table.add_row("Density",    "LOW  threshold",     f"0 – {cfg.density.low_max}")
    table.add_row("Density",    "MED  threshold",     f"{cfg.density.low_max+1} – {cfg.density.medium_max}")
    table.add_row("Density",    "HIGH threshold",     f"{cfg.density.medium_max+1}+")
    table.add_row("Signal",     "GREEN (LOW)",        f"{cfg.signal.green_times['LOW']}s")
    table.add_row("Signal",     "GREEN (MEDIUM)",     f"{cfg.signal.green_times['MEDIUM']}s")
    table.add_row("Signal",     "GREEN (HIGH)",       f"{cfg.signal.green_times['HIGH']}s")
    table.add_row("Signal",     "YELLOW",             f"{cfg.signal.yellow_time}s")
    table.add_row("Simulation", "SUMO binary",        cfg.simulation.sumo_binary)
    table.add_row("Simulation", "Max steps",          str(cfg.simulation.max_steps))
    table.add_row("Database",   "Backend",            cfg.database.backend)
    table.add_row("Database",   "Connection",         cfg.database.connection_string)

    console.print(table)
    console.print("\n[bold green]✅ Config loaded successfully.[/bold green]")
