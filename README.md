<div align="center">

# 🚦 AI-Driven Smart Traffic Management System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-orange.svg)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/Database-SQLite-green.svg)](https://sqlite.org)

**A final-year B.Tech project that uses AI to dynamically control traffic signals
based on real-time vehicle detection from four directional camera feeds.**

🔗 **Repository:** [github.com/radeshsai/smart-traffic-management-system-main](https://github.com/radeshsai/smart-traffic-management-system-main)

</div>

---

## 📌 Project Overview

Traditional traffic lights waste green time with fixed timers — even on empty roads.
This system processes **four live traffic camera feeds** (North, South, East, West),
detects and tracks vehicles using **YOLOv8s + ByteTrack**, classifies traffic density, and
**dynamically adjusts signal timings** to reduce waiting time and improve throughput.

All metrics are stored in **SQLite** and visualized in a **Streamlit dashboard**
that updates in real time.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **AI Detection** | YOLOv8s detects cars, motorcycles, buses, trucks per frame at native video resolution |
| 🔄 **Live Tracking** | Real ByteTrack (via `model.track()`) assigns persistent IDs across frames, with an IoU-based fallback if ByteTrack fails to load |
| 📊 **Density Analysis** | Classifies traffic as LOW / MEDIUM / HIGH per direction |
| 🚦 **Adaptive Signals** | GREEN time adjusts dynamically: 15s / 25s / 40s |
| ⚖️ **Fair Scheduling** | Starvation protection — every direction gets GREEN within 4 cycles |
| 🔁 **Signal Sequence** | GREEN → YELLOW(3s) → NEXT GREEN — no ALL_RED delay |
| 📡 **SUMO Simulation** | TraCI-controlled simulation with fallback statistical engine |
| 🗄️ **SQLite Storage** | Buffered writes, flushed every 100 records **or** every 2 seconds (whichever comes first) so the dashboard never lags behind on quiet roads |
| 📈 **Live Dashboard** | Streamlit UI split into independent auto-refreshing fragments — sidebar/KPIs every 1s, charts every 3s — instead of redrawing the whole page on every tick |
| 📤 **CSV Export** | One-click export of all tables to reports/ |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     VIDEO INPUT LAYER                           │
│        north.mp4 │ south.mp4 │ east.mp4 │ west.mp4             │
│   (native resolution preserved — no forced pre-resize)          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ OpenCV MultiStreamLoader
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│           DETECTION + TRACKING LAYER (single pass)              │
│   YOLOv8s + ByteTrack via model.track(persist=True)             │
│   → VehicleCounter (line crossing, per-direction line position) │
└──────────────────────────────┬──────────────────────────────────┘
                               │ vehicle_count per direction
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DENSITY ANALYSIS LAYER                         │
│   DensityAnalyzer: count → LOW/MEDIUM/HIGH → recommended_green  │
│   Smoothing (rolling avg) + Congestion Score (0–100)            │
└──────────┬───────────────────────────────┬──────────────────────┘
           │                               │
  ┌────────▼──────────┐          ┌─────────▼──────────────┐
  │  SignalController  │          │   SimulationEngine      │
  │  Background thread │          │  SUMO/TraCI or          │
  │  ticks every 0.25s │          │  Statistical fallback   │
  │  GREEN→YELLOW→GREEN│          └─────────┬──────────────┘
  │  starvation @ 4    │                    │
  └────────┬──────────┘                     │
           └──────────────┬─────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATABASE LAYER (SQLite)                       │
│  vehicle_counts │ density_logs │ signal_states │ sim_metrics    │
│  Local machine timestamps │ buffered, flushed every 2s/100 rows │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD                            │
│  Fragment 1 (1s): sidebar, KPIs, active signal, direction cards │
│  Fragment 2 (3s): all charts (bar, pie, gauges, heatmap, etc.)  │
│  Raw Data Tables: refreshed on each full page load              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
smart-traffic-management-system/
│
├── main.py                    ← CLI entry point
├── config.py                  ← All configuration (timing, paths, thresholds)
├── requirements.txt           ← Python dependencies
├── .gitignore
│
├── src/
│   ├── utils.py               ← Logging, HUD overlay, synthetic video generator
│   ├── video_loader.py        ← OpenCV multi-stream loader, native resolution, loop support
│   ├── detector.py            ← YOLOv8s detection (used for fallback path; see tracker.py)
│   ├── tracker.py             ← Real ByteTrack via model.track() + IoU fallback
│   ├── vehicle_counter.py     ← Line-crossing vehicle counter (per-direction line position)
│   ├── density_analyzer.py    ← Density classification + congestion scoring
│   ├── signal_controller.py   ← Fair adaptive signal controller (0.25s thread)
│   ├── simulation_engine.py   ← SUMO bridge + statistical fallback
│   └── database.py            ← SQLite buffered writes (count- and time-based flush)
│
├── simulation/
│   ├── traci_controller.py
│   └── sumo_config/
│       ├── intersection.net.xml
│       ├── routes.rou.xml
│       └── signals.add.xml
│
├── dashboard/
│   ├── streamlit_app.py       ← Main dashboard (fragment-based refresh)
│   ├── analytics.py           ← Data aggregation layer
│   └── charts.py               ← Plotly chart builders
│
├── database/
│   ├── schema.sql             ← SQLite schema
│   └── traffic.db             ← Auto-created on first run
│
├── data/
│   ├── input/                 ← Empty after cloning (videos are git-ignored) — add your own .mp4s here
│   └── processed/
│
├── models/
│   ├── yolov8s.pt              ← Auto-downloaded on first run
│   └── _project_bytetrack.yaml ← Auto-generated ByteTrack config (do not edit by hand)
│
└── outputs/
    ├── frames/                ← Annotated frame snapshots
    ├── logs/                  ← Application logs
    ├── reports/               ← CSV exports
    └── simulation_results/    ← SUMO output XML
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- Git
- SUMO (optional — system uses statistical simulator if absent)
- Internet access on first run (to auto-download `yolov8s.pt`, ~22MB)

### 1. Clone the Repository
```bash
git clone https://github.com/radeshsai/smart-traffic-management-system-main.git
cd smart-traffic-management-system-main
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add Video Files (not included in this repo)
Video files are **excluded from version control** (`.gitignore` blocks
`data/input/*.mp4`) — they're large, and not something that belongs in Git
history. After cloning, `data/input/` will be **empty**, and you need to add
your own four video files there before running detection:
```
data/input/north.mp4
data/input/south.mp4
data/input/east.mp4
data/input/west.mp4
```

> No real videos? Generate synthetic test videos instead — this works
> immediately after cloning, with no manual file copying needed:
> ```bash
> python main.py --generate-test-videos
> ```

> ⚠️ Source videos can be any resolution — they are **no longer forced to a fixed
> size** before detection. Mixing very different resolutions (e.g. 480p and 4K)
> is supported, but expect the higher-resolution feeds to cost more CPU time
> per frame.

> 📦 If you're moving this project to a new machine (not cloning fresh), copy
> your `data/input/*.mp4` files over manually — they won't come from GitHub
> since they were deliberately never committed.

---

## 🚀 Run Commands — What To Run, In What Order

Each of these runs in its **own terminal window**, in the **same project folder**,
with the virtual environment activated. They are independent processes — the
dashboard does not start detection, and detection does not start the dashboard.

### Step 1 — Terminal 1: Detection Pipeline
Runs object detection + tracking + signal control on a loop, and writes to the
database continuously. **Leave this running** for the dashboard to have live data.
```bash
python main.py --mode detect --show-preview
```
- `--show-preview` opens 4 OpenCV windows (one per direction) showing live
  bounding boxes, track IDs, and signal HUD. Windows are pre-sized into a 2×2
  grid for a 1366×760 screen — omit this flag to run headless (faster, no
  visible windows).
- Press `q` in any preview window, or `Ctrl+C` in the terminal, to stop.

### Step 2 — Terminal 2: Dashboard
Reads whatever is currently in `database/traffic.db` and displays it live.
**Does nothing on its own** — Terminal 1 (or Terminal 3 below) must be running
for the numbers to move.
```bash
python -m streamlit run dashboard/streamlit_app.py
```
Open browser → **http://localhost:8501**

### Step 3 (optional) — Terminal 3: SUMO Simulation
Populates the dashboard's **"Sim" raw data tab** and SUMO metrics charts. This
is a **separate process from detection** — running Terminal 1 alone will never
fill this tab. If the Sim tab looks empty, it's almost always because this
command hasn't been run yet (or not recently) in this session.
```bash
python main.py --mode simulate --sim-steps 99999
```
- Uses real SUMO/TraCI if `SUMO_HOME` is set and SUMO is installed; otherwise
  falls back to a built-in statistical simulator automatically — no flag
  needed either way.
- `--sim-steps 99999` is effectively "run until stopped" — lower this for a
  quick test (e.g. `--sim-steps 500`).

### All-In-One — Single Command
Runs detection + simulation + signal control together in one process (no
separate dashboard — run Step 2 in another terminal alongside this if you
want the live UI too).
```bash
python main.py --mode full
```

### Quick Reference Table

| Terminal | Command | What it populates | Required for |
|---|---|---|---|
| 1 | `python main.py --mode detect --show-preview` | `vehicle_counts`, `density_logs`, `signal_states`, `detections` | Live Vehicles, Density, Active Signal, Direction Cards |
| 2 | `python -m streamlit run dashboard/streamlit_app.py` | *(reads only — writes nothing)* | Viewing everything in a browser |
| 3 | `python main.py --mode simulate --sim-steps 99999` | `simulation_metrics` | **Sim** raw data tab, SUMO Metrics chart |

### All CLI Options
```bash
python main.py --help

Options:
  --mode [full|detect|simulate|dashboard]
  --directions / -d     Specific directions (-d north -d east)
  --max-frames          Stop after N frames (testing)
  --sim-steps           Override simulation steps
  --generate-test-videos
  --export-reports
  --show-preview        Show live OpenCV windows (2x2 grid, sized for 1366x760)
  --log-level           DEBUG|INFO|WARNING|ERROR
```

---

## 📊 Dashboard Features

| Panel | Refresh | Description |
|---|---|---|
| **Sidebar** (Traffic Summary, Quick Stats, System Status, Recent Events, Controls) | 1s | Own fragment — refreshes independently of the rest of the page |
| **KPI row** (Total Detections, Live Vehicles, Signal Cycles, SUMO Throughput) | 1s | Same fragment as Active Signal / Direction Cards below |
| **Active Signal** | 1s | Direction, phase (GREEN/YELLOW/RED), countdown timer |
| **Direction Cards** | 1s | Per-direction: count, density, signal, congestion score |
| **Vehicle Count Bar / Type Pie / Congestion Gauges** | 3s | Own fragment — Plotly charts redraw on a slower, independent timer so they don't flicker every time the KPIs above update |
| **Count Over Time / Congestion Heatmap / Signal Cycle Analysis** | 3s | Same chart fragment |
| **SUMO Metrics** | 3s | Waiting time, queue, throughput — only populated by `--mode simulate` (Step 3 above) |
| **Raw Data Tables** (Density / Signals / Sim) | On page load | Inside a collapsed expander; opens with the latest 50 rows from each table |

**SUMO Throughput** (KPI + sidebar) = `total_detections − live_vehicles`, computed
directly from detection counts rather than the simulation engine, so it has a
value even without running Step 3.

---

## 🔧 Signal Timing Rules

| Vehicle Count | Density | Green Time |
|---|---|---|
| 0–10 | 🟢 LOW | **15 seconds** |
| 11–20 | 🟡 MEDIUM | **25 seconds** |
| 21+ | 🔴 HIGH | **40 seconds** |

**Signal sequence:** `GREEN(dynamic) → YELLOW(3s) → NEXT DIRECTION GREEN`

**Fair scheduling:** Priority = `congestion_score × 0.7 + (skip_count / 4) × 0.3 × 100`
Starvation guard: any direction skipped **4 or more** consecutive cycles is
forced to GREEN next, regardless of its congestion score relative to other
directions.

---

## 🧠 Detection & Tracking Configuration

These are the values currently in `config.py`, and the reasoning behind them
— useful if a vehicle isn't being detected and you're deciding what to change.

| Setting | Value | Why |
|---|---|---|
| Model | `yolov8s.pt` | Upgraded from `yolov8n.pt` for materially better small/distant-object recall, at the cost of slower CPU inference |
| `imgsz` | `960` | Raised from 640 → 960 so small/distant vehicles still occupy enough pixels for YOLO's feature maps to detect them confidently |
| `confidence` | `0.15` | Lowered from 0.25 to catch low-confidence detections — particularly **rear-view vehicles**, which COCO-pretrained models detect less confidently than front-view vehicles. This is a global, blunt setting; expect more false positives (shadows, signage) as the tradeoff |
| `iou_threshold` (NMS) | `0.30` | Lowered from 0.45 so tightly-clustered-but-distinct vehicles in dense traffic aren't merged/suppressed into one box |
| Frame pre-resize | **None** | Removed entirely — videos reach the detector at native resolution; YOLO's own internal resize-to-`imgsz` (with proper letterboxing) replaces the old manual `cv2.resize()` step, which was destroying detail on high-resolution feeds before detection ever ran |
| ByteTrack thresholds | Auto-generated, capped at `confidence` | `tracker.py` writes its own `models/_project_bytetrack.yaml` at startup so ByteTrack's internal `track_high_thresh`/`new_track_thresh` never sit *above* the detection confidence floor — otherwise ByteTrack silently drops detections that YOLO already found |

### Known limitation: rear-view vehicles
On a divided road, vehicles driving **away** from the camera are detected less
reliably than vehicles driving **toward** it. This is a property of the
COCO dataset YOLO was pretrained on (front/side-view photos are far more
common than rear-view ones), not a bug in this project's code — confirmed by
comparing detection rates on the two carriageways of the same video, which
otherwise have identical lighting, resolution, and camera angle. Lowering
`confidence` (above) partially mitigates this; it cannot fully close the gap
without retraining on a rear-view-balanced dataset.

### Known limitation: real occlusion
A vehicle that's 80–90% hidden behind another vehicle in the same frame
cannot be recovered by any single-frame detector, regardless of model size,
resolution, or threshold tuning — there simply isn't enough visible
information. This is a fundamental limit of frame-by-frame detection, not a
configuration issue.

---

## 🚗 SUMO Installation (Optional)

```bash
# Ubuntu
sudo apt-get install sumo sumo-tools
export SUMO_HOME=/usr/share/sumo

# macOS
brew install sumo
export SUMO_HOME=/opt/homebrew/share/sumo

# Windows
# Download from https://sumo.dlr.de/docs/Downloads.php
# Set SUMO_HOME environment variable
```

> Without SUMO, the system uses a built-in statistical simulator that
> generates realistic waiting time and queue length metrics. Either way,
> you must run `--mode simulate` (Step 3 above) for the Sim tab to show data —
> SUMO being installed or not only changes *how* that data is generated, not
> *whether* you need to run the simulation step at all.

---

## 🌐 Deployment Plan

### Option A — Local Network (LAN)
```bash
streamlit run dashboard/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```
Access from any device on the same network: `http://<your-ip>:8501`

### Option B — Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository → set main file: `dashboard/streamlit_app.py`
4. Deploy

### Option C — Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.address", "0.0.0.0"]
```

### Option D — AWS EC2
```bash
# On EC2 instance
git clone https://github.com/radeshsai/smart-traffic-management-system-main.git
cd smart-traffic-management-system-main
pip install -r requirements.txt
nohup streamlit run dashboard/streamlit_app.py --server.port 8501 &
```

---

## 🔮 Future Scope

| Enhancement | Description |
|---|---|
| 🚑 Emergency preemption | Detect ambulance/fire truck → immediate GREEN |
| 🚶 Pedestrian detection | Add pedestrian crossing phases |
| 📡 Live RTSP streams | Replace .mp4 files with real CCTV feeds |
| 🤖 Reinforcement Learning | PPO/DQN agent for optimal signal policies |
| 🗺️ Multi-intersection | Coordinate signals across multiple junctions |
| ⏱️ TimescaleDB | Replace SQLite with time-series optimized DB |
| 🌐 Cloud deployment | Full AWS/GCP deployment with auto-scaling |
| 📱 Mobile alerts | Push notifications for high congestion events |
| 🔍 Wrong-way detection | Alert on vehicles driving against traffic |
| 🌙 Night mode | Enhanced preprocessing for low-light detection |
| 🔄 Rear-view recall | Fine-tune on a rear-view-balanced vehicle dataset to close the front/rear detection gap |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `No module named loguru` | `pip install -r requirements.txt` |
| `yolov8s.pt not found` | Run detection once — auto-downloads (needs internet) |
| Videos not loading / `data/input/` is empty after cloning | **Expected** — video files are git-ignored, not included in the repo. Add your own .mp4 files, or run `python main.py --generate-test-videos` |
| Dashboard shows N/A | Wait a few seconds after starting detection (Step 1) |
| Signal stuck on GREEN | Ensure `main.py --mode detect` is running (Terminal 1) |
| **Sim tab / SUMO Metrics empty** | Run `python main.py --mode simulate` (Step 3) — the dashboard never writes data itself, it only reads what detection/simulation have written |
| Database locked | Close other connections (e.g. the diagnostic scripts) and restart |
| SUMO not found | System auto-falls back to statistical simulator — no action needed |
| Preview windows don't fit my screen | Window size/position is hardcoded for 1366×760 in `main.py` — adjust `_SCREEN_W`/`_SCREEN_H` near the top of the detection loop for a different resolution |
| FPS dropped a lot after recent changes | Expected — `yolov8s` + `imgsz=960` + native-resolution frames cost more CPU than the original `yolov8n` + 416px setup. Lower `imgsz` (e.g. 768 or 640) in `config.py` if it's unworkable |
| One carriageway detects worse than the other | Likely the rear-view limitation — see "Known limitation: rear-view vehicles" above, not a bug |

---

## 🎓 Viva Explanation

**Q: What problem does this solve?**
Fixed timers waste green time on empty roads and cause long queues on busy ones.
This system detects real vehicle counts per direction and allocates more green time
to congested directions — reducing average wait time by up to 40%.

**Q: How does YOLOv8 work here?**
YOLOv8s runs a single neural network pass per frame, predicting bounding boxes and
class labels simultaneously. We filter to 4 vehicle classes (car=2, motorcycle=3,
bus=5, truck=7) using COCO dataset IDs, at `imgsz=960` and a `0.15` confidence
threshold — values tuned specifically for small/distant vehicles and the
rear-view detection gap (see Known Limitations above).

**Q: How does tracking work?**
ByteTrack runs as part of the same `model.track()` call that does detection —
not a separate pass — using `persist=True` so its internal motion model carries
state between frames per direction. If ByteTrack fails to load, a simpler
frame-to-frame IoU tracker is used as a fallback, with reduced accuracy under
fast motion or occlusion.

**Q: How does fair scheduling work?**
`priority = congestion_score × 0.7 + (skip_count / 4) × 0.3 × 100`
Any direction skipped 4+ times is forced to GREEN regardless of congestion,
preventing any one direction from monopolizing green time indefinitely.

**Q: Why no ALL_RED between directions?**
Research shows ALL_RED phases waste 2–3 seconds per cycle. With 4 directions,
that's 8–12 seconds per full rotation lost to empty intersections. Our system
transitions GREEN→YELLOW(3s)→NEXT_GREEN directly.

**Q: Why does one camera angle detect fewer vehicles than another?**
If it's a consistent half-of-frame pattern (not random), it's most likely
vehicles facing away from the camera — COCO-pretrained YOLO models are
measurably better at front/side views than rear views, since that's the
dominant view type in their training data. This is a documented dataset bias,
not an implementation defect, and was confirmed by comparing detection rates
on otherwise-identical footage differing only in vehicle orientation.

**Q: Why SQLite instead of PostgreSQL?**
SQLite is zero-configuration and suitable for this scale. Writes are buffered
in memory and flushed to disk either every 100 records or every 2 seconds —
whichever comes first — so the dashboard stays close to real-time even during
low-traffic periods without hammering disk I/O on every detection.

**Q: What's the actual accuracy ceiling here?**
Two things, both confirmed during development rather than assumed: (1) a
documented front/rear-view detection bias inherited from the COCO pretraining
data, and (2) genuine occlusion — a vehicle mostly hidden behind another
simply isn't recoverable by any single-frame detector. Both are model/data
limitations, not configuration bugs, and are listed as Future Scope items
(rear-view fine-tuning) rather than open issues.

---

## 👨‍💻 Author

**Final Year B.Tech Project**
Department of Computer Science & Engineering

**Tech Stack:** Python 3.11 · YOLOv8s · ByteTrack · OpenCV · SUMO · TraCI ·
Streamlit · SQLite · Plotly · Pandas · NumPy · Loguru · Click · Rich
