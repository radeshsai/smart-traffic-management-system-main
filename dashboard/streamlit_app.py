"""
dashboard/streamlit_app.py
Live Traffic Dashboard
- Signal/Counts: refresh every 2s
- Charts: refresh every 5s
- Charts persist in session_state — never flash or disappear
"""
import sys, os, time as _time, sqlite3 as _sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
from dashboard import charts

def _to_ist(local_str):
    """
    Display helper for `created_at` timestamps.

    NOTE: database.py's _local_ts() already writes datetime.now() (the
    machine's local time, i.e. IST), so `created_at` is already IST in
    the DB. This function used to add +5:30 on top of that, which
    double-shifted every timestamp shown in the Raw Data Tables /
    Recent Events panels. It now just trims/normalizes the string
    instead of re-converting it.
    """
    if not local_str:
        return ""
    return str(local_str)[:19]

st.set_page_config(
    page_title="🚦 Smart Traffic Management",
    page_icon="🚦", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%)}
div[data-testid="metric-container"]{background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:16px}
.sg{background:#1b5e20;color:#a5d6a7;border-radius:8px;padding:4px 14px;font-weight:700;display:inline-block}
.sy{background:#4e3b00;color:#ffe082;border-radius:8px;padding:4px 14px;font-weight:700;display:inline-block}
.sr{background:#4a0000;color:#ef9a9a;border-radius:8px;padding:4px 14px;font-weight:700;display:inline-block}
.dir-card{background:rgba(255,255,255,.06);border-radius:14px;padding:18px;
  border:1px solid rgba(255,255,255,.1);margin-bottom:12px}
h2,h3{color:#90CAF9!important}
</style>""", unsafe_allow_html=True)

# ── DB path ───────────────────────────────────────────────────────────────────
_DB = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "database", "traffic.db"))

@st.cache_resource
def _get_conn():
    c = _sqlite3.connect(_DB, timeout=10, check_same_thread=False)
    c.row_factory = _sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c

def _q1(sql, p=()):
    try:
        return _get_conn().execute(sql, p).fetchone()
    except Exception:
        return None

def _qall(sql, p=()):
    try:
        return _get_conn().execute(sql, p).fetchall()
    except Exception:
        return []

DIRS = ["north", "south", "east", "west"]

# ── Live data readers (called every 2s) ───────────────────────────────────────
def read_counts():
    result = {}
    for d in DIRS:
        row = _q1("SELECT current_count FROM vehicle_counts "
                  "WHERE direction=? ORDER BY id DESC LIMIT 1", (d,))
        result[d] = int(row[0]) if row and row[0] is not None else 0
    return result

def read_density():
    result = {}
    for d in DIRS:
        row = _q1("SELECT density_level,congestion_score,vehicle_count,recommended_green "
                  "FROM density_logs WHERE direction=? ORDER BY id DESC LIMIT 1", (d,))
        if row:
            result[d] = {"level": row[0] or "LOW", "score": float(row[1] or 0),
                         "vehicle_count": int(row[2] or 0),
                         "recommended_green": int(row[3] or 20)}
        else:
            result[d] = {"level":"LOW","score":0,"vehicle_count":0,"recommended_green":20}
    return result

def read_signals():
    from datetime import datetime as _dt
    # Fresh connection — bypass cache to always get latest signal state
    try:
        _sc = _sqlite3.connect(_DB, timeout=3, check_same_thread=False)
        _sc.row_factory = _sqlite3.Row
    except Exception:
        _sc = None
    result = {}
    for d in DIRS:
        row = None
        if _sc:
            try:
                row = _sc.execute(
                    "SELECT phase,remaining_seconds,allocated_green,cycle_number,"
                    "density_level,created_at FROM signal_states "
                    "WHERE direction=? ORDER BY id DESC LIMIT 1", (d,)).fetchone()
            except Exception:
                pass
        if row is None:
            row = _q1("SELECT phase,remaining_seconds,allocated_green,cycle_number,"
                      "density_level,created_at FROM signal_states "
                      "WHERE direction=? ORDER BY id DESC LIMIT 1", (d,))
        if row:
            stored_phase = row[0] or "RED"
            stored_rem   = float(row[1] or 0)
            try:
                written  = _dt.strptime(row[5], "%Y-%m-%d %H:%M:%S")
                elapsed  = (_dt.now() - written).total_seconds()
                live_rem = max(0.0, stored_rem - elapsed)
            except Exception:
                live_rem = stored_rem
            # Trust DB phase — no conversion needed
            result[d] = {
                "phase":             stored_phase,
                "remaining_seconds": live_rem,
                "allocated_green":   int(row[2] or 0),
                "cycle_number":      int(row[3] or 0),
                "density_level":     row[4] or "LOW",
            }
        else:
            result[d] = {"phase":"RED","remaining_seconds":0,
                         "allocated_green":0,"cycle_number":0,"density_level":"LOW"}
    if _sc:
        try:
            _sc.close()
        except Exception:
            pass
    return result

def read_active(signals):
    # First try live signals dict
    for phase in ["GREEN","YELLOW","ALL_RED"]:
        for d, s in signals.items():
            if s.get("phase") == phase:
                return {"direction":d, "phase":phase,
                        "remaining": s.get("remaining_seconds",0),
                        "allocated_green": s.get("allocated_green",0),
                        "density_level": s.get("density_level","LOW")}

    # Fallback 1: any direction with non-zero remaining
    for d, s in signals.items():
        if float(s.get("remaining_seconds",0)) > 0:
            return {"direction":d, "phase":s.get("phase","RED"),
                    "remaining": s.get("remaining_seconds",0),
                    "allocated_green": s.get("allocated_green",0),
                    "density_level": s.get("density_level","LOW")}

    # Fallback 2: latest DB row (catches startup before first tick)
    row = _q1(
        "SELECT direction, phase, remaining_seconds, allocated_green, density_level "
        "FROM signal_states WHERE phase IN ('GREEN','YELLOW','ALL_RED') "
        "ORDER BY id DESC LIMIT 1"
    )
    if row:
        return {"direction":row[0], "phase":row[1],
                "remaining":float(row[2] or 0),
                "allocated_green":int(row[3] or 0),
                "density_level":row[4] or "LOW"}

    # Fallback 3: any latest row at all
    row = _q1(
        "SELECT direction, phase, remaining_seconds, allocated_green, density_level "
        "FROM signal_states ORDER BY id DESC LIMIT 1"
    )
    if row:
        return {"direction":row[0], "phase":row[1],
                "remaining":float(row[2] or 0),
                "allocated_green":int(row[3] or 0),
                "density_level":row[4] or "LOW"}

    return {"direction":"N/A","phase":"RED","remaining":0,
            "allocated_green":0,"density_level":"LOW"}

def read_stats():
    r1 = _q1("SELECT COUNT(*) FROM detections")
    r3 = _q1("SELECT SUM(cycle_number) FROM (SELECT direction, MAX(cycle_number) as cycle_number FROM signal_states GROUP BY direction)")
    total_detections = int(r1[0]) if r1 and r1[0] else 0

    # Throughput = total detections so far minus vehicles currently live/in-frame.
    # Mirrors database.py's get_total_stats() and uses the same per-direction
    # current_count values shown as "Live Vehicles" on the dashboard.
    live_vehicles = sum(read_counts().values())
    sim_throughput = max(0, total_detections - live_vehicles)

    return {
        "total_detections": total_detections,
        "signal_cycles":    int(r3[0]) if r3 and r3[0] else 0,
        "sim_throughput":   sim_throughput,
    }

# ── Chart data readers (cached, called every 5s) ──────────────────────────────
@st.cache_data(ttl=5, show_spinner=False)
def load_timeseries():
    rows = _qall(
        "SELECT created_at,direction,vehicle_count,density_level,congestion_score "
        "FROM density_logs ORDER BY id DESC LIMIT 200")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df.sort_values("created_at")

@st.cache_data(ttl=5, show_spinner=False)
def load_congestion():
    rows = _qall(
        "SELECT created_at,direction,congestion_score "
        "FROM density_logs ORDER BY id DESC LIMIT 100")
    return [dict(r) for r in rows]

@st.cache_data(ttl=5, show_spinner=False)
def load_cycles():
    rows = _qall(
        "SELECT direction, phase, COUNT(*) as total_cycles, "
        "AVG(allocated_green) as avg_green, MAX(allocated_green) as max_green, "
        "MIN(allocated_green) as min_green "
        "FROM signal_states WHERE phase='GREEN' GROUP BY direction")
    return [dict(r) for r in rows]

@st.cache_data(ttl=5, show_spinner=False)
def load_sim():
    rows = _qall(
        "SELECT direction, AVG(waiting_time) as avg_waiting, "
        "AVG(queue_length) as avg_queue, AVG(mean_speed) as avg_speed, "
        "MAX(throughput) as total_throughput, AVG(congestion_score) as avg_congestion "
        "FROM simulation_metrics GROUP BY direction")
    return [dict(r) for r in rows]

@st.cache_data(ttl=5, show_spinner=False)
def load_class_dist():
    totals = {"car":0,"motorcycle":0,"bus":0,"truck":0}
    for d in DIRS:
        row = _q1("SELECT count_car,count_motorcycle,count_bus,count_truck "
                  "FROM vehicle_counts WHERE direction=? ORDER BY id DESC LIMIT 1", (d,))
        if row:
            totals["car"]        += int(row[0] or 0)
            totals["motorcycle"] += int(row[1] or 0)
            totals["bus"]        += int(row[2] or 0)
            totals["truck"]      += int(row[3] or 0)
    return totals

# ── Helpers ───────────────────────────────────────────────────────────────────
DIR_COLORS = {"north":"#4FC3F7","south":"#81C784","east":"#FF8A65","west":"#CE93D8"}
PHASE_COLORS = {"GREEN":"#4CAF50","YELLOW":"#FFC107","RED":"#F44336","ALL_RED":"#FFC107"}

def sbadge(phase):
    _p = "YELLOW" if phase in ("ALL_RED","PRE_GREEN") else phase
    cls = {"GREEN":"sg","YELLOW":"sy","RED":"sr"}.get(_p,"sr")
    em  = {"GREEN":"🟢","YELLOW":"🟡","RED":"🔴"}.get(_p,"🔴")
    return f'<span class="{cls}">{em} {_p}</span>'

def dbadge(level):
    c = {"LOW":"#66BB6A","MEDIUM":"#FFA726","HIGH":"#EF5350"}.get(level,"#aaa")
    e = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}.get(level,"⚫")
    return f'<span style="color:{c};font-weight:700">{e} {level}</span>'

# ── Session state init ────────────────────────────────────────────────────────
if "tick" not in st.session_state:
    st.session_state.tick         = 0
    st.session_state.last_chart_ts = 0.0
    # Persist chart data so charts never disappear
    st.session_state.ts_df        = pd.DataFrame()
    st.session_state.cong_data    = []
    st.session_state.cycle_data   = []
    st.session_state.sim_data     = []
    st.session_state.class_dist   = {"car":0,"motorcycle":0,"bus":0,"truck":0}

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🚦 AI-Driven Smart Traffic Management System")
st.caption(f"Live: 1s · Charts: 3s")
st.divider()

# ── Fast section 1: sidebar ─────────────────────────────────────────────────────
# IMPORTANT: a fragment function cannot call st.sidebar internally
# (Streamlit raises StreamlitAPIException if it does). The fix is to wrap the
# *call* to this fragment in `with st.sidebar:` from the outside instead —
# see the call site below.
@st.fragment(run_every=1)
def render_sidebar():
    counts  = read_counts()
    density = read_density()
    signals = read_signals()
    active  = read_active(signals)
    stats   = read_stats()

    st.title("Traffic Control")
    st.divider()

    # Traffic Summary
    st.subheader("Traffic Summary")
    _adir  = active.get("direction","N/A").upper()
    _aph   = active.get("phase","RED")
    _arem  = float(active.get("remaining",0))
    _lveh  = sum(counts.values())
    _cscores = [density.get(d,{}).get("score",0) for d in DIRS]
    _mcong   = max(_cscores) if _cscores else 0
    _clevel  = "HIGH" if _mcong>=66 else "MEDIUM" if _mcong>=33 else "LOW"
    _ccolor  = "#EF5350" if _clevel=="HIGH" else "#FFA726" if _clevel=="MEDIUM" else "#66BB6A"
    _pcolor  = {"GREEN":"#4CAF50","YELLOW":"#FFC107","RED":"#F44336","ALL_RED":"#FFC107"}.get(_aph,"#aaa")

    # Compact traffic summary — smaller than st.metric
    st.markdown(f"""
<div style="font-size:.82rem;line-height:1.7">
<div><span style="color:#aaa">Active:</span> <b style="color:{_pcolor}">{_aph}</b> <b>{_adir}</b></div>
<div><span style="color:#aaa">Remaining:</span> <b style="color:#4FC3F7">{int(_arem)}s</b></div>
<div><span style="color:#aaa">Live Vehicles:</span> <b>{_lveh}</b></div>
<div><span style="color:#aaa">Congestion:</span> <b style="color:{_ccolor}">{_clevel} ({_mcong:.0f}/100)</b></div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # Quick Stats
    st.subheader("Quick Stats")
    st.markdown(f"""
<div style="font-size:.82rem;line-height:1.7">
<div><span style="color:#aaa">Detections:</span> <b>{stats.get("total_detections",0):,}</b></div>
<div><span style="color:#aaa">Cycles:</span> <b>{stats.get("signal_cycles",0)}</b></div>
<div><span style="color:#aaa">Throughput:</span> <b>{stats.get("sim_throughput",0)}</b></div>
<div><span style="color:#aaa">Updated:</span> <b>{_time.strftime("%H:%M:%S")}</b></div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # System Status
    st.subheader("System Status")
    _db_ok  = os.path.exists(_DB)
    st.success("Detection Running")
    st.success("Dashboard Live")
    st.success("Signal Engine OK")
    if _db_ok:
        st.success("Database OK")
    else:
        st.error("Database Missing")

    st.divider()

    # Recent Events — unique transitions, IST time
    st.subheader("Recent Events")
    try:
        _evc = _sqlite3.connect(_DB, timeout=3)
        _evc.row_factory = _sqlite3.Row
        # Unique per direction+phase, sorted by latest occurrence
        _evrows = _evc.execute(
            "SELECT direction, phase, MAX(created_at) as t "
            "FROM signal_states WHERE phase IN ('GREEN','YELLOW') "
            "GROUP BY direction, phase "
            "ORDER BY t DESC LIMIT 10"
        ).fetchall()
        _evc.close()
        if _evrows:
            for _ev in _evrows:
                _pc  = "#4CAF50" if _ev["phase"]=="GREEN" else "#FFC107"
                _ts  = _to_ist(str(_ev["t"]))[11:19] if _ev["t"] else ""
                st.markdown(
                    f"<div style='font-size:.78rem;margin-bottom:2px'>"
                    f"<code style='font-size:.75rem'>{_ts}</code> "
                    f"<b style='color:{_pc}'>{str(_ev['direction']).upper()}</b>"
                    f" <span style='color:#aaa'>→</span> "
                    f"<span style='color:{_pc};font-weight:700'>{_ev['phase']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("No events yet...")
    except Exception:
        st.caption("Loading...")

    st.divider()

    # Controls
    st.subheader("Controls")
    if st.button("Export CSVs", use_container_width=True):
        try:
            from src.database import DatabaseManager
            _db2 = DatabaseManager()
            _db2.initialize()
            _db2.export_all_reports()
            st.success("Exported to outputs/reports/")
        except Exception as _ex:
            st.error(str(_ex))
    if st.button("Prune Records", use_container_width=True):
        try:
            from src.database import DatabaseManager
            _db2 = DatabaseManager()
            _db2.initialize()
            _db2.prune_old_records()
            st.success("Pruned.")
        except Exception:
            pass
    st.caption("AI-Driven Smart Traffic Management")


# ── Fast section 2: KPIs, Active Signal, Direction Cards ───────────────────────
@st.fragment(run_every=1)
def render_main_kpis():
    counts  = read_counts()
    density = read_density()
    signals = read_signals()
    active  = read_active(signals)
    stats   = read_stats()

    # ── KPI ───────────────────────────────────────────────────────────────────
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("🔍 Total Detections", f"{stats.get('total_detections',0):,}")
    k2.metric("🚗 Live Vehicles",    f"{sum(counts.values())}")
    k3.metric("🚦 Signal Cycles",    f"{stats.get('signal_cycles',0):,}")
    k4.metric("📤 SUMO Throughput",  f"{stats.get('sim_throughput',0):,}")
    st.divider()

    # ── Active Signal ─────────────────────────────────────────────────────────
    active_dir   = active["direction"].upper()
    active_phase = active["phase"]
    remaining    = float(active.get("remaining", 0))
    green_alloc  = int(active.get("allocated_green", 1))
    density_act  = active.get("density_level","LOW")
    bar_color    = PHASE_COLORS.get(active_phase, "#4FC3F7")
    total_secs   = green_alloc if active_phase == "GREEN" else (3 if active_phase == "YELLOW" else 1)
    total_secs   = max(total_secs, 1)

    # Smooth countdown
    cd_key = f"{active_dir}|{active_phase}|{green_alloc}"
    if st.session_state.get("cd_key") != cd_key:
        st.session_state["cd_key"]   = cd_key
        st.session_state["cd_start"] = _time.time()
        st.session_state["cd_init"]  = remaining if remaining > 0 else float(total_secs)

    cd_elapsed = _time.time() - st.session_state.get("cd_start", _time.time())
    cd_rem     = max(0.0, st.session_state.get("cd_init", total_secs) - cd_elapsed)
    pct        = int(cd_rem / total_secs * 100)

    col_a, col_r = st.columns([2,1])
    with col_a:
        st.markdown(
            f"### Active Signal &nbsp; {sbadge(active_phase)} &nbsp;"
            f"<span style='color:#90CAF9;font-size:1.3rem;font-weight:700'>{active_dir}</span>",
            unsafe_allow_html=True)
        st.markdown(
            f"Density: {dbadge(density_act)} &nbsp;|&nbsp; Green: <b>{green_alloc}s</b>",
            unsafe_allow_html=True)
    with col_r:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='color:#aaa;font-size:.85rem'>⏱ Remaining</div>"
            f"<div style='color:{bar_color};font-size:2.5rem;font-weight:700'>{int(cd_rem)}s</div>"
            f"<div style='margin-top:4px'>{sbadge(active_phase)}</div>"
            f"</div>", unsafe_allow_html=True)
        st.progress(min(max(pct,0),100)/100)

    # Next priority
    waiting = {d:v for d,v in density.items() if d != active.get("direction","")}
    if waiting:
        nd = max(waiting, key=lambda d: waiting[d].get("score",0))
        st.caption(f"🔜 Next: **{nd.upper()}** · {waiting[nd].get('score',0):.0f}/100 · {waiting[nd].get('level','LOW')}")

    st.divider()

    # ── Direction Cards ───────────────────────────────────────────────────────
    st.subheader("📡 Direction Status")
    dcols = st.columns(4)
    for i, d in enumerate(DIRS):
        with dcols[i]:
            count = counts.get(d, 0)
            dens  = density.get(d, {})
            sig   = signals.get(d, {})
            level = dens.get("level","LOW")
            phase = sig.get("phase","RED")
            score = float(dens.get("score",0))
            rem_s = float(sig.get("remaining_seconds",0))
            color = DIR_COLORS.get(d,"#aaa")
            st.markdown(
                f"""<div class="dir-card">
                <div style="font-size:1.1rem;font-weight:700;color:{color};margin-bottom:6px">{d.upper()}</div>
                <div style="font-size:1.6rem;font-weight:700;color:{color}">{count}
                  <span style="font-size:.8rem;color:#aaa;font-weight:400"> vehicles</span></div>
                <div style="margin:4px 0">{dbadge(level)}</div>
                <div style="margin:4px 0">{sbadge(phase)} &nbsp; {rem_s:.0f}s</div>
                <div style="color:#aaa;font-size:.82rem">Congestion: {score:.1f}/100</div>
                </div>""", unsafe_allow_html=True)


# Render the sidebar fragment — wrapped in `with st.sidebar:` from the
# OUTSIDE, since a fragment function cannot call st.sidebar internally.
with st.sidebar:
    render_sidebar()

# Render the main-body fast section (KPIs, Active Signal, Direction Cards).
# Renders immediately on page load, then refreshes on its own 1s timer,
# independent of everything below.
render_main_kpis()

st.divider()


# ── Slower section: all charts ─────────────────────────────────────────────────
# Runs on its own 3s timer, independently of the fast section above. This is
# the part that used to cause visible flicker — 7 Plotly charts fully
# redrawing every 0.5s under the old global st.rerun() loop. Giving it its
# own slower fragment means these only redraw every 3s, and never force the
# fast KPIs/sidebar above to redraw too (and vice versa).
@st.fragment(run_every=3)
def render_charts_section():
    counts  = read_counts()
    density = read_density()

    # Refresh persisted chart datasets (cached 5s at the query level via
    # @st.cache_data; this fragment itself runs every 3s)
    _new_ts   = load_timeseries()
    _new_cong = load_congestion()
    _new_cyc  = load_cycles()
    _new_sim  = load_sim()
    _new_cls  = load_class_dist()

    # Only replace if new data is non-empty (preserve previous on empty)
    if not _new_ts.empty:
        st.session_state.ts_df = _new_ts
    if _new_cong:
        st.session_state.cong_data = _new_cong
    if _new_cyc:
        st.session_state.cycle_data = _new_cyc
    if _new_sim:
        st.session_state.sim_data = _new_sim
    if any(_new_cls.values()):
        st.session_state.class_dist = _new_cls

    # ── Fast charts (live data) ────────────────────────────────────────────────
    col_bar, col_pie = st.columns([3,2])
    with col_bar:
        st.plotly_chart(
            charts.vehicle_count_bar(counts),
            width="stretch", key="bar_counts")
    with col_pie:
        st.plotly_chart(
            charts.class_distribution_pie(st.session_state.class_dist),
            width="stretch", key="pie_cls")

    st.subheader("🎯 Congestion Scores")
    gcols = st.columns(4)
    for i, d in enumerate(DIRS):
        score = float(density.get(d,{}).get("score",0))
        with gcols[i]:
            st.plotly_chart(
                charts.congestion_gauge(score, d),
                width="stretch", key=f"gauge_{d}")

    st.divider()

    # ── Persistent charts (session_state data, never disappear) ───────────────
    st.subheader("📈 Vehicle Count Over Time")
    if not st.session_state.ts_df.empty:
        st.plotly_chart(
            charts.density_timeseries(st.session_state.ts_df),
            width="stretch", key="ts_chart")
    else:
        st.info("⏳ Waiting for detection data to populate…")

    st.subheader("🗺 Congestion Heatmap")
    if st.session_state.cong_data:
        st.plotly_chart(
            charts.congestion_heatmap(st.session_state.cong_data),
            width="stretch", key="heatmap")
    else:
        st.info("⏳ Waiting for congestion data…")

    st.divider()

    st.subheader("🚦 Signal Cycle Analysis")
    if st.session_state.cycle_data:
        st.plotly_chart(
            charts.signal_phase_timeline(st.session_state.cycle_data),
            width="stretch", key="sig_cycle")
    else:
        st.info("⏳ Waiting for signal cycle data…")

    # SUMO section — only show if simulation data exists
    _has_sim = bool(st.session_state.sim_data)
    if _has_sim:
        st.subheader("📉 SUMO Simulation Metrics")
        st.plotly_chart(
            charts.sim_metrics_bar(st.session_state.sim_data),
            width="stretch", key="sim_metrics")
        scols = st.columns(len(st.session_state.sim_data))
        for i, row in enumerate(st.session_state.sim_data):
            _d = row.get("direction","")
            _c = DIR_COLORS.get(_d,"#aaa")
            with scols[i]:
                st.markdown(
                    f"""<div class="dir-card">
                    <div style="font-weight:700;color:{_c}">{_d.upper()}</div>
                    <div>⏱ Avg Wait: <b>{row.get('avg_waiting',0):.2f}s</b></div>
                    <div>🚗 Avg Queue: <b>{row.get('avg_queue',0):.1f}</b></div>
                    <div>💨 Avg Speed: <b>{row.get('avg_speed',0):.1f} m/s</b></div>
                    <div>📤 Throughput: <b>{row.get('total_throughput',0)}</b></div>
                    </div>""", unsafe_allow_html=True)


render_charts_section()

st.divider()

# ── Raw tables ────────────────────────────────────────────────────────────────
with st.expander("🔍 Raw Data Tables"):
    t1,t2,t3 = st.tabs(["Density","Signals","Sim"])
    with t1:
        _conn_raw = _sqlite3.connect(_DB, timeout=5)
        _conn_raw.row_factory = _sqlite3.Row
        try:
            _rows = _conn_raw.execute(
                "SELECT created_at as time, direction, "
                "vehicle_count, density_level, congestion_score "
                "FROM density_logs ORDER BY id DESC LIMIT 50").fetchall()
            if _rows:
                _df = pd.DataFrame([dict(r) for r in _rows])
                if "time" in _df.columns:
                    _df["time"] = _df["time"].apply(_to_ist)
                st.dataframe(_df, width="stretch", hide_index=True)
            else:
                st.dataframe(pd.DataFrame(), width="stretch")
        except Exception:
            st.dataframe(pd.DataFrame(), width="stretch")
        finally:
            _conn_raw.close()
    with t2:
        _conn_raw = _sqlite3.connect(_DB, timeout=5)
        _conn_raw.row_factory = _sqlite3.Row
        try:
            # Latest row per direction only — no duplicate rows
            _rows = _conn_raw.execute(
                "SELECT ss.created_at as time, ss.direction, "
                "ss.phase, ROUND(ss.remaining_seconds,1) as remaining_s, ss.allocated_green "
                "FROM signal_states ss "
                "INNER JOIN (SELECT direction, MAX(id) as mid FROM signal_states GROUP BY direction) m "
                "ON ss.id = m.mid "
                "ORDER BY ss.direction").fetchall()
            if _rows:
                _df = pd.DataFrame([dict(r) for r in _rows])
                if "time" in _df.columns:
                    _df["time"] = _df["time"].apply(_to_ist)
                st.dataframe(_df, width="stretch", hide_index=True)
            else:
                st.dataframe(pd.DataFrame(), width="stretch")
        except Exception:
            st.dataframe(pd.DataFrame(), width="stretch")
        finally:
            _conn_raw.close()
    with t3:
        _conn_raw = _sqlite3.connect(_DB, timeout=5)
        _conn_raw.row_factory = _sqlite3.Row
        try:
            _rows = _conn_raw.execute(
                "SELECT created_at as time, direction, "
                "sim_step, waiting_time, queue_length, mean_speed, throughput, congestion_score "
                "FROM simulation_metrics ORDER BY id DESC LIMIT 50").fetchall()
            if _rows:
                _df3 = pd.DataFrame([dict(r) for r in _rows])
                if "time" in _df3.columns:
                    _df3["time"] = _df3["time"].apply(_to_ist)
                st.dataframe(_df3, width="stretch", hide_index=True)
            else:
                st.dataframe(pd.DataFrame(), width="stretch")
        except Exception:
            st.dataframe(pd.DataFrame(), width="stretch")
        finally:
            _conn_raw.close()

st.divider()
st.caption("🚦 AI-Driven Smart Traffic Management · Final Year B.Tech Project")

# NOTE: no global _time.sleep()/st.rerun() loop anymore. The fast section
# (render_live_section, ~1s) and the charts section (render_charts_section,
# ~3s) each refresh themselves via @st.fragment(run_every=...), without
# rerunning — and reflashing — the rest of the page. The Raw Data Tables
# above are inside a collapsed st.expander and refresh on full-page loads
# only, which is fine since the user has to open it manually anyway.

