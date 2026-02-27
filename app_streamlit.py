"""
YOLO CV Suite — Production Streamlit UI
Connects directly to yolo_backend.py inference functions.
Run with: streamlit run app_streamlit.py
"""

import cv2
import time
import tempfile
import os
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ─── PAGE CONFIG (must be first) ──────────────────────────────
st.set_page_config(
    page_title="YOLO CV Suite",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── INJECT CUSTOM CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root tokens ── */
:root {
  --accent:  #00d2c8;
  --accent2: #0085ff;
  --red:     #ff4f6b;
  --green:   #00e09a;
  --yellow:  #ffb800;
}

/* ── Global ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: #080c10 !important;
  color: #e4eaf2 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 1rem !important; max-width: 100% !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2a38; border-radius: 99px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #0d1117 !important;
  border-right: 1px solid rgba(255,255,255,.07) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Sidebar selectbox / radio / slider labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label {
  color: #5a6a7d !important;
  font-size: 0.7rem !important;
  font-family: 'JetBrains Mono', monospace !important;
  letter-spacing: .08em;
  text-transform: uppercase;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
  background: #111820 !important;
  border: 1px solid rgba(255,255,255,.07) !important;
  border-radius: 8px !important;
  color: #e4eaf2 !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectbox"] > div > div:hover { border-color: rgba(0,210,200,.35) !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
  background: linear-gradient(to right, #00d2c8, #00d2c8) !important;
}
[data-testid="stSlider"] div[role="slider"] {
  background: #00d2c8 !important;
  box-shadow: 0 0 10px rgba(0,210,200,.5) !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div {
  background: #111820 !important;
  border: 1px solid rgba(255,255,255,.07) !important;
  border-radius: 8px !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
  background: rgba(0,210,200,.15) !important;
  border: 1px solid rgba(0,210,200,.3) !important;
  color: #00d2c8 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .68rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: #111820 !important;
  border: 1.5px dashed rgba(255,255,255,.1) !important;
  border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(0,210,200,.35) !important; }

/* ── Toggle / Checkbox ── */
[data-testid="stCheckbox"] > label > div[role="checkbox"] {
  background: #00d2c8 !important;
  border-color: #00d2c8 !important;
}

/* ── Primary button (Run) ── */
[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, #00d2c8, #0085ff) !important;
  border: none !important;
  border-radius: 10px !important;
  color: #fff !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: .9rem !important;
  letter-spacing: .04em !important;
  padding: 0.65rem 1.2rem !important;
  width: 100% !important;
  transition: all .22s !important;
  box-shadow: 0 4px 24px rgba(0,210,200,.2) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 32px rgba(0,210,200,.35) !important;
}

/* ── Secondary button (Stop) ── */
[data-testid="stButton"] > button:not([kind="primary"]) {
  background: rgba(255,79,107,.1) !important;
  border: 1px solid rgba(255,79,107,.3) !important;
  border-radius: 10px !important;
  color: #ff4f6b !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  width: 100% !important;
}

/* ── Metric cards (custom) ── */
.cv-card {
  background: #0d1117;
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 12px;
  padding: 14px 18px;
  transition: border-color .2s;
}
.cv-card:hover { border-color: rgba(0,210,200,.35); }
.cv-card-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem;
  color: #5a6a7d;
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 5px;
}
.cv-card-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.5rem;
  font-weight: 800;
  line-height: 1;
}

/* ── Nav bar (fake top bar) ── */
.cv-nav {
  background: rgba(8,12,16,.9);
  backdrop-filter: blur(18px);
  border-bottom: 1px solid rgba(255,255,255,.07);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 14px;
  margin: -1rem -1rem 1.5rem -1rem;
}
.cv-nav-logo {
  width: 32px; height: 32px;
  background: linear-gradient(135deg, #00d2c8, #0085ff);
  border-radius: 8px;
  display: inline-flex;
  align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif;
  font-weight: 800; font-size: 15px;
  color: #fff;
  box-shadow: 0 0 18px rgba(0,210,200,.3);
}
.cv-nav-title {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 1.05rem;
  color: #e4eaf2;
}
.cv-nav-title span { color: #00d2c8; }
.cv-model-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem;
  padding: 4px 10px;
  background: rgba(0,133,255,.1);
  border: 1px solid rgba(0,133,255,.2);
  border-radius: 6px;
  color: #0085ff;
  letter-spacing: .03em;
}
.cv-status {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 13px;
  background: #0d1117;
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 99px;
  font-family: 'JetBrains Mono', monospace;
  font-size: .7rem;
}
.dot-idle    { width:8px;height:8px;border-radius:50%;background:#5a6a7d; display:inline-block; }
.dot-running { width:8px;height:8px;border-radius:50%;background:#00d2c8; display:inline-block; animation: pulse 1.2s infinite; }
.dot-done    { width:8px;height:8px;border-radius:50%;background:#00e09a; display:inline-block; }
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(0,210,200,.6); }
  70%  { box-shadow: 0 0 0 7px rgba(0,210,200,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,210,200,0); }
}

/* ── Video frame container ── */
.cv-video-wrap {
  background: #080c10;
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 14px;
  overflow: hidden;
  position: relative;
}
.cv-video-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 11px 16px;
  background: #0d1117;
  border-bottom: 1px solid rgba(255,255,255,.07);
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: .85rem;
}
.fps-tag {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem;
  padding: 3px 9px;
  background: rgba(0,210,200,.1);
  border: 1px solid rgba(0,210,200,.2);
  color: #00d2c8;
  border-radius: 6px;
}
.obj-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem;
  padding: 3px 9px;
  background: rgba(0,133,255,.1);
  border: 1px solid rgba(0,133,255,.2);
  color: #0085ff;
  border-radius: 6px;
}
.rec-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem;
  padding: 3px 9px;
  background: rgba(255,79,107,.1);
  border: 1px solid rgba(255,79,107,.2);
  color: #ff4f6b;
  border-radius: 6px;
}

/* ── Log panel ── */
.cv-log {
  background: #0d1117;
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 10px;
  padding: 12px 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: .68rem;
  color: #5a6a7d;
  max-height: 130px;
  overflow-y: auto;
}
.cv-log-entry { margin-bottom: 3px; }
.cv-log-entry .t { color: #00d2c8; margin-right: 10px; }
.cv-log-entry.ok .m  { color: #00e09a; }
.cv-log-entry.warn .m { color: #ffb800; }
.cv-log-entry.err .m  { color: #ff4f6b; }

/* ── Sidebar header ── */
.sb-section {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #5a6a7d;
  margin: 18px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODELS (cached) ─────────────────────────────────────
@st.cache_resource
def load_models():
    det  = YOLO("models/yolov8n.pt")
    seg  = YOLO("models/yolov8n-seg.pt")
    pose = YOLO("models/yolov8n-pose.pt")
    return det, seg, pose

det_model, seg_model, pose_model = load_models()
CLASS_NAMES = det_model.names
os.makedirs("output", exist_ok=True)

# ─── SESSION STATE ─────────────────────────────────────────────
if "running"   not in st.session_state: st.session_state.running   = False
if "logs"      not in st.session_state: st.session_state.logs      = [
    ("info",  "YOLO CV Suite initialized"),
    ("ok",    "Models loaded: det ✓  seg ✓  pose ✓"),
    ("info",  "Configure settings and press Run Analysis"),
]

def add_log(msg, kind="info"):
    ts = time.strftime("%H:%M:%S")
    st.session_state.logs.append((kind, f"[{ts}]  {msg}"))

# ─── TOP NAV ──────────────────────────────────────────────────
status_dot  = "dot-running" if st.session_state.running else "dot-idle"
status_text = "RUNNING"     if st.session_state.running else "IDLE"

st.markdown(f"""
<div class="cv-nav">
  <div class="cv-nav-logo">Y</div>
  <div class="cv-nav-title">YOLO <span>CV</span> Suite</div>
  <div class="cv-model-badge">YOLOv8n · ultralytics</div>
  <div class="cv-status">
    <span class="{status_dot}"></span>
    <span style="color:{'#00d2c8' if st.session_state.running else '#5a6a7d'}">{status_text}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-section">Task</div>', unsafe_allow_html=True)
    task = st.selectbox(
        "Task",
        ["Object Detection", "Object Tracking", "Object Counting", "Segmentation", "Pose Estimation"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sb-section">Input Source</div>', unsafe_allow_html=True)
    source = st.radio("Source", ["Webcam", "Video File"], horizontal=True, label_visibility="collapsed")

    uploaded_file = None
    if source == "Video File":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"], label_visibility="collapsed")

    st.markdown('<div class="sb-section">Confidence Threshold</div>', unsafe_allow_html=True)
    conf = st.slider("Confidence", 0.10, 1.00, 0.40, 0.05, label_visibility="collapsed")
    st.markdown(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:.75rem;color:#00d2c8;'
        f'background:rgba(0,210,200,.1);border:1px solid rgba(0,210,200,.2);'
        f'border-radius:6px;padding:3px 10px;display:inline-block;margin-bottom:4px">'
        f'conf ≥ {conf:.2f}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sb-section">Class Filter</div>', unsafe_allow_html=True)
    selected_classes = st.multiselect(
        "Classes",
        options=list(CLASS_NAMES.values()),
        default=[],
        placeholder="All classes (leave empty for ALL)",
        label_visibility="collapsed",
    )

    st.markdown('<div class="sb-section">Output</div>', unsafe_allow_html=True)
    save_output = st.checkbox("💾  Save output video", value=False)

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.running:
        if st.button("🚀  Run Analysis", type="primary"):
            if source == "Video File" and uploaded_file is None:
                st.error("Please upload a video file first.")
            else:
                st.session_state.running = True
                add_log(f"Starting {task} on {source.lower()}…")
                if selected_classes:
                    add_log(f"Filtering: {', '.join(selected_classes[:4])}{'…' if len(selected_classes)>4 else ''}", "info")
                if save_output:
                    add_log("Recording output video…", "warn")
                st.rerun()
    else:
        if st.button("⏹  Stop Analysis"):
            st.session_state.running = False
            add_log("Analysis stopped by user.", "ok")
            st.rerun()

# ─── MAIN AREA ────────────────────────────────────────────────
def get_class_ids(names):
    if not names:
        return None
    return [k for k, v in CLASS_NAMES.items() if v in names]

def pick_model():
    if task in ("Segmentation",):
        return seg_model, False, False
    elif task == "Pose Estimation":
        return pose_model, False, False
    elif task == "Object Tracking":
        return det_model, True, False
    elif task == "Object Counting":
        return det_model, False, True
    else:
        return det_model, False, False

# ── Stats row (top) ──
stat_fps_ph   = st.empty()
stat_fps_ph.markdown("""
<div style="display:flex;gap:12px;margin-bottom:16px">
  <div class="cv-card" style="flex:1"><div class="cv-card-label">Frames / sec</div><div class="cv-card-val" style="color:#00d2c8">—</div></div>
  <div class="cv-card" style="flex:1"><div class="cv-card-label">Objects Detected</div><div class="cv-card-val" style="color:#0085ff">—</div></div>
  <div class="cv-card" style="flex:1"><div class="cv-card-label">Avg Confidence</div><div class="cv-card-val" style="color:#00e09a">—</div></div>
  <div class="cv-card" style="flex:1"><div class="cv-card-label">Inference ms</div><div class="cv-card-val" style="color:#ffb800">—</div></div>
</div>
""", unsafe_allow_html=True)

# ── Video panel header ──
vp_header_ph = st.empty()
vp_header_ph.markdown(f"""
<div class="cv-video-wrap">
  <div class="cv-video-header">
    <span style="color:#5a6a7d;font-size:.75rem;font-family:'JetBrains Mono',monospace">◉</span>
    {task} — {source}
    <span class="fps-tag">0 FPS</span>
    <span class="obj-tag">0 OBJECTS</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Video frame ──
frame_ph = st.empty()

# ── Placeholder when idle ──
if not st.session_state.running:
    frame_ph.markdown("""
<div style="background:#080c10;border:1px solid rgba(255,255,255,.07);border-radius:14px;
            padding:80px 20px;text-align:center;margin-top:-4px;border-top:none;border-top-left-radius:0;border-top-right-radius:0">
  <div style="font-size:52px;opacity:.15;margin-bottom:16px">◉</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#5a6a7d;letter-spacing:.06em">
    AWAITING FEED — SELECT SOURCE &amp; PRESS RUN ANALYSIS
  </div>
</div>
""", unsafe_allow_html=True)

# ── Log panel ──
log_ph = st.empty()

def render_log():
    rows = "".join(
        f'<div class="cv-log-entry {k}"><span class="t">›</span><span class="m">{m}</span></div>'
        for k, m in st.session_state.logs[-20:]
    )
    log_ph.markdown(f'<div style="margin-top:14px"><div style="font-family:JetBrains Mono,monospace;font-size:.62rem;color:#5a6a7d;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px">⬛ SYSTEM LOG</div><div class="cv-log">{rows}</div></div>', unsafe_allow_html=True)

render_log()

# ─── INFERENCE LOOP ───────────────────────────────────────────
if st.session_state.running:
    model, do_track, do_count = pick_model()
    class_ids = get_class_ids(selected_classes)

    # Open video source
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Output writer
    out_writer = None
    out_path   = None
    if save_output:
        out_path = f"output/{task.replace(' ', '_')}.mp4"
        out_writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src,
            (width, height),
        )

    frame_count = 0
    fps_display = 0
    loop_start  = time.time()
    t_fps_reset = time.time()

    add_log(f"Feed open — {width}×{height} @ {fps_src}fps", "ok")
    render_log()

    while st.session_state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            add_log("Stream ended.", "ok")
            st.session_state.running = False
            break

        t0 = time.perf_counter()

        # ── Run inference ──
        if do_track:
            results = model.track(frame, persist=True, conf=conf, classes=class_ids, verbose=False)
        else:
            results = model(frame, conf=conf, classes=class_ids, verbose=False)

        infer_ms = int((time.perf_counter() - t0) * 1000)

        annotated = results[0].plot()

        # Count overlay
        n_obj = len(results[0].boxes) if results[0].boxes is not None else 0
        if do_count:
            cv2.putText(annotated, f"COUNT: {n_obj}", (16, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 210, 200), 2)

        if out_writer:
            out_writer.write(annotated)

        # FPS calc
        frame_count += 1
        elapsed = time.time() - t_fps_reset
        if elapsed >= 1.0:
            fps_display  = round(frame_count / elapsed)
            frame_count  = 0
            t_fps_reset  = time.time()

        # Avg confidence
        if results[0].boxes is not None and len(results[0].boxes):
            confs = results[0].boxes.conf.cpu().numpy()
            avg_conf = f"{np.mean(confs):.2f}"
        else:
            avg_conf = "—"

        # ── Update stats bar ──
        stat_fps_ph.markdown(f"""
<div style="display:flex;gap:12px;margin-bottom:16px">
  <div class="cv-card" style="flex:1"><div class="cv-card-label">Frames / sec</div><div class="cv-card-val" style="color:#00d2c8">{fps_display}</div></div>
  <div class="cv-card" style="flex:1"><div class="cv-card-label">Objects Detected</div><div class="cv-card-val" style="color:#0085ff">{n_obj}</div></div>
  <div class="cv-card" style="flex:1"><div class="cv-card-label">Avg Confidence</div><div class="cv-card-val" style="color:#00e09a">{avg_conf}</div></div>
  <div class="cv-card" style="flex:1"><div class="cv-card-label">Inference ms</div><div class="cv-card-val" style="color:#ffb800">{infer_ms}ms</div></div>
</div>
""", unsafe_allow_html=True)

        # ── Update video header ──
        rec_tag = '<span class="rec-tag">● REC</span>' if save_output else ""
        vp_header_ph.markdown(f"""
<div class="cv-video-wrap">
  <div class="cv-video-header">
    <span style="color:#00d2c8;font-size:.75rem;font-family:'JetBrains Mono',monospace;animation:pulse 1.2s infinite">◉</span>
    {task} — {source}
    <span class="fps-tag">{fps_display} FPS</span>
    <span class="obj-tag">{n_obj} OBJECTS</span>
    {rec_tag}
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Render frame ──
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_ph.image(
            annotated_rgb,
            channels="RGB",
            use_container_width=True,
        )

    cap.release()
    if out_writer:
        out_writer.release()
        add_log(f"Saved → {out_path}", "ok")

    st.session_state.running = False
    render_log()

    if save_output and out_path and os.path.exists(out_path):
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️  Download Output Video",
                data=f,
                file_name=os.path.basename(out_path),
                mime="video/mp4",
            )
