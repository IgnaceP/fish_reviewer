import streamlit as st
import decord
from decord import VideoReader, cpu
import pandas as pd
import threading
import numpy as np
import time
import os
from datetime import datetime, timedelta

# --- THREAD SAFETY ---
if 'lock' not in st.session_state:
    st.session_state.lock = threading.Lock()

# --- CONFIG & PATHS ---
VIDEO_PATH = "data/scratch/fish_2025-11-04_09-14-34.mp4"
try:
    VIDEO_DATETIME = datetime.strptime(VIDEO_PATH.split('/')[-1][5:24], '%Y-%m-%d_%H-%M-%S')
except:
    VIDEO_DATETIME = datetime.now()

YMIN, YMAX, XMIN, XMAX = 160, 920, 600, 1400

# --- HELPERS ---
def cluster_indices(indices, gap=180):
    if not indices: return []
    clusters, sorted_inds = [], sorted(indices)
    current_cluster = [sorted_inds[0]]
    for i in range(1, len(sorted_inds)):
        if sorted_inds[i] <= sorted_inds[i-1] + gap:
            current_cluster.append(sorted_inds[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_inds[i]]
    clusters.append(current_cluster)
    return clusters

@st.cache_resource
def get_reader(path):
    return VideoReader(path, ctx=cpu(0))

# --- INITIALIZATION ---
if 'events' not in st.session_state:
    try:
        raw = list(np.load(VIDEO_PATH.replace('.mp4', '_suspicious_frames.npy')))
    except:
        raw = list(range(1000, 1051))
    st.session_state.events = cluster_indices(raw, gap=180)

# CRITICAL: Initialize the Master DataFrame if it doesn't exist
if 'df_counts' not in st.session_state:
    vr_init = get_reader(VIDEO_PATH)
    fps_init = vr_init.get_avg_fps()
    data = []
    for i, event in enumerate(st.session_state.events):
        data.append({
            "event_id": i,
            "start_frame": event[0],
            "time": (VIDEO_DATETIME + timedelta(seconds=event[0] / fps_init)).strftime("%H:%M:%S"),
            "L_in": 0, "R_in": 0, "T_out_R": 0, "T_out_L": 0, "L2R": 0, "R2L": 0
        })
    st.session_state.df_counts = pd.DataFrame(data)

if 'event_idx' not in st.session_state:
    st.session_state.event_idx = 0
if 'sub_frame_idx' not in st.session_state:
    st.session_state.sub_frame_idx = 0

# --- LOGIC FUNCTIONS ---
def sync_to_df():
    """Saves current screen widget values into the Master DataFrame"""
    idx = st.session_state.event_idx
    for cat in ["L_in", "R_in", "T_out_R", "T_out_L", "L2R", "R2L"]:
        widget_key = f"{cat}_widget_{idx}"
        if widget_key in st.session_state:
            st.session_state.df_counts.at[idx, cat] = st.session_state[widget_key]

def move_event(delta):
    sync_to_df() # Save current event before leaving
    new_idx = st.session_state.event_idx + delta
    if 0 <= new_idx < len(st.session_state.events):
        st.session_state.event_idx = new_idx
        st.session_state.sub_frame_idx = 0

def move_frame(delta):
    current_event = st.session_state.events[st.session_state.event_idx]
    new_val = st.session_state.sub_frame_idx + delta
    st.session_state.sub_frame_idx = max(0, min(new_val, len(current_event) - 1))

# --- HOTKEYS ---
st.components.v1.html("""
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (doc.activeElement.tagName === 'INPUT') return;
        if(["ArrowLeft","ArrowRight", "Tab", "w", "x"].includes(e.key.toLowerCase())) e.preventDefault();
        const click = (txt) => {
            const b = Array.from(doc.querySelectorAll('button')).find(btn => btn.innerText.includes(txt));
            if (b) b.click();
        };
        switch(e.key.toLowerCase()) {
            case 'tab': click('Next Event'); break;
            case 'w': click('(-1)'); break;
            case 'x': click('(+1)'); break;
            case 'arrowleft': click('(-5)'); break;
            case 'arrowright': click('(+5)'); break;
        }
    }, {passive: false});
    </script>
    """, height=0)

# --- UI DISPLAY ---
st.title("🐟 Fish Review")
events = st.session_state.events
vr = get_reader(VIDEO_PATH)
fps = vr.get_avg_fps()

idx = st.session_state.event_idx
current_cluster = events[idx]
real_frame_num = current_cluster[st.session_state.sub_frame_idx]

st.subheader(f"Event {idx + 1} / {len(events)} — Frame {st.session_state.sub_frame_idx + 1} of {len(current_cluster)}")

# Sidebar
sc1, sc2 = st.sidebar.columns(2)
sc1.button("⬅️ Prev Event", on_click=move_event, args=(-1,), disabled=(idx == 0))
sc2.button("Next Event ➡️", on_click=move_event, args=(1,), disabled=(idx == len(events)-1))

# Main Image
with st.session_state.lock:
    time.sleep(0.01)
    frame = vr[real_frame_num].asnumpy()[YMIN:YMAX, XMIN:XMAX]
st.image(frame, width=800)

# Playback Controls
c1, c2, c3, c4, c5 = st.columns(5)
c1.button("⏪ (-5)", on_click=move_frame, args=(-5,))
c2.button("(-1) W", on_click=move_frame, args=(-1,))
c3.markdown(f"<center><b>Frame {real_frame_num}</b></center>", unsafe_allow_html=True)
c4.button("X (+1)", on_click=move_frame, args=(1,))
c5.button("(+5) ⏩", on_click=move_frame, args=(5,))

# --- COUNTERS (THE FIX) ---
st.divider()
cats = ["L_in", "R_in", "T_out_R", "T_out_L", "L2R", "R2L"]
labels = ["from left into trap", "from right into trap", "from trap to right", "from trap to left", "left to right", "right to left"]

cols = st.columns(3)
cols2 = st.columns(3)
all_cols = cols + cols2

for i, cat in enumerate(cats):
    # We pull the value DIRECTLY from our master dataframe
    current_val = int(st.session_state.df_counts.at[idx, cat])
    all_cols[i].number_input(
        labels[i],
        min_value=0,
        value=current_val,
        key=f"{cat}_widget_{idx}",
        on_change=sync_to_df # Auto-sync if user types manually
    )

# --- TABLE VIEW & EXPORT ---
st.divider()
with st.expander("📊 View Live Tally Table"):
    sync_to_df() # One last sync before showing
    st.dataframe(st.session_state.df_counts, use_container_width=True)

if st.sidebar.button("💾 Save to CSV"):
    sync_to_df()
    out = VIDEO_PATH.replace('.mp4', '_fish_counts.csv')
    st.session_state.df_counts.to_csv(out, index=False)
    st.sidebar.success(f"Saved to {os.path.basename(out)}")