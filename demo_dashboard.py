"""
================================================================================
M10 EDPS Interactive Panel Demo Dashboard
================================================================================
An interactive Streamlit web dashboard designed specifically for project
reviews and panel presentations.

Features:
  1. Live Recording Selector (Prophesee Gen1 DAT + BBOX pairs).
  2. Interactive Time Window Slider (0s to 60s).
  3. Live EDPS Gate Threshold Slider (0.10 to 0.90) to interactively demonstrate
     token pruning in real time before the panel!
  4. Side-by-Side Comparison:
     - Left: Standard Dense Transformer (100% tokens, heavy compute)
     - Right: Proposed EDPS Sparse Model (Dynamic token selection, energy efficient)
  5. Live Performance Metrics (Tokens Retained %, FLOPs Saved, Energy Efficiency Boost).

Usage (Local Windows):
    pip install streamlit matplotlib
    streamlit run demo_dashboard.py
================================================================================
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches_plt
import streamlit as st

# Ensure project modules are importable across Colab subdirectories
current_dir = os.path.dirname(os.path.abspath(__file__))
for path in [current_dir, os.getcwd(), "/content"]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

# Dynamically find directory containing event_parser.py if needed
for search_root in ["/content", "/content/drive", current_dir, os.getcwd()]:
    if os.path.exists(search_root):
        for root, _, files in os.walk(search_root):
            if "event_parser.py" in files:
                if root not in sys.path:
                    sys.path.insert(0, root)
                break
from event_parser import EventParser
from voxel_grid import VoxelGridConfig, events_to_voxel_grid
from patch_config import PatchConfig
from patch_partition import compute_patch_grid_dims, partition_voxel_grid
from patch_statistics import compute_patch_activity_stats
from patch_adjacency import build_adjacency_map
from patch_embedding_config import PatchEmbeddingConfig
from patch_embedding_module import PatchEmbeddingModule
from edps_config import EDPSConfig
from edps_module import EDPSModule

# ------------------------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="EDPS Event Vision Demo",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy-Efficient Dynamic Sparse Self-Attention for Event Vision")
st.subheader("Interactive Research Panel Presentation & Real-Time Token Pruning Demo")

# ------------------------------------------------------------------------------
# Data Input (Prominent Main Page Upload + Dataset Fallback)
# ------------------------------------------------------------------------------
st.sidebar.header("🎛️ Demo Controls")

dat_path = None
bbox_path = None

st.subheader("📁 1. Upload Event Recording File (.dat)")
uploaded_file = st.file_uploader("Drag & Drop your Prophesee .dat Recording File Here", type=["dat"])

if uploaded_file is not None:
    temp_dir = os.path.join(current_dir, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    dat_path = os.path.join(temp_dir, uploaded_file.name)
    with open(dat_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    uploaded_bbox = st.file_uploader("🏷️ Upload Optional _bbox.npy Label File", type=["npy"])
    if uploaded_bbox is not None:
        bbox_path = os.path.join(temp_dir, uploaded_bbox.name)
        with open(bbox_path, "wb") as f:
            f.write(uploaded_bbox.getbuffer())
else:
    # Check for local dataset as fallback option
    dataset_candidates = ["/content/gen1_local", "/content", os.path.join(current_dir, "gen1_local")]
    dataset_root = None
    for candidate in dataset_candidates:
        if os.path.exists(candidate):
            dataset_root = candidate
            break

    dat_files = []
    if dataset_root:
        for root, _, files in os.walk(dataset_root):
            for f in files:
                if f.endswith(".dat"):
                    dat_files.append(os.path.relpath(os.path.join(root, f), dataset_root))

    if dat_files:
        st.info("💡 Alternatively, select a recording from the local dataset folder below:")
        selected_rel_dat = st.selectbox("Select Recording from Dataset", sorted(dat_files))
        dat_path = os.path.join(dataset_root, selected_rel_dat)
        possible_bbox = os.path.join(dataset_root, selected_rel_dat.replace("_td.dat", "_bbox.npy").replace(".dat", ".npy"))
        if os.path.exists(possible_bbox):
            bbox_path = possible_bbox
    else:
        st.warning("👈 Please drag and drop a `.dat` event file above to start the live presentation demo.")
        st.stop()

# Processing Parameters
window_ms = st.sidebar.slider("Window Duration (ms)", min_value=10, max_value=100, value=50, step=10)
time_start_s = st.sidebar.slider("Start Time (seconds)", min_value=0.0, max_value=55.0, value=2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🎯 EDPS Pruning Controls")
gate_thresh = st.sidebar.slider("EDPS Gate Threshold (thresh)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
patch_size = st.sidebar.selectbox("Patch Size (pixels)", options=[16, 32], index=0)

# ------------------------------------------------------------------------------
# Data Loading & Model Components
# ------------------------------------------------------------------------------
parser = EventParser()
try:
    header = parser.reader.parse_header(dat_path)
    t_min, t_max, _ = parser.reader.get_time_range(dat_path, header)
except Exception as e:
    st.error(f"Error reading file header: {e}")
    st.stop()

t_start_us = int(time_start_s * 1e6)
t_end_us = t_start_us + int(window_ms * 1000)

events = parser.load_events_window(dat_path, t_start=t_start_us, t_end=t_end_us, validate=False)
boxes = parser.load_annotations(bbox_path) if (bbox_path and os.path.exists(bbox_path)) else []
# Filter boxes for this window
window_boxes = [b for b in boxes if (b["t"] >= t_start_us and b["t"] < t_end_us)]

# Voxelization & Patching
voxel_cfg = VoxelGridConfig(num_bins=10, temporal_mode="bilinear", polarity_encoding="signed")
grid_np = events_to_voxel_grid(
    events["t"], events["x"], events["y"], events["p"],
    t_start_us, t_end_us, events["header"]["height"], events["header"]["width"], voxel_cfg
)
grid_tensor = torch.from_numpy(grid_np)

patch_cfg = PatchConfig(patch_size=patch_size, padding_mode="zero")
patches, meta = partition_voxel_grid(grid_tensor, patch_cfg)
stats = compute_patch_activity_stats(patches, voxel_cfg)
n_rows, n_cols, eff_h, eff_w = compute_patch_grid_dims(events["header"]["height"], events["header"]["width"], patch_cfg)
adj = build_adjacency_map(n_rows, n_cols, patch_cfg)

# Instantiate EDPS Module
embed_cfg = PatchEmbeddingConfig(embedding_dim=128)
embed_mod = PatchEmbeddingModule(embed_cfg, voxel_cfg.num_channels, patch_size, n_rows, n_cols)
edps_cfg = EDPSConfig(gate_threshold=gate_thresh, selection_mode="normal")
edps_mod = EDPSModule(edps_cfg, embed_cfg.embedding_dim, voxel_cfg.num_bins)

with torch.no_grad():
    embeddings, embed_meta = embed_mod(patches, meta)
    edps_out = edps_mod(embeddings, stats, meta, embed_meta, adj)

scores = edps_out.importance_scores.numpy()
binary_mask = edps_out.binary_mask.numpy()
n_total = len(scores)
n_kept = int(binary_mask.sum())
keep_ratio = (n_kept / n_total) * 100.0
pruned_ratio = 100.0 - keep_ratio
flops_reduction = 100.0 / max(1.0, keep_ratio)

# ------------------------------------------------------------------------------
# Metric Cards
# ------------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Input Patches", f"{n_total} patches")
col2.metric("Retained Tokens (Object Focus)", f"{n_kept} ({keep_ratio:.1f}%)", delta=f"-{pruned_ratio:.1f}% Pruned", delta_color="normal")
col3.metric("Energy / FLOPs Reduction", f"{flops_reduction:.2f}x Speedup", delta="Compute Saved")
col4.metric("Active Window Events", f"{len(events['t']):,} events")

st.markdown("---")

# ------------------------------------------------------------------------------
# Side-by-Side Visualization Plots
# ------------------------------------------------------------------------------
st.subheader("🖼️ Real-Time Token Selection & Pruning Overlay")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 2D Event Accumulation Image
event_img = np.sum(np.abs(grid_np), axis=0)

# Left: Baseline (Dense 100% Tokens)
ax1.imshow(event_img, cmap="gray_r", origin="upper")
ax1.set_title(f"Baseline Standard Transformer\n(100% Tokens = {n_total}/{n_total} Processed | Heavy Compute)", fontsize=12, fontweight="bold", color="crimson")
for m in meta:
    rect = patches_plt.Rectangle((m.x0, m.y0), patch_size, patch_size, linewidth=0.5, edgecolor="blue", facecolor="none", alpha=0.3)
    ax1.add_patch(rect)
for b in window_boxes:
    rect = patches_plt.Rectangle((b["x"], b["y"]), b["w"], b["h"], linewidth=2, edgecolor="gold", facecolor="none")
    ax1.add_patch(rect)
ax1.set_axis_off()

# Right: Proposed EDPS Model (Dynamic Sparse Tokens)
ax2.imshow(event_img, cmap="gray_r", origin="upper")
ax2.set_title(f"Proposed Dynamic Sparse Model (EDPS)\n(Only {n_kept}/{n_total} Tokens Kept [{keep_ratio:.1f}%] | Threshold={gate_thresh:.2f})", fontsize=12, fontweight="bold", color="darkgreen")

for i, m in enumerate(meta):
    is_kept = binary_mask[i]
    score = scores[i]
    if is_kept:
        # Green box for retained object token
        rect = patches_plt.Rectangle((m.x0, m.y0), patch_size, patch_size, linewidth=1.5, edgecolor="lime", facecolor="green", alpha=0.25)
    else:
        # Red transparent box for pruned background token
        rect = patches_plt.Rectangle((m.x0, m.y0), patch_size, patch_size, linewidth=0.5, edgecolor="red", facecolor="red", alpha=0.08)
    ax2.add_patch(rect)

for b in window_boxes:
    rect = patches_plt.Rectangle((b["x"], b["y"]), b["w"], b["h"], linewidth=2, edgecolor="gold", facecolor="none")
    ax2.add_patch(rect)
ax2.set_axis_off()

plt.tight_layout()
st.pyplot(fig)

# ------------------------------------------------------------------------------
# Explanation & Research Takeaways for Panel
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("💡 Key Research Highlights for the Panel")
st.markdown(f"""
- **Dynamic Sparse Token Pruning:** Notice how EDPS automatically places **green highlight boxes** around object areas (cars, pedestrians, movement) while discarding background static noise (**red boxes**).
- **Interactive Thresholding:** Moving the **EDPS Gate Threshold** slider dynamically increases or decreases token retention live.
- **Energy & Compute Savings:** At threshold `{gate_thresh:.2f}`, the vision backbone processes **only {keep_ratio:.1f}% of tokens**, saving **{pruned_ratio:.1f}% of self-attention FLOPs** for real-time edge execution.
""")
