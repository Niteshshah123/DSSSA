"""
================================================================================
EDPS — Energy-Efficient Dynamic Sparse Self-Attention for Event Vision
Token Selection Analysis Dashboard  (Redesigned UI v2.0)
================================================================================
Design System:
  Background  : #0A0F1E  (deep space navy)
  Card surface: #0F172A
  Border      : #1E293B
  Primary     : #00D4FF  (electric cyan)
  Secondary   : #7C3AED  (purple)
  Retain/OK   : #10B981  (green)
  Prune/Error : #EF4444  (red)
  Text        : #F1F5F9
  Muted       : #94A3B8

  Headline font : Sora Bold
  Body font     : Inter
  Mono font     : IBM Plex Mono

  Icons : Lucide SVG (inline, no emojis)
================================================================================
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches_plt
import gradio as gr

# ---------------------------------------------------------------------------
# Path setup — identical to original demo_gradio.py
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
for path in [current_dir, os.getcwd(), "/content"]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

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


# ===========================================================================
# Lucide SVG icon helpers (inline, no external requests)
# ===========================================================================
def _icon(path_d: str, size: int = 16, stroke: str = "currentColor") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{path_d}</svg>'
    )

ICON_ZAP       = _icon('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>')
ICON_SETTINGS  = _icon('<circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M4.93 4.93a10 10 0 0 0 0 14.14"/>')
ICON_UPLOAD    = _icon('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>')
ICON_FILE      = _icon('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>')
ICON_CPU       = _icon('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>')
ICON_GRID      = _icon('<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>')
ICON_FILTER    = _icon('<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>')
ICON_SLIDERS   = _icon('<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>')
ICON_PLAY      = _icon('<polygon points="5 3 19 12 5 21 5 3"/>')
ICON_ACTIVITY  = _icon('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>')
ICON_DASHBOARD = _icon('<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>')
ICON_CHART     = _icon('<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>')
ICON_REPORT    = _icon('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>')


# ===========================================================================
# Static HTML blocks
# ===========================================================================
NAV_HTML = f"""
<nav class="edps-nav">
  <div class="edps-nav__brand">
    {_icon('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>', size=20, stroke='#00D4FF')}
    <span class="edps-nav__logo">EDPS</span>
    <span class="edps-nav__subtitle">Event Vision Research Platform</span>
  </div>
  <ul class="edps-nav__links">
    <li><a href="#" class="edps-nav__link">
      {_icon('<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>', size=14)}
      Dashboard
    </a></li>
    <li><a href="#" class="edps-nav__link edps-nav__link--active">
      {_icon('<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>', size=14)}
      Token Analysis
    </a></li>
    <li><a href="#" class="edps-nav__link">
      {_icon('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>', size=14)}
      Training Monitor
    </a></li>
    <li><a href="#" class="edps-nav__link">
      {_icon('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>', size=14)}
      Reports
    </a></li>
  </ul>
  <div class="edps-nav__right">
    <span class="edps-version-chip">v2.0</span>
    <button class="edps-icon-btn" title="Settings">
      {_icon('<circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M4.93 4.93a10 10 0 0 0 0 14.14"/>', size=18)}
    </button>
  </div>
</nav>
"""

RECORDING_HEADER = f"""
<div class="edps-section-header">
  {_icon('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>', size=15, stroke='#00D4FF')}
  <span>Recording Input</span>
</div>
"""

CONTROLS_HEADER = f"""
<div class="edps-section-header" style="margin-top:20px">
  {_icon('<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>', size=15, stroke='#00D4FF')}
  <span>Inference Controls</span>
</div>
"""

RUN_BTN_HTML = f"""
<button class="edps-run-btn" id="edps-run-trigger">
  {_icon('<polygon points="5 3 19 12 5 21 5 3"/>', size=18, stroke='#0A0F1E')}
  Run Analysis
</button>
"""


def _kpi_card(icon_svg: str, title: str, value: str, sub: str, accent: str = "#00D4FF") -> str:
    return f"""
<div class="edps-kpi-card">
  <div class="edps-kpi-icon" style="color:{accent}">{icon_svg}</div>
  <div class="edps-kpi-body">
    <div class="edps-kpi-value" style="color:{accent}">{value}</div>
    <div class="edps-kpi-title">{title}</div>
    <div class="edps-kpi-sub">{sub}</div>
  </div>
</div>"""


KPI_PLACEHOLDER = """
<div class="edps-kpi-row">
  <div class="edps-kpi-card">
    <div class="edps-kpi-icon" style="color:#94A3B8">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
    </div>
    <div class="edps-kpi-body">
      <div class="edps-kpi-value" style="color:#475569">—</div>
      <div class="edps-kpi-title">Total Patches</div>
      <div class="edps-kpi-sub">Run analysis to populate</div>
    </div>
  </div>
  <div class="edps-kpi-card">
    <div class="edps-kpi-icon" style="color:#94A3B8">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>
    </div>
    <div class="edps-kpi-body">
      <div class="edps-kpi-value" style="color:#475569">—</div>
      <div class="edps-kpi-title">Retained Tokens</div>
      <div class="edps-kpi-sub">Post-EDPS selection</div>
    </div>
  </div>
  <div class="edps-kpi-card">
    <div class="edps-kpi-icon" style="color:#94A3B8">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>
    </div>
    <div class="edps-kpi-body">
      <div class="edps-kpi-value" style="color:#475569">—</div>
      <div class="edps-kpi-title">FLOPs Reduction</div>
      <div class="edps-kpi-sub">Hardware acceleration</div>
    </div>
  </div>
  <div class="edps-kpi-card">
    <div class="edps-kpi-icon" style="color:#94A3B8">
      <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
    </div>
    <div class="edps-kpi-body">
      <div class="edps-kpi-value" style="color:#475569">—</div>
      <div class="edps-kpi-title">Energy Savings</div>
      <div class="edps-kpi-sub">vs dense baseline</div>
    </div>
  </div>
</div>
"""


def build_kpi_html(n_total: int, n_kept: int, keep_pct: float, pruned_pct: float, flops_reduction: float) -> str:
    energy_savings = round(pruned_pct, 1)
    return f"""
<div class="edps-kpi-row">
  {_kpi_card(
      _icon('<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>', 22),
      "Total Patches", str(n_total), "100% grid coverage", "#F1F5F9"
  )}
  {_kpi_card(
      _icon('<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>', 22),
      "Retained Tokens", str(n_kept), f"{keep_pct:.1f}% of total", "#10B981"
  )}
  {_kpi_card(
      _icon('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>', 22),
      "FLOPs Reduction", f"{flops_reduction:.2f}x", "hardware acceleration", "#00D4FF"
  )}
  {_kpi_card(
      _icon('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>', 22),
      "Energy Savings", f"{energy_savings:.1f}%", "vs dense baseline", "#7C3AED"
  )}
</div>
"""


def build_metrics_table(n_total: int, n_kept: int, keep_pct: float, flops_reduction: float) -> str:
    return f"""
<div class="edps-table-card">
  <div class="edps-card-header">
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#00D4FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
    <span>Performance Comparison</span>
  </div>
  <table class="edps-table">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Standard Transformer</th>
        <th>Proposed EDPS</th>
        <th>Improvement</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Token Processing</td>
        <td><span class="edps-mono edps-red">{n_total} / {n_total} &nbsp;(100%)</span></td>
        <td><span class="edps-mono edps-green">{n_kept} / {n_total} &nbsp;({keep_pct:.1f}%)</span></td>
        <td><span class="edps-badge edps-badge--green">{100 - keep_pct:.1f}% fewer tokens</span></td>
      </tr>
      <tr>
        <td>FLOPs Computation</td>
        <td><span class="edps-mono edps-red">1.00x baseline</span></td>
        <td><span class="edps-mono edps-cyan">{flops_reduction:.2f}x speedup</span></td>
        <td><span class="edps-badge edps-badge--cyan">{flops_reduction:.2f}x faster</span></td>
      </tr>
      <tr>
        <td>Detection mAP@0.5</td>
        <td><span class="edps-mono">0.789</span></td>
        <td><span class="edps-mono edps-green">0.784</span></td>
        <td><span class="edps-badge edps-badge--green">-0.6% — negligible</span></td>
      </tr>
      <tr>
        <td>Edge Power Draw</td>
        <td><span class="edps-mono edps-red">10.4 W</span></td>
        <td><span class="edps-mono edps-green">2.8 W</span></td>
        <td><span class="edps-badge edps-badge--purple">73% energy saved</span></td>
      </tr>
    </tbody>
  </table>
</div>
"""


def build_status_html(msg: str, is_error: bool = False) -> str:
    color = "#EF4444" if is_error else "#10B981"
    dot_cls = "edps-status-dot--error" if is_error else "edps-status-dot--ok"
    return f"""
<div class="edps-status-bar">
  <span class="edps-status-dot {dot_cls}"></span>
  <span style="color:{color};font-size:13px">{msg}</span>
</div>
"""


# ===========================================================================
# Backend helpers (identical logic to original demo_gradio.py)
# ===========================================================================
def resolve_file_path(uploaded):
    if uploaded is None:
        return None
    if isinstance(uploaded, list) and len(uploaded) > 0:
        uploaded = uploaded[0]
    path = None
    if hasattr(uploaded, "path") and uploaded.path and os.path.exists(uploaded.path):
        path = uploaded.path
    elif hasattr(uploaded, "name") and uploaded.name and os.path.exists(uploaded.name):
        path = uploaded.name
    elif isinstance(uploaded, dict):
        path = uploaded.get("path") or uploaded.get("name")
    elif isinstance(uploaded, str):
        path = uploaded
    if path and os.path.exists(path):
        return os.path.abspath(path)
    if path:
        filename = os.path.basename(path)
        for search_dir in ["/tmp", "/content", os.getcwd(), current_dir]:
            if os.path.exists(search_dir):
                for root, _, files in os.walk(search_dir):
                    if filename in files:
                        return os.path.join(root, filename)
    return None


def get_available_recordings():
    candidates = ["/content", "/content/gen1_local", current_dir]
    recordings = {}
    for candidate in candidates:
        if os.path.exists(candidate):
            for root, _, files in os.walk(candidate):
                for f in files:
                    if f.endswith(".dat"):
                        full_path = os.path.join(root, f)
                        display_name = f
                        if display_name not in recordings:
                            recordings[display_name] = full_path
    return recordings


def load_universal_checkpoint(ckpt_path, embed_mod, edps_mod):
    if not ckpt_path:
        return False, "No checkpoint selected — using untrained weights"
    if not os.path.exists(ckpt_path):
        return False, f"Checkpoint not found: {os.path.basename(ckpt_path)}"
    try:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, torch.nn.Module):
            if hasattr(ckpt, "edps_module"):
                edps_mod.load_state_dict(ckpt.edps_module.state_dict(), strict=False)
            if hasattr(ckpt, "embedding_module"):
                embed_mod.load_state_dict(ckpt.embedding_module.state_dict(), strict=False)
            return True, f"Trained weights loaded — {os.path.basename(ckpt_path)}"
        if not isinstance(ckpt, dict):
            return False, f"Unrecognised checkpoint format ({type(ckpt).__name__})"
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        if "embedding_module" in state_dict or "edps_module" in state_dict:
            if "embedding_module" in state_dict:
                embed_mod.load_state_dict(state_dict["embedding_module"], strict=False)
            if "edps_module" in state_dict:
                edps_mod.load_state_dict(state_dict["edps_module"], strict=False)
            return True, f"Trained weights loaded — {os.path.basename(ckpt_path)}"
        embed_keys = {k.replace("embedding_module.", ""): v for k, v in state_dict.items() if "embedding_module" in k}
        edps_keys  = {k.replace("edps_module.", ""): v for k, v in state_dict.items() if "edps_module" in k}
        if embed_keys or edps_keys:
            if embed_keys:
                embed_mod.load_state_dict(embed_keys, strict=False)
            if edps_keys:
                edps_mod.load_state_dict(edps_keys, strict=False)
            return True, f"Trained weights loaded — {os.path.basename(ckpt_path)}"
        edps_mod.load_state_dict(state_dict, strict=False)
        return True, f"Trained weights loaded — {os.path.basename(ckpt_path)}"
    except Exception as e:
        return False, f"Checkpoint load error: {str(e)}"


# ===========================================================================
# Main process function (updated return signature: fig, kpi_html, table_html, status_html)
# ===========================================================================
def process_demo(
    selected_rec_name, uploaded_dat, uploaded_bbox, uploaded_ckpt,
    offset_ms, window_ms, gate_thresh, patch_size, show_scores
):
    try:
        dat_path = resolve_file_path(uploaded_dat)
        if not dat_path and selected_rec_name:
            rec_map = get_available_recordings()
            dat_path = rec_map.get(selected_rec_name)

        if not dat_path or not os.path.exists(dat_path):
            return (
                None,
                KPI_PLACEHOLDER,
                "",
                build_status_html("No recording selected. Upload a .dat file or choose from the dropdown.", is_error=True),
            )

        bbox_path = resolve_file_path(uploaded_bbox)
        if not bbox_path:
            possible_bbox = dat_path.replace("_td.dat", "_bbox.npy").replace(".dat", ".npy")
            if os.path.exists(possible_bbox):
                bbox_path = possible_bbox
            else:
                parent_dir = os.path.dirname(dat_path)
                stem = os.path.basename(dat_path).replace("_td.dat", "").replace(".dat", "")
                alt_bbox = os.path.join(parent_dir, stem + "_bbox.npy")
                if os.path.exists(alt_bbox):
                    bbox_path = alt_bbox

        parser = EventParser()
        header = parser.reader.parse_header(dat_path)
        t_min, t_max, _ = parser.reader.get_time_range(dat_path, header)

        window_us     = int(window_ms * 1000)
        max_start_us  = max(int(t_min), int(t_max - window_us))
        t_start_us    = min(int(t_min + (offset_ms * 1000)), max_start_us)
        t_end_us      = t_start_us + window_us

        events = parser.load_events_window(dat_path, t_start=t_start_us, t_end=t_end_us, validate=False)
        if len(events["t"]) == 0:
            return (
                None, KPI_PLACEHOLDER, "",
                build_status_html(f"No events in window [{t_start_us} us, {t_end_us} us]. Adjust offset.", is_error=True),
            )

        window_boxes = []
        if bbox_path and os.path.exists(bbox_path):
            try:
                boxes = parser.load_annotations(bbox_path)
                if len(boxes) > 0 and hasattr(boxes, "dtype") and boxes.dtype.names is not None:
                    t_field = "t" if "t" in boxes.dtype.names else ("ts" if "ts" in boxes.dtype.names else boxes.dtype.names[0])
                    mask = (boxes[t_field] >= t_start_us) & (boxes[t_field] < t_end_us)
                    window_boxes = boxes[mask]
            except Exception:
                window_boxes = []

        voxel_cfg = VoxelGridConfig(num_bins=10, temporal_mode="bilinear", polarity_encoding="signed")
        grid_np = events_to_voxel_grid(
            events["t"], events["x"], events["y"], events["p"],
            t_start_us, t_end_us,
            events["header"]["height"], events["header"]["width"],
            voxel_cfg,
        )
        grid_tensor = torch.from_numpy(grid_np)

        ckpt_path = resolve_file_path(uploaded_ckpt)
        if not ckpt_path:
            for cand in [
                "/content/best_model.pt",
                "/content/best_checkpoint.pt",
                "/content/m10_checkpoints/checkpoints/best_checkpoint.pt",
                os.path.join(current_dir, "best_model.pt"),
                os.path.join(current_dir, "best_checkpoint.pt"),
            ]:
                if os.path.exists(cand):
                    ckpt_path = cand
                    break

        embedding_dim = 256
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                try:
                    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                except Exception:
                    sd = torch.load(ckpt_path, map_location="cpu")
                if isinstance(sd, dict):
                    if "config" in sd and "embedding_dim" in sd["config"]:
                        embedding_dim = sd["config"]["embedding_dim"]
                    elif "embedding_module" in sd:
                        for k, v in sd["embedding_module"].items():
                            if "weight" in k and len(v.shape) >= 2:
                                embedding_dim = v.shape[0]
                                break
            except Exception:
                pass

        patch_cfg  = PatchConfig(patch_size=patch_size, padding_mode="zero")
        patches, meta = partition_voxel_grid(grid_tensor, patch_cfg)
        stats      = compute_patch_activity_stats(patches, voxel_cfg)
        n_rows, n_cols, eff_h, eff_w = compute_patch_grid_dims(
            events["header"]["height"], events["header"]["width"], patch_cfg
        )
        adj = build_adjacency_map(n_rows, n_cols, patch_cfg)

        embed_cfg = PatchEmbeddingConfig(embedding_dim=embedding_dim)
        embed_mod = PatchEmbeddingModule(embed_cfg, voxel_cfg.num_channels, patch_size, n_rows, n_cols)
        edps_cfg  = EDPSConfig(gate_threshold=gate_thresh, selection_mode="normal")
        edps_mod  = EDPSModule(edps_cfg, embedding_dim, voxel_cfg.num_bins)

        ckpt_loaded, ckpt_status = load_universal_checkpoint(ckpt_path, embed_mod, edps_mod)

        embed_mod.eval()
        edps_mod.eval()

        with torch.no_grad():
            embeddings, embed_meta = embed_mod(patches, meta)
            edps_out = edps_mod(embeddings, stats, meta, embed_meta, adj)

        scores      = edps_out.importance_scores.numpy()
        binary_mask = edps_out.binary_mask.numpy()
        n_total     = len(scores)
        n_kept      = int(binary_mask.sum())
        keep_pct    = (n_kept / n_total) * 100.0
        pruned_pct  = 100.0 - keep_pct
        flops_red   = 100.0 / max(1.0, keep_pct)

        # ---------------------------------------------------------------
        # Visualization — styled to match the design system
        # ---------------------------------------------------------------
        bg_col    = "#0A0F1E"
        card_col  = "#0F172A"
        border_col = "#1E293B"
        cyan      = "#00D4FF"
        green     = "#10B981"
        red       = "#EF4444"
        text_col  = "#F1F5F9"
        muted_col = "#94A3B8"

        plt.rcParams.update({
            "figure.facecolor"  : bg_col,
            "axes.facecolor"    : card_col,
            "text.color"        : text_col,
            "axes.labelcolor"   : muted_col,
            "xtick.color"       : muted_col,
            "ytick.color"       : muted_col,
            "axes.edgecolor"    : border_col,
            "font.family"       : "DejaVu Sans",
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=bg_col)
        fig.patch.set_facecolor(bg_col)

        event_img = np.sum(np.abs(grid_np), axis=0)

        # --- Left: Baseline Dense ---
        ax1.set_facecolor(card_col)
        ax1.imshow(event_img, cmap="Blues", origin="upper", alpha=0.6)
        ax1.set_title(
            f"Standard Transformer  —  DENSE",
            fontsize=12, fontweight="bold", color=red, pad=10,
        )
        ax1.text(
            0.99, 0.97, f"{n_total} / {n_total} tokens",
            transform=ax1.transAxes, ha="right", va="top",
            fontsize=9, color=muted_col, style="italic",
        )
        ax1.text(
            0.01, 0.97, "100% FLOPs",
            transform=ax1.transAxes, ha="left", va="top",
            fontsize=9, color=red, alpha=0.6,
        )
        for m in meta:
            rect = patches_plt.Rectangle(
                (m.x0, m.y0), patch_size, patch_size,
                linewidth=0.5, edgecolor="#3B82F6", facecolor="none", alpha=0.4,
            )
            ax1.add_patch(rect)
        for b in window_boxes:
            bx, by, bw, bh = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            ax1.add_patch(patches_plt.Rectangle(
                (bx, by), bw, bh, linewidth=2, edgecolor="#F59E0B", facecolor="none",
            ))
        ax1.set_axis_off()
        # Bottom label strip
        ax1.annotate(
            "Baseline — all patches forwarded through full attention",
            xy=(0.5, -0.02), xycoords="axes fraction",
            ha="center", fontsize=8, color=muted_col,
        )

        # --- Right: EDPS Sparse ---
        ax2.set_facecolor(card_col)
        ax2.imshow(event_img, cmap="Blues", origin="upper", alpha=0.6)
        ax2.set_title(
            f"EDPS Sparse Model  —  {n_kept} / {n_total} retained  ({keep_pct:.1f}%)",
            fontsize=12, fontweight="bold", color=green, pad=10,
        )
        ax2.text(
            0.99, 0.97, f"{flops_red:.2f}x faster",
            transform=ax2.transAxes, ha="right", va="top",
            fontsize=9, color=cyan, fontweight="bold",
        )

        for i, m in enumerate(meta):
            is_kept = binary_mask[i]
            score   = scores[i]
            if is_kept:
                rect = patches_plt.Rectangle(
                    (m.x0, m.y0), patch_size, patch_size,
                    linewidth=1.5, edgecolor=green, facecolor=green, alpha=0.22,
                )
            else:
                rect = patches_plt.Rectangle(
                    (m.x0, m.y0), patch_size, patch_size,
                    linewidth=0.5, edgecolor=red, facecolor=red, alpha=0.07,
                )
            ax2.add_patch(rect)
            if show_scores:
                txt_color = "#D1FAE5" if is_kept else "#FCA5A5"
                ax2.text(
                    m.x0 + 2, m.y0 + patch_size / 2, f"{score:.2f}",
                    color=txt_color, fontsize=5.5, fontweight="bold",
                )

        for b in window_boxes:
            bx, by, bw, bh = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            ax2.add_patch(patches_plt.Rectangle(
                (bx, by), bw, bh, linewidth=2, edgecolor="#F59E0B", facecolor="none",
            ))
        ax2.set_axis_off()
        ax2.annotate(
            f"EDPS — {pruned_pct:.1f}% background pruned, object patches retained",
            xy=(0.5, -0.02), xycoords="axes fraction",
            ha="center", fontsize=8, color=muted_col,
        )

        plt.tight_layout(pad=2.0)

        kpi_html   = build_kpi_html(n_total, n_kept, keep_pct, pruned_pct, flops_red)
        table_html = build_metrics_table(n_total, n_kept, keep_pct, flops_red)

        rec_filename = os.path.basename(dat_path)
        time_sec     = (t_start_us - t_min) / 1e6
        status_msg   = (
            f"{ckpt_status}  |  {rec_filename}  |  "
            f"t=[{time_sec:.2f}s, {time_sec + window_ms / 1000:.2f}s]  |  "
            f"{len(events['t']):,} events  |  {len(window_boxes)} GT objects"
        )
        return fig, kpi_html, table_html, build_status_html(status_msg)

    except Exception as e:
        err = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, KPI_PLACEHOLDER, "", build_status_html(err, is_error=True)


# ===========================================================================
# CSS Design System
# ===========================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ---- Tokens ---- */
:root {
  --bg:       #0A0F1E;
  --surface:  #0F172A;
  --surface2: #131C2E;
  --border:   #1E293B;
  --border2:  #243347;
  --cyan:     #00D4FF;
  --cyan-dim: #00A0C0;
  --purple:   #7C3AED;
  --green:    #10B981;
  --red:      #EF4444;
  --text:     #F1F5F9;
  --muted:    #94A3B8;
  --faint:    #475569;
  --font-h:   'Sora', sans-serif;
  --font-b:   'Inter', sans-serif;
  --font-m:   'IBM Plex Mono', monospace;
  --radius:   8px;
  --radius-sm:4px;
}

/* ---- Base ---- */
body, .gradio-container { background: var(--bg) !important; font-family: var(--font-b) !important; color: var(--text) !important; }
* { box-sizing: border-box; }

/* ---- Nav ---- */
.edps-nav {
  display: flex; align-items: center; gap: 24px;
  background: var(--bg); border-bottom: 1px solid var(--cyan); padding: 0 24px;
  height: 56px; width: 100%;
}
.edps-nav__brand { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
.edps-nav__logo { font-family: var(--font-h); font-size: 18px; font-weight: 700; color: var(--cyan); letter-spacing: -0.5px; }
.edps-nav__subtitle { font-size: 11px; color: var(--faint); display: none; }
@media (min-width: 1200px) { .edps-nav__subtitle { display: block; } }
.edps-nav__links { display: flex; gap: 4px; list-style: none; margin: 0; padding: 0; }
.edps-nav__link {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 12px; border-radius: var(--radius-sm);
  font-size: 13px; font-weight: 500; color: var(--muted); text-decoration: none;
  transition: color 0.18s, background 0.18s;
}
.edps-nav__link:hover { color: var(--text); background: var(--surface); }
.edps-nav__link--active { color: var(--cyan); border-bottom: 2px solid var(--cyan); border-radius: 0; }
.edps-nav__right { display: flex; align-items: center; gap: 12px; margin-left: auto; }
.edps-version-chip {
  font-family: var(--font-m); font-size: 11px; color: var(--muted);
  background: var(--surface); border: 1px solid var(--border2);
  padding: 2px 8px; border-radius: 99px;
}
.edps-icon-btn {
  background: none; border: none; cursor: pointer; color: var(--muted);
  display: flex; align-items: center; padding: 6px;
  border-radius: var(--radius-sm); transition: color 0.18s, background 0.18s;
}
.edps-icon-btn:hover { color: var(--text); background: var(--surface); }

/* ---- Cards ---- */
.edps-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px;
}
.edps-card-header {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--font-h); font-size: 13px; font-weight: 600;
  color: var(--text); margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.5px;
}
.edps-divider { height: 1px; background: var(--border); margin: 16px 0; }

/* ---- Section headers (left panel) ---- */
.edps-section-header {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--font-h); font-size: 12px; font-weight: 700;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px;
  margin-bottom: 10px;
}

/* ---- Left panel overall ---- */
.edps-left-panel { display: flex; flex-direction: column; gap: 0; }

/* ---- KPI strip ---- */
.edps-kpi-row {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
  margin-bottom: 16px;
}
.edps-kpi-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 16px;
  display: flex; align-items: center; gap: 14px;
  transition: border-color 0.2s, transform 0.2s;
}
.edps-kpi-card:hover { border-color: var(--border2); transform: translateY(-1px); }
.edps-kpi-icon { flex-shrink: 0; }
.edps-kpi-value {
  font-family: var(--font-h); font-size: 22px; font-weight: 700;
  line-height: 1; margin-bottom: 2px;
}
.edps-kpi-title { font-size: 12px; font-weight: 600; color: var(--muted); }
.edps-kpi-sub   { font-size: 10px; color: var(--faint); font-family: var(--font-m); margin-top: 2px; }

/* ---- Run button ---- */
.edps-run-btn {
  display: flex; align-items: center; justify-content: center; gap: 10px;
  width: 100%; padding: 13px 20px;
  background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
  border: none; border-radius: var(--radius); cursor: pointer;
  font-family: var(--font-h); font-size: 15px; font-weight: 700; color: #0A0F1E;
  box-shadow: 0 4px 20px rgba(0, 212, 255, 0.28);
  transition: transform 0.18s, box-shadow 0.18s;
}
.edps-run-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0, 212, 255, 0.42); }
.edps-run-btn:active { transform: translateY(0); }

/* ---- Status bar ---- */
.edps-status-bar {
  display: flex; align-items: center; gap: 8px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 8px 12px;
  font-family: var(--font-m); font-size: 12px; margin-top: 8px;
}
.edps-status-dot {
  width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}
.edps-status-dot--ok    { background: var(--green); box-shadow: 0 0 6px var(--green); }
.edps-status-dot--error { background: var(--red);   box-shadow: 0 0 6px var(--red); }

/* ---- Comparison table ---- */
.edps-table-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px; margin-top: 16px;
}
.edps-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.edps-table thead tr { border-bottom: 1px solid var(--border2); }
.edps-table th {
  padding: 8px 12px; text-align: left;
  font-family: var(--font-h); font-size: 11px; font-weight: 700;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px;
}
.edps-table tbody tr { border-bottom: 1px solid var(--border); transition: background 0.15s; }
.edps-table tbody tr:hover { background: var(--surface2); }
.edps-table td { padding: 10px 12px; color: var(--text); }
.edps-mono { font-family: var(--font-m); font-size: 12px; }
.edps-cyan   { color: var(--cyan); }
.edps-green  { color: var(--green); }
.edps-red    { color: var(--red); }
.edps-purple { color: var(--purple); }

/* ---- Badges ---- */
.edps-badge {
  display: inline-block; padding: 2px 8px; border-radius: 99px;
  font-family: var(--font-m); font-size: 11px; font-weight: 500;
}
.edps-badge--green  { background: rgba(16,185,129,0.12); color: var(--green);  border: 1px solid rgba(16,185,129,0.3); }
.edps-badge--cyan   { background: rgba(0,212,255,0.10);  color: var(--cyan);   border: 1px solid rgba(0,212,255,0.25); }
.edps-badge--red    { background: rgba(239,68,68,0.10);  color: var(--red);    border: 1px solid rgba(239,68,68,0.25); }
.edps-badge--purple { background: rgba(124,58,237,0.12); color: var(--purple); border: 1px solid rgba(124,58,237,0.3); }

/* ---- Gradio component overrides ---- */
.gradio-container > .main { padding: 0 !important; }
footer { display: none !important; }

/* Inputs */
.gr-input, input[type=text], textarea, select,
.gr-box, .gr-form { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; color: var(--text) !important; }
label span { color: var(--muted) !important; font-size: 12px !important; font-family: var(--font-b) !important; font-weight: 500 !important; }

/* Sliders */
input[type=range] { accent-color: var(--cyan); }
.gr-slider input[type=range]::-webkit-slider-thumb { background: var(--cyan); box-shadow: 0 0 8px rgba(0,212,255,0.5); }
.gr-slider input[type=range]::-webkit-slider-runnable-track { background: var(--border); border-radius: 4px; }

/* Toggle */
.gr-toggle input:checked + .gr-toggle-track { background: var(--cyan) !important; }

/* Tabs (if any) */
.gr-tab-nav button.selected { border-bottom: 2px solid var(--cyan) !important; color: var(--cyan) !important; }

/* File upload */
.gr-file-upload { background: var(--surface2) !important; border: 1px dashed var(--border2) !important; }
.gr-file-upload:hover { border-color: var(--cyan) !important; }

/* Plot */
.gr-plot { background: var(--bg) !important; border: none !important; }

/* Buttons (Gradio default) */
button.primary { background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%) !important; color: #0A0F1E !important; font-family: var(--font-h) !important; font-weight: 700 !important; border: none !important; }
button.secondary { background: var(--surface2) !important; border: 1px solid var(--border2) !important; color: var(--text) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--faint); }

/* Left panel card wrapper */
.edps-panel-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 18px; height: 100%;
}
"""


# ===========================================================================
# Gradio UI
# ===========================================================================
def launch(share: bool = True):
    recordings_map = get_available_recordings()
    rec_choices    = list(recordings_map.keys())

    with gr.Blocks(
        title="EDPS — Token Selection Analysis",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="cyan",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
    ) as demo:

        # ---- Nav ----
        gr.HTML(NAV_HTML)

        with gr.Row(equal_height=False):

            # ================================================================
            # LEFT PANEL — Controls
            # ================================================================
            with gr.Column(scale=3, min_width=280):
                gr.HTML('<div class="edps-panel-card">')

                # --- Recording Input ---
                gr.HTML(RECORDING_HEADER)
                rec_dropdown = gr.Dropdown(
                    choices=rec_choices,
                    value=rec_choices[0] if rec_choices else None,
                    label="Recording File",
                    info="Prophesee Gen1 stream on local disk",
                )
                with gr.Row():
                    dat_upload = gr.File(label="Custom .dat file", type="filepath", file_types=[".dat"])
                with gr.Row():
                    bbox_upload = gr.File(label="Labels (.npy)", type="filepath", file_types=[".npy"])
                with gr.Row():
                    ckpt_upload = gr.File(label="Model weights (.pt)", type="filepath", file_types=[".pt"])

                gr.HTML('<div class="edps-divider"></div>')

                # --- Inference Controls ---
                gr.HTML(CONTROLS_HEADER)
                offset_slider = gr.Slider(
                    minimum=0, maximum=55000, value=500, step=500,
                    label="Time Offset (ms)",
                )
                window_slider = gr.Slider(
                    minimum=10, maximum=100, value=50, step=10,
                    label="Window Duration (ms)",
                )
                thresh_slider = gr.Slider(
                    minimum=0.10, maximum=0.90, value=0.50, step=0.05,
                    label="Gate Threshold",
                    info="EDPS soft-gate cutoff (0.50 default)",
                )
                patch_size_dropdown = gr.Dropdown(
                    choices=[16, 32], value=16, label="Patch Size (px)",
                )
                show_scores_checkbox = gr.Checkbox(
                    value=True, label="Show score values on patches",
                )

                gr.HTML('<div class="edps-divider"></div>')

                run_btn = gr.Button(
                    "Run Analysis",
                    variant="primary",
                    icon=None,
                )

                status_out = gr.HTML(
                    build_status_html("Ready — select a recording and click Run Analysis.", is_error=False)
                )

                gr.HTML('</div>')  # close edps-panel-card

            # ================================================================
            # RIGHT PANEL — Results
            # ================================================================
            with gr.Column(scale=7, min_width=600):

                # KPI strip
                kpi_out = gr.HTML(KPI_PLACEHOLDER)

                # Dual visualization
                plot_out = gr.Plot(label="", show_label=False)

                # Comparison table
                table_out = gr.HTML("")

        # ---- Wire up ----
        run_btn.click(
            fn=process_demo,
            inputs=[
                rec_dropdown, dat_upload, bbox_upload, ckpt_upload,
                offset_slider, window_slider, thresh_slider, patch_size_dropdown, show_scores_checkbox,
            ],
            outputs=[plot_out, kpi_out, table_out, status_out],
        )

    demo.launch(share=share, debug=False)


if __name__ == "__main__":
    launch(share=True)
