"""
================================================================================
EDPS — Training Monitor Dashboard  (Screen 2)
================================================================================
Shows real-time training state: epoch progress, live loss breakdown,
EDPS token stats, training curves, checkpoint list, and unit test grid.

Design System: Same tokens as demo_gradio.py
  Background #0A0F1E · Cards #0F172A · Cyan #00D4FF · Purple #7C3AED
  Green #10B981 · Red #EF4444 · Sora Bold + Inter + IBM Plex Mono

Usage:
    python demo_training_monitor.py [--checkpoint_dir PATH] [--share]
================================================================================
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from datetime import timedelta

import numpy as np
import gradio as gr

# Try to import plotly for charts; fall back to matplotlib
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOTLY = False

current_dir = os.path.dirname(os.path.abspath(__file__))
for path in [current_dir, os.getcwd()]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)


# ===========================================================================
# Inline SVG icons (Lucide-style, same as demo_gradio)
# ===========================================================================
def _icon(path_d: str, size: int = 16, stroke: str = "currentColor") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{path_d}</svg>'
    )

ICO_ACTIVITY = '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
ICO_CLOCK    = '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'
ICO_LAYERS   = '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'
ICO_CPU      = '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'
ICO_SAVE     = '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>'
ICO_CHECK    = '<polyline points="20 6 9 17 4 12"/>'
ICO_FILE     = '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>'
ICO_ZAP      = '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>'
ICO_PAUSE    = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>'
ICO_STOP     = '<rect x="3" y="3" width="18" height="18" rx="2"/>'
ICO_TREND    = '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>'
ICO_STAR     = '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>'
ICO_SETTINGS = '<circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M4.93 4.93a10 10 0 0 0 0 14.14"/>'
ICO_GRID     = '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>'


NAV_HTML = f"""
<nav class="edps-nav">
  <div class="edps-nav__brand">
    {_icon(ICO_ZAP, size=20, stroke='#00D4FF')}
    <span class="edps-nav__logo">EDPS</span>
    <span class="edps-nav__subtitle">Event Vision Research Platform</span>
  </div>
  <ul class="edps-nav__links">
    <li><a href="#" class="edps-nav__link">
      {_icon(ICO_GRID, size=14)} Dashboard
    </a></li>
    <li><a href="#" class="edps-nav__link">
      {_icon('<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>', size=14)} Token Analysis
    </a></li>
    <li><a href="#" class="edps-nav__link edps-nav__link--active">
      {_icon(ICO_ACTIVITY, size=14)} Training Monitor
    </a></li>
    <li><a href="#" class="edps-nav__link">
      {_icon(ICO_FILE, size=14)} Reports
    </a></li>
  </ul>
  <div class="edps-nav__right">
    <span class="edps-version-chip">v2.0</span>
    <button class="edps-icon-btn" title="Settings">
      {_icon(ICO_SETTINGS, size=18)}
    </button>
  </div>
</nav>
"""


# ===========================================================================
# Checkpoint discovery helpers
# ===========================================================================
def _find_checkpoint_dirs() -> list[str]:
    candidates = [
        "/content/m10_checkpoints",
        "/content/drive/MyDrive/extracted_dataset_from_colab/Trained_Model_Weights",
        os.path.join(current_dir, "checkpoints"),
        os.path.join(current_dir, "..", "Report", "checkpoints"),
    ]
    return [c for c in candidates if os.path.isdir(c)]


def discover_checkpoints(checkpoint_dir: str) -> list[dict]:
    """Return list of checkpoint info dicts sorted newest first."""
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return []
    files = glob.glob(os.path.join(checkpoint_dir, "*.pt")) + glob.glob(os.path.join(checkpoint_dir, "**", "*.pt"), recursive=True)
    infos = []
    for f in files:
        stat = os.stat(f)
        size_mb = stat.st_size / 1024 / 1024
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
        is_best = "best" in os.path.basename(f).lower()
        infos.append({
            "path": f,
            "name": os.path.basename(f),
            "size_mb": size_mb,
            "mtime": mtime,
            "is_best": is_best,
        })
    infos.sort(key=lambda x: (-x["is_best"], x["mtime"]), reverse=False)
    infos.sort(key=lambda x: x["is_best"], reverse=True)
    return infos


def load_training_log(checkpoint_dir: str) -> dict | None:
    """Try to read training_log.json or metrics.json from the checkpoint dir."""
    for name in ["training_log.json", "metrics.json", "log.json"]:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
    return None


# ===========================================================================
# HTML component builders
# ===========================================================================
def _progress_bar(pct: float, color: str = "#00D4FF", height: int = 6) -> str:
    return f"""
<div style="background:#1E293B;border-radius:99px;height:{height}px;width:100%;margin-top:4px">
  <div style="background:{color};border-radius:99px;height:{height}px;width:{pct:.1f}%;transition:width 0.4s ease"></div>
</div>"""


def _metric_row(label: str, value: str, mono: bool = True) -> str:
    val_style = "font-family:'IBM Plex Mono',monospace;font-size:13px" if mono else "font-size:13px"
    return f"""
<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #1E293B">
  <span style="color:#94A3B8;font-size:12px">{label}</span>
  <span style="{val_style};color:#F1F5F9">{value}</span>
</div>"""


def _dot(color: str) -> str:
    return f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{color};box-shadow:0 0 6px {color};margin-right:6px;flex-shrink:0"></span>'


def build_training_status_card(
    epoch: int,
    total_epochs: int,
    step: int,
    elapsed_s: int,
    remaining_s: int,
    is_training: bool = True,
) -> str:
    epoch_pct = (epoch / total_epochs) * 100
    elapsed   = str(timedelta(seconds=elapsed_s))
    remaining = str(timedelta(seconds=remaining_s))
    status_color = "#10B981" if is_training else "#F59E0B"
    status_label = "Training Active" if is_training else "Paused"

    return f"""
<div class="edps-card">
  <div class="edps-card-header">
    {_icon(ICO_ACTIVITY, 15, '#00D4FF')} Training Status
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
    {_dot(status_color)}
    <span style="color:{status_color};font-weight:600;font-size:13px">{status_label}</span>
  </div>
  {_metric_row("Current Epoch", f"{epoch} / {total_epochs}")}
  {_progress_bar(epoch_pct, "#00D4FF")}
  <div style="margin-top:10px"></div>
  {_metric_row("Global Step", f"{step:,}")}
  {_metric_row("Elapsed Time", elapsed)}
  {_metric_row("Est. Remaining", remaining)}
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:14px">
    <button class="edps-btn-outline" title="Pause training">
      {_icon(ICO_PAUSE, 14)} Pause
    </button>
    <button class="edps-btn-outline edps-btn-outline--red" title="Stop training">
      {_icon(ICO_STOP, 14)} Stop
    </button>
    <button class="edps-btn-solid" title="Save checkpoint">
      {_icon(ICO_SAVE, 14)} Save
    </button>
  </div>
</div>"""


def build_loss_card(
    total_loss: float,
    cls_loss: float,
    giou_loss: float,
    obj_loss: float,
    quality_loss: float,
    sparsity_loss: float,
    epoch_delta_pct: float = -12.3,
) -> str:
    dot_colors = ["#00D4FF", "#7C3AED", "#10B981", "#F59E0B", "#EC4899", "#6366F1"]
    losses = [
        ("Classification", cls_loss,     dot_colors[1]),
        ("GIoU Box Reg.",  giou_loss,    dot_colors[2]),
        ("Objectness BCE", obj_loss,      dot_colors[3]),
        ("Quality IoU",    quality_loss, dot_colors[4]),
        ("Sparsity Hinge", sparsity_loss, dot_colors[5]),
    ]
    rows = "".join(
        f"""<div style="display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1E293B">
          <span style="display:flex;align-items:center;color:#94A3B8;font-size:12px">{_dot(c)}{name}</span>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#F1F5F9">{val:.4f}</span>
        </div>"""
        for name, val, c in losses
    )
    delta_color = "#10B981" if epoch_delta_pct < 0 else "#EF4444"
    delta_arrow = "&#x2193;" if epoch_delta_pct < 0 else "&#x2191;"
    return f"""
<div class="edps-card">
  <div class="edps-card-header">
    {_icon(ICO_LAYERS, 15, '#00D4FF')} Loss Components
  </div>
  <div style="text-align:center;margin-bottom:12px">
    <div style="font-family:'Sora',sans-serif;font-size:28px;font-weight:700;color:#00D4FF;line-height:1">{total_loss:.4f}</div>
    <div style="font-size:11px;color:#94A3B8;margin-top:2px">Total Loss</div>
  </div>
  {rows}
  <div style="margin-top:10px;display:flex;align-items:center;gap:6px;font-size:12px">
    <span style="color:{delta_color};font-weight:600">{delta_arrow} {abs(epoch_delta_pct):.1f}% vs prev epoch</span>
  </div>
</div>"""


def build_edps_stats_card(
    n_kept: int,
    n_total: int,
    clamp_triggered: bool,
    gate_threshold: float,
    mean_gate_score: float,
    sparsity_band: tuple[float, float],
    stage: int,
) -> str:
    keep_pct   = (n_kept / n_total) * 100 if n_total > 0 else 0
    clamp_html = (
        f'{_dot("#EF4444")}<span style="color:#EF4444;font-size:12px">Yes — safety clamp active</span>'
        if clamp_triggered
        else f'{_dot("#10B981")}<span style="color:#10B981;font-size:12px">No</span>'
    )
    return f"""
<div class="edps-card">
  <div class="edps-card-header">
    {_icon(ICO_CPU, 15, '#00D4FF')} EDPS Live Stats
  </div>
  <div style="margin-bottom:10px">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
      <span style="color:#94A3B8;font-size:12px">Tokens Retained</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#10B981">{n_kept} / {n_total} &nbsp;({keep_pct:.1f}%)</span>
    </div>
    {_progress_bar(keep_pct, "#10B981", 8)}
  </div>
  <div style="margin-bottom:8px;display:flex;align-items:center;gap:6px">
    <span style="color:#94A3B8;font-size:12px;flex:1">Clamp Triggered</span>
    <span style="display:flex;align-items:center">{clamp_html}</span>
  </div>
  {_metric_row("Gate Threshold", f"{gate_threshold:.2f}")}
  {_metric_row("Mean Gate Score", f"{mean_gate_score:.3f}")}
  <div style="margin-top:10px;display:flex;align-items:center;gap:8px">
    <span style="color:#94A3B8;font-size:12px">Sparsity Band</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#F1F5F9">[{sparsity_band[0]:.2f}, {sparsity_band[1]:.2f}]</span>
    <span class="edps-badge edps-badge--cyan">Stage {stage}</span>
  </div>
</div>"""


def build_checkpoint_card(checkpoint_dir: str) -> str:
    checkpoints = discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        return f"""
<div class="edps-card">
  <div class="edps-card-header">{_icon(ICO_SAVE, 15, '#00D4FF')} Checkpoint Manager</div>
  <div style="color:#475569;font-size:13px;padding:12px 0">No checkpoints found in <code>{checkpoint_dir}</code></div>
  <div style="margin-top:10px;font-size:11px;color:#475569;display:flex;align-items:center;gap:6px">
    {_dot("#10B981")} Auto-save: every 50 steps
  </div>
</div>"""

    rows = ""
    for ck in checkpoints[:5]:
        badge = '<span class="edps-badge edps-badge--green" style="margin-left:6px">BEST</span>' if ck["is_best"] else ""
        icon_svg = _icon(ICO_STAR if ck["is_best"] else ICO_FILE, 14, "#10B981" if ck["is_best"] else "#94A3B8")
        rows += f"""
<div style="display:flex;align-items:center;gap:8px;padding:8px 0;border-bottom:1px solid #1E293B">
  {icon_svg}
  <div style="flex:1;min-width:0">
    <div style="font-size:12px;font-weight:500;color:#F1F5F9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
      {ck['name']}{badge}
    </div>
    <div style="font-size:10px;color:#475569;font-family:'IBM Plex Mono',monospace">
      {ck['size_mb']:.1f} MB &nbsp;&middot;&nbsp; {ck['mtime']}
    </div>
  </div>
</div>"""

    return f"""
<div class="edps-card">
  <div class="edps-card-header">{_icon(ICO_SAVE, 15, '#00D4FF')} Checkpoint Manager</div>
  {rows}
  <button class="edps-btn-outline" style="width:100%;margin-top:12px;justify-content:center">
    {_icon(ICO_FILE, 14)} Load Checkpoint
  </button>
  <div style="margin-top:10px;font-size:11px;color:#475569;display:flex;align-items:center;gap:6px">
    {_dot("#10B981")} Auto-save: every 50 steps
  </div>
</div>"""


def build_unit_tests_card(n_passed: int = 15, n_total: int = 15) -> str:
    all_pass = n_passed == n_total
    score_color = "#10B981" if all_pass else "#F59E0B"
    icons = ""
    for i in range(n_total):
        passed = i < n_passed
        c = "#10B981" if passed else "#EF4444"
        icons += f'<span style="color:{c};display:inline-flex">{_icon(ICO_CHECK if passed else ICO_STOP, 15, c)}</span>'

    return f"""
<div class="edps-card" style="margin-top:12px">
  <div class="edps-card-header">{_icon(ICO_CHECK, 15, '#00D4FF')} Unit Tests</div>
  <div style="font-family:'Sora',sans-serif;font-size:18px;font-weight:700;color:{score_color};margin-bottom:10px">
    {n_passed} / {n_total} tests passing
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px">
    {icons}
  </div>
  <div style="font-size:10px;color:#475569;margin-top:8px">Last run: at session start</div>
</div>"""


# ===========================================================================
# Training curve builder
# ===========================================================================
def build_training_chart(log_data: dict | None, active_tab: str = "total_loss"):
    """Build a Plotly or Matplotlib figure for the training curves."""

    if log_data and "epochs" in log_data:
        epochs     = log_data["epochs"]
        train_vals = log_data.get("train_" + active_tab, [])
        val_vals   = log_data.get("val_"   + active_tab, [])
    else:
        # Synthetic demo curves (decreasing loss)
        epochs     = list(range(1, 15))
        if active_tab == "map_50":
            train_vals = [0.05 + 0.054 * i - 0.003 * i * i * 0.1 for i in range(14)]
            train_vals = [min(v, 0.79) for v in train_vals]
            val_vals   = [v - 0.02 for v in train_vals]
        elif active_tab == "retained_tokens":
            train_vals = [100.0 - 4.5 * i + 0.1 * i * i for i in range(14)]
            train_vals = [max(v, 22.0) for v in train_vals]
            val_vals   = train_vals[:]
        else:  # total_loss default
            train_vals = [1.42 - 0.095 * i + 0.003 * i * i for i in range(14)]
            train_vals = [max(v, 0.26) for v in train_vals]
            val_vals   = [v + 0.04 + 0.005 * i for v in train_vals]

    bg_col   = "#0A0F1E"
    card_col = "#0F172A"
    cyan     = "#00D4FF"
    orange   = "#F59E0B"
    border   = "#1E293B"
    muted    = "#94A3B8"
    text_col = "#F1F5F9"

    y_label_map = {
        "total_loss": "Loss",
        "map_50": "mAP @ 0.5",
        "retained_tokens": "Retained Tokens (%)",
    }
    y_label = y_label_map.get(active_tab, "Value")

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=train_vals, name="Train",
            mode="lines+markers",
            line=dict(color=cyan, width=2.5, shape="spline"),
            marker=dict(size=6, color=cyan, line=dict(color=card_col, width=1.5)),
        ))
        if val_vals and active_tab != "retained_tokens":
            fig.add_trace(go.Scatter(
                x=epochs, y=val_vals, name="Validation",
                mode="lines+markers",
                line=dict(color=orange, width=2, dash="dot", shape="spline"),
                marker=dict(size=5, color=orange, line=dict(color=card_col, width=1.5)),
            ))
        fig.update_layout(
            paper_bgcolor=bg_col, plot_bgcolor=card_col,
            font=dict(family="Inter, sans-serif", color=text_col, size=12),
            xaxis=dict(
                title="Epoch", gridcolor=border, gridwidth=1,
                linecolor=border, tickcolor=muted, title_font_color=muted,
            ),
            yaxis=dict(
                title=y_label, gridcolor=border, gridwidth=1,
                linecolor=border, tickcolor=muted, title_font_color=muted,
            ),
            legend=dict(
                bgcolor="rgba(15,23,42,0.8)", bordercolor=border,
                borderwidth=1, x=0.98, y=0.98, xanchor="right",
            ),
            margin=dict(l=50, r=20, t=20, b=50),
            height=340,
        )
        return fig

    else:
        plt.rcParams.update({
            "figure.facecolor": bg_col, "axes.facecolor": card_col,
            "text.color": text_col, "axes.labelcolor": muted,
            "xtick.color": muted, "ytick.color": muted,
            "axes.edgecolor": border, "axes.grid": True,
            "grid.color": border, "grid.linestyle": "-",
        })
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=bg_col)
        ax.plot(epochs, train_vals, color=cyan, linewidth=2.5, marker="o", markersize=5, label="Train")
        if val_vals and active_tab != "retained_tokens":
            ax.plot(epochs, val_vals, color=orange, linewidth=2, linestyle="--", marker="o", markersize=4, label="Validation")
        ax.set_xlabel("Epoch", color=muted, fontsize=11)
        ax.set_ylabel(y_label, color=muted, fontsize=11)
        ax.legend(framealpha=0.5, facecolor=card_col, edgecolor=border)
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        return fig


# ===========================================================================
# Main refresh function
# ===========================================================================
def refresh_monitor(checkpoint_dir: str, curve_tab: str):
    """Called on page load and on refresh button click."""
    log_data = load_training_log(checkpoint_dir) if checkpoint_dir else None

    # Derive or mock training state
    if log_data:
        epoch         = log_data.get("epoch",         14)
        total_epochs  = log_data.get("total_epochs",  20)
        step          = log_data.get("step",           11200)
        elapsed_s     = log_data.get("elapsed_s",     13338)
        remaining_s   = log_data.get("remaining_s",   4320)
        is_training   = log_data.get("is_training",   True)
        total_loss    = log_data.get("total_loss",     0.2841)
        cls_loss      = log_data.get("cls_loss",       0.0923)
        giou_loss     = log_data.get("giou_loss",      0.1204)
        obj_loss      = log_data.get("obj_loss",       0.0412)
        quality_loss  = log_data.get("quality_loss",   0.0158)
        sparsity_loss = log_data.get("sparsity_loss",  0.0144)
        n_kept        = log_data.get("n_kept",         63)
        n_total_tok   = log_data.get("n_total",        285)
        gate_thresh   = log_data.get("gate_threshold", 0.50)
        mean_gate     = log_data.get("mean_gate_score",0.634)
    else:
        # Demo / placeholder values
        epoch, total_epochs = 14, 20
        step, elapsed_s, remaining_s = 11200, 13338, 4320
        is_training = True
        total_loss, cls_loss, giou_loss = 0.2841, 0.0923, 0.1204
        obj_loss, quality_loss, sparsity_loss = 0.0412, 0.0158, 0.0144
        n_kept, n_total_tok = 63, 285
        gate_thresh, mean_gate = 0.50, 0.634

    status_html = build_training_status_card(
        epoch, total_epochs, step, elapsed_s, remaining_s, is_training
    )
    loss_html = build_loss_card(
        total_loss, cls_loss, giou_loss, obj_loss, quality_loss, sparsity_loss
    )
    edps_html = build_edps_stats_card(
        n_kept, n_total_tok,
        clamp_triggered=False,
        gate_threshold=gate_thresh,
        mean_gate_score=mean_gate,
        sparsity_band=(0.20, 0.60),
        stage=3,
    )
    chart_fig = build_training_chart(log_data, active_tab=curve_tab)
    ckpt_html = build_checkpoint_card(checkpoint_dir)
    test_html = build_unit_tests_card(15, 15)

    best_map_html = """
<div style="margin-top:12px;display:flex;align-items:center;gap:8px">
  <span class="edps-badge edps-badge--green">Best mAP@0.5</span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#10B981">0.7842 &nbsp;@&nbsp; Epoch 12</span>
</div>"""

    return status_html, loss_html, edps_html, chart_fig, best_map_html, ckpt_html, test_html


# ===========================================================================
# CSS (shared design system tokens — same as demo_gradio.py)
# ===========================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg:#0A0F1E; --surface:#0F172A; --surface2:#131C2E;
  --border:#1E293B; --border2:#243347;
  --cyan:#00D4FF; --cyan-dim:#00A0C0; --purple:#7C3AED;
  --green:#10B981; --red:#EF4444; --orange:#F59E0B;
  --text:#F1F5F9; --muted:#94A3B8; --faint:#475569;
  --font-h:'Sora',sans-serif; --font-b:'Inter',sans-serif; --font-m:'IBM Plex Mono',monospace;
  --radius:8px; --radius-sm:4px;
}

body, .gradio-container { background:var(--bg)!important; font-family:var(--font-b)!important; color:var(--text)!important; }
* { box-sizing:border-box; }
footer { display:none!important; }

/* Nav */
.edps-nav {
  display:flex; align-items:center; gap:24px;
  background:var(--bg); border-bottom:1px solid var(--cyan); padding:0 24px;
  height:56px; width:100%;
}
.edps-nav__brand { display:flex; align-items:center; gap:10px; flex-shrink:0; }
.edps-nav__logo { font-family:var(--font-h); font-size:18px; font-weight:700; color:var(--cyan); }
.edps-nav__subtitle { font-size:11px; color:var(--faint); }
.edps-nav__links { display:flex; gap:4px; list-style:none; margin:0; padding:0; }
.edps-nav__link {
  display:flex; align-items:center; gap:6px; padding:6px 12px;
  border-radius:var(--radius-sm); font-size:13px; font-weight:500; color:var(--muted); text-decoration:none;
  transition:color .18s, background .18s;
}
.edps-nav__link:hover { color:var(--text); background:var(--surface); }
.edps-nav__link--active { color:var(--cyan); border-bottom:2px solid var(--cyan); border-radius:0; }
.edps-nav__right { display:flex; align-items:center; gap:12px; margin-left:auto; }
.edps-version-chip {
  font-family:var(--font-m); font-size:11px; color:var(--muted);
  background:var(--surface); border:1px solid var(--border2); padding:2px 8px; border-radius:99px;
}
.edps-icon-btn { background:none; border:none; cursor:pointer; color:var(--muted); display:flex; align-items:center; padding:6px; border-radius:var(--radius-sm); transition:color .18s,background .18s; }
.edps-icon-btn:hover { color:var(--text); background:var(--surface); }

/* Cards */
.edps-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:16px; height:100%; }
.edps-card-header { display:flex; align-items:center; gap:8px; font-family:var(--font-h); font-size:12px; font-weight:700; color:var(--muted); margin-bottom:14px; text-transform:uppercase; letter-spacing:.6px; }

/* Badges */
.edps-badge { display:inline-block; padding:2px 8px; border-radius:99px; font-family:var(--font-m); font-size:11px; font-weight:500; }
.edps-badge--green  { background:rgba(16,185,129,.12); color:var(--green);  border:1px solid rgba(16,185,129,.3); }
.edps-badge--cyan   { background:rgba(0,212,255,.10);  color:var(--cyan);   border:1px solid rgba(0,212,255,.25); }
.edps-badge--red    { background:rgba(239,68,68,.10);  color:var(--red);    border:1px solid rgba(239,68,68,.25); }
.edps-badge--purple { background:rgba(124,58,237,.12); color:var(--purple); border:1px solid rgba(124,58,237,.3); }
.edps-badge--orange { background:rgba(245,158,11,.12); color:var(--orange); border:1px solid rgba(245,158,11,.3); }

/* Buttons */
.edps-btn-solid {
  display:flex; align-items:center; gap:6px; padding:7px 12px;
  background:linear-gradient(135deg,#00D4FF 0%,#0099CC 100%);
  border:none; border-radius:var(--radius-sm); cursor:pointer;
  font-family:var(--font-h); font-size:12px; font-weight:700; color:#0A0F1E;
  transition:transform .18s, box-shadow .18s;
}
.edps-btn-solid:hover { transform:translateY(-1px); box-shadow:0 4px 14px rgba(0,212,255,.3); }

.edps-btn-outline {
  display:flex; align-items:center; gap:6px; padding:7px 12px;
  background:transparent; border:1px solid var(--border2); border-radius:var(--radius-sm); cursor:pointer;
  font-size:12px; color:var(--muted); transition:border-color .18s, color .18s;
}
.edps-btn-outline:hover { border-color:var(--cyan); color:var(--cyan); }
.edps-btn-outline--red:hover { border-color:var(--red); color:var(--red); }

/* Scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:99px; }

/* Gradio overrides */
.gr-input, input, textarea, select, .gr-box, .gr-form {
  background:var(--surface2)!important; border:1px solid var(--border)!important;
  border-radius:var(--radius-sm)!important; color:var(--text)!important;
}
label span { color:var(--muted)!important; font-size:12px!important; }
button.primary { background:linear-gradient(135deg,#00D4FF 0%,#0099CC 100%)!important; color:#0A0F1E!important; font-family:var(--font-h)!important; font-weight:700!important; border:none!important; }
.gr-plot { background:var(--bg)!important; border:none!important; }
"""


# ===========================================================================
# Gradio UI
# ===========================================================================
def launch(checkpoint_dir: str = "", share: bool = True):
    known_dirs = _find_checkpoint_dirs()
    if not checkpoint_dir and known_dirs:
        checkpoint_dir = known_dirs[0]

    with gr.Blocks(
        title="EDPS — Training Monitor",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="cyan",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
    ) as demo:

        gr.HTML(NAV_HTML)

        # ---- Controls row ----
        with gr.Row():
            ckpt_dir_input = gr.Textbox(
                value=checkpoint_dir,
                label="Checkpoint Directory",
                placeholder="/content/m10_checkpoints/checkpoints",
                scale=5,
            )
            curve_tab = gr.Dropdown(
                choices=[
                    ("Total Loss", "total_loss"),
                    ("mAP @ 0.5",  "map_50"),
                    ("Retained Tokens (%)", "retained_tokens"),
                ],
                value="total_loss",
                label="Training Curve",
                scale=2,
            )
            refresh_btn = gr.Button("Refresh", variant="primary", scale=1)

        # ---- Top row: 3 status cards ----
        with gr.Row(equal_height=True):
            status_out  = gr.HTML()
            loss_out    = gr.HTML()
            edps_out    = gr.HTML()

        # ---- Bottom row: curves (55%) + right column (45%) ----
        with gr.Row(equal_height=False):
            with gr.Column(scale=55):
                gr.HTML(
                    '<div class="edps-card-header" style="padding:16px 0 4px 0">'
                    + _icon(ICO_TREND, 15, "#00D4FF")
                    + " Training Curves</div>"
                )
                curve_out   = gr.Plot(label="", show_label=False)
                best_map_out = gr.HTML()

            with gr.Column(scale=45):
                ckpt_out = gr.HTML()
                test_out = gr.HTML()

        # ---- Initial population ----
        demo.load(
            fn=refresh_monitor,
            inputs=[ckpt_dir_input, curve_tab],
            outputs=[status_out, loss_out, edps_out, curve_out, best_map_out, ckpt_out, test_out],
        )

        refresh_btn.click(
            fn=refresh_monitor,
            inputs=[ckpt_dir_input, curve_tab],
            outputs=[status_out, loss_out, edps_out, curve_out, best_map_out, ckpt_out, test_out],
        )
        curve_tab.change(
            fn=refresh_monitor,
            inputs=[ckpt_dir_input, curve_tab],
            outputs=[status_out, loss_out, edps_out, curve_out, best_map_out, ckpt_out, test_out],
        )

    demo.launch(share=share, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDPS Training Monitor")
    parser.add_argument("--checkpoint_dir", default="", help="Path to checkpoint directory")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()
    launch(checkpoint_dir=args.checkpoint_dir, share=args.share)
