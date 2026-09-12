"""
================================================================================
Trainer (M10)
================================================================================
Orchestrates the full training loop: AMP forward/backward, gradient
clipping, cosine LR schedule with warmup, the EDPS sparsity curriculum,
per-epoch validation with mAP, checkpointing (periodic + best-separate),
early stopping, and automatic resume from the latest checkpoint.
================================================================================
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

from training_config import TrainingConfig
from optimizer_factory import build_optimizer
from scheduler_factory import build_scheduler
from sparsity_curriculum import get_current_sparsity_band
from edps_loss import sparsity_loss
from detection_losses import focal_loss, giou_loss, smooth_l1_ltrb_loss, objectness_bce_loss, quality_iou_loss
from map_metric import compute_map
from gate_score_logger import GateScoreLogger
from checkpoint_manager import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    periodic_checkpoint_path, best_checkpoint_path, save_training_config_snapshot, TrainingState,
)
from reproducibility import seed_everything


def _decode_boxes_from_ltrb(centers_xy: torch.Tensor, ltrb_raw: torch.Tensor, distance_scale: float) -> torch.Tensor:
    ltrb = torch.nn.functional.softplus(ltrb_raw) * distance_scale
    cx, cy = centers_xy[:, 0], centers_xy[:, 1]
    l, t, r, b = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]
    return torch.stack([cx - l, cy - t, cx + r, cy + b], dim=1)


@dataclass
class EpochSummary:
    epoch: int
    train_loss: float
    train_loss_components: Dict[str, float]
    val_map50: Optional[float]
    val_map50_95: Optional[float]
    avg_retained_tokens: float
    sparsity_band: Dict[str, float]
    n_unmatched_gt_fraction: float
    lr: float
    epoch_time_s: float


class Trainer:
    def __init__(
        self,
        embedding_module: nn.Module,
        edps_module: nn.Module,
        encoder: nn.Module,
        detection_head: nn.Module,
        train_loader,
        val_loader,
        config: TrainingConfig,
        num_classes: int = 2,
        distance_scale: float = 16.0,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is not available.")
        self.embedding_module = embedding_module.to(device)
        self.edps_module = edps_module.to(device)
        self.encoder = encoder.to(device)
        self.detection_head = detection_head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.num_classes = num_classes
        self.distance_scale = distance_scale
        self.device = device
        print(f"[M10] Trainer device: {self.device}")
        if self.device == "cuda":
            print(f"[M10] GPU: {torch.cuda.get_device_name(0)}")

        self.optimizer = build_optimizer(embedding_module, edps_module, encoder, detection_head, config)
        total_steps = max(1, len(train_loader)) * config.total_epochs // max(1, config.grad_accumulation_steps)
        self.scheduler = build_scheduler(self.optimizer, config, total_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(config.use_amp and device == "cuda"))

        self.state = TrainingState(epoch=0, global_step=0, best_metric=-1.0, best_epoch=-1)
        self.gate_logger = GateScoreLogger()
        self.history: List[EpochSummary] = []
        self._epochs_without_improvement = 0

    # ------------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, Any], sparsity_band):
        raw = batch["raw_predictions"]
        assignments = batch["assignments"]

        all_cls_logits_pos, all_cls_targets_pos = [], []
        all_pred_ltrb_pos, all_target_ltrb_pos = [], []
        all_obj_logits, all_obj_targets = [], []
        all_quality_logits_pos, all_pred_boxes_pos, all_target_boxes_pos = [], [], []
        n_unmatched_total, n_gt_total = 0, 0

        for i, assignment in enumerate(assignments):
            k = assignment.is_positive.shape[0]
            pos_mask = assignment.is_positive

            centers = torch.tensor(
                [[m.center_x, m.center_y] for m in batch["selected_metadata"][i]],
                dtype=torch.float32,
                device=self.device,
            )
            pred_boxes_all = _decode_boxes_from_ltrb(centers, raw["ltrb_raw"][i, :k], self.distance_scale)

            all_obj_logits.append(raw["objectness_logit"][i, :k, 0])
            all_obj_targets.append(
                pos_mask.to(
                    device=raw["objectness_logit"].device,
                    dtype=raw["objectness_logit"].dtype,
                ).float()
            )

            if pos_mask.any():
                all_cls_logits_pos.append(raw["class_logits"][i, :k][pos_mask])
                all_cls_targets_pos.append(assignment.target_class[pos_mask].to(self.device))
                all_pred_ltrb_pos.append(
                    torch.nn.functional.softplus(raw["ltrb_raw"][i, :k][pos_mask]) * self.distance_scale
                )

                # Assignment targets are produced by the dataset/target
                # assignment code and may still be CPU tensors. Keep every
                # target on the same device/dtype as the model predictions.
                all_target_ltrb_pos.append(
                    assignment.target_ltrb[pos_mask].to(
                        device=self.device,
                        dtype=raw["ltrb_raw"].dtype,
                    )
                )
                all_pred_boxes_pos.append(pred_boxes_all[pos_mask])
                all_target_boxes_pos.append(
                    assignment.target_box[pos_mask].to(
                        device=self.device,
                        dtype=pred_boxes_all.dtype,
                    )
                )
                all_quality_logits_pos.append(raw["quality_logit"][i, :k, 0][pos_mask])

            n_unmatched_total += assignment.n_unmatched_gt_boxes
            n_gt_total += assignment.n_gt_boxes

        cls_logits = torch.cat(all_cls_logits_pos) if all_cls_logits_pos else torch.zeros(
            0, self.num_classes, device=self.device
        )
        cls_targets = torch.cat(all_cls_targets_pos) if all_cls_targets_pos else torch.zeros(
            0, dtype=torch.long, device=self.device
        )
        pred_ltrb = torch.cat(all_pred_ltrb_pos) if all_pred_ltrb_pos else torch.zeros(
            0, 4, device=self.device
        )
        target_ltrb = torch.cat(all_target_ltrb_pos) if all_target_ltrb_pos else torch.zeros(
            0, 4, device=self.device
        )
        pred_boxes = torch.cat(all_pred_boxes_pos) if all_pred_boxes_pos else torch.zeros(
            0, 4, device=self.device
        )
        target_boxes = torch.cat(all_target_boxes_pos) if all_target_boxes_pos else torch.zeros(
            0, 4, device=self.device
        )
        obj_logits = torch.cat(all_obj_logits)
        obj_targets = torch.cat(all_obj_targets).to(
            device=obj_logits.device,
            dtype=obj_logits.dtype,
        )
        quality_logits = torch.cat(all_quality_logits_pos) if all_quality_logits_pos else torch.zeros(
            0, device=self.device
        )

        loss_cls = focal_loss(cls_logits, cls_targets, self.config.focal_alpha, self.config.focal_gamma)
        target_boxes = target_boxes.to(
            pred_boxes.device,
            dtype=pred_boxes.dtype,
        )

        loss_giou = giou_loss(pred_boxes, target_boxes)

        target_ltrb = target_ltrb.to(
            device=pred_ltrb.device,
            dtype=pred_ltrb.dtype,
        )
        # Compute smooth L1 on normalized LTRB distances (units of distance_scale / patch_size)
        loss_smooth_l1 = smooth_l1_ltrb_loss(
            pred_ltrb / self.distance_scale,
            target_ltrb / self.distance_scale,
        )
        loss_obj = objectness_bce_loss(obj_logits, obj_targets)
        loss_quality = quality_iou_loss(quality_logits, pred_boxes, target_boxes)

        sparsity_terms = [
            sparsity_loss(m["importance_scores"], sparsity_band.min_ratio, sparsity_band.max_ratio)
            for m in batch["meta"]
        ]
        loss_sparsity = torch.stack(sparsity_terms).mean()

        total = (
            self.config.weight_cls * loss_cls
            + self.config.weight_giou * loss_giou
            + self.config.weight_smooth_l1_aux * loss_smooth_l1
            + self.config.weight_objectness * loss_obj
            + self.config.weight_quality * loss_quality
            + self.config.lambda_sparsity * loss_sparsity
        )

        components = {
            "cls": float(loss_cls.item()), "giou": float(loss_giou.item()),
            "smooth_l1_aux": float(loss_smooth_l1.item()), "objectness": float(loss_obj.item()),
            "quality": float(loss_quality.item()), "sparsity": float(loss_sparsity.item()),
            "total": float(total.item()),
            "n_unmatched_gt_boxes": n_unmatched_total, "n_gt_boxes": n_gt_total,
        }
        return total, components

    # ------------------------------------------------------------------
    def train_one_epoch(self) -> Dict[str, Any]:
        self.embedding_module.train(); self.edps_module.train()
        self.encoder.train(); self.detection_head.train()

        progress = self.state.epoch / max(1, self.config.total_epochs)
        band = get_current_sparsity_band(progress, self.config)

        epoch_losses = []
        epoch_components = []
        retained_counts = []

        self.optimizer.zero_grad()
        total_steps = len(self.train_loader)
        for step, batch in enumerate(self.train_loader):
            if step == 0 or (step + 1) % 10 == 0 or (step + 1) == total_steps:
                print(
                    f"[M10] Epoch {self.state.epoch + 1}/{self.config.total_epochs} "
                    f"step {step + 1}/{total_steps}",
                    flush=True,
                )
            with torch.amp.autocast(self.device, enabled=self.scaler.is_enabled()):
                loss, components = self.compute_loss(batch, band)
                loss = loss / self.config.grad_accumulation_steps

            self.scaler.scale(loss).backward()

            for m in batch["meta"]:
                retained_counts.append(m["token_reduction_stats"]["n_retained"])
                self.gate_logger.log(m["importance_scores"], threshold=0.5, step=self.state.global_step)

            if (step + 1) % self.config.grad_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.embedding_module.parameters()) + list(self.edps_module.parameters())
                    + list(self.encoder.parameters()) + list(self.detection_head.parameters()),
                    self.config.grad_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.state.global_step += 1

                # Sync checkpoint to Drive every 50 steps
                if (step + 1) % 50 == 0:
                    try:
                        save_checkpoint(
                            best_checkpoint_path(self.config.checkpoint_dir),
                            self.embedding_module, self.edps_module, self.encoder, self.detection_head,
                            self.optimizer, self.scheduler, self.scaler, self.state, self.config,
                        )
                        print(f"💾 [Auto-Save Step {step + 1}] Checkpoint synced to Drive!", flush=True)
                    except Exception as e:
                        print(f"⚠️ Warning: Step auto-save failed: {e}", flush=True)

            epoch_losses.append(components["total"])
            epoch_components.append(components)

        avg_components = {
            k: float(np.mean([c[k] for c in epoch_components]))
            for k in epoch_components[0] if k not in ("n_unmatched_gt_boxes", "n_gt_boxes")
        }
        total_unmatched = sum(c["n_unmatched_gt_boxes"] for c in epoch_components)
        total_gt = sum(c["n_gt_boxes"] for c in epoch_components)

        return {
            "avg_loss": float(np.mean(epoch_losses)),
            "avg_components": avg_components,
            "avg_retained_tokens": float(np.mean(retained_counts)) if retained_counts else 0.0,
            "band": band,
            "unmatched_gt_fraction": (total_unmatched / total_gt) if total_gt else 0.0,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        self.embedding_module.eval(); self.edps_module.eval()
        self.encoder.eval(); self.detection_head.eval()

        all_predictions, all_gts = [], []
        all_scores_flat = []
        n_raw_preds = 0
        n_gt_boxes_total = 0

        for batch in self.val_loader:
            raw = batch["raw_predictions"]
            for i in range(len(batch["selected_metadata"])):
                k = len(batch["selected_metadata"][i])
                n_raw_preds += k
                n_gt_boxes_total += len(batch["boxes"][i])

                centers = torch.tensor(
                    [[m.center_x, m.center_y] for m in batch["selected_metadata"][i]],
                    dtype=torch.float32,
                    device=self.device,
                )
                boxes = _decode_boxes_from_ltrb(centers, raw["ltrb_raw"][i, :k], self.distance_scale)
                cls_probs = torch.softmax(raw["class_logits"][i, :k], dim=-1)
                obj = torch.sigmoid(raw["objectness_logit"][i, :k, 0])
                quality = torch.sigmoid(raw["quality_logit"][i, :k, 0])

                preds = []
                for j in range(k):
                    predicted_class = int(torch.argmax(cls_probs[j]).item())
                    score = float(obj[j].item() * quality[j].item() * cls_probs[j, predicted_class].item())
                    preds.append({"box": boxes[j].tolist(), "class": predicted_class, "score": score})
                    all_scores_flat.append(score)

                all_predictions.append(preds)
                all_gts.append(batch["boxes"][i])

        map_metrics = compute_map(
            all_predictions, all_gts, self.num_classes,
            iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        )

        scores_arr = np.array(all_scores_flat) if all_scores_flat else np.zeros((0,))
        map_metrics["prediction_stats"] = {
            "n_raw_predictions": n_raw_preds,
            "n_gt_boxes": n_gt_boxes_total,
            "min_confidence": float(np.min(scores_arr)) if len(scores_arr) else 0.0,
            "max_confidence": float(np.max(scores_arr)) if len(scores_arr) else 0.0,
            "mean_confidence": float(np.mean(scores_arr)) if len(scores_arr) else 0.0,
            "median_confidence": float(np.median(scores_arr)) if len(scores_arr) else 0.0,
        }
        return map_metrics

    # ------------------------------------------------------------------
    def fit(self, resume: bool = True, resume_checkpoint: Optional[str] = None):
        seed_everything(self.config.seed, self.config.cudnn_deterministic)
        save_training_config_snapshot(self.config.checkpoint_dir, self.config)

        ckpt_to_load = resume_checkpoint if (resume_checkpoint and os.path.exists(resume_checkpoint)) else None
        if ckpt_to_load is None and resume:
            ckpt_to_load = find_latest_checkpoint(self.config.checkpoint_dir)

        if ckpt_to_load is not None and os.path.exists(ckpt_to_load):
            print(f"[M10] Loading checkpoint: {ckpt_to_load}")
            self.state = load_checkpoint(
                ckpt_to_load, self.embedding_module, self.edps_module, self.encoder, self.detection_head,
                self.optimizer, self.scheduler, self.scaler, restore_rng=True,
            )
            self.state.epoch += 1  # resume from the epoch AFTER the saved one
            print(f"[M10] Resumed training state: epoch={self.state.epoch}/{self.config.total_epochs}, global_step={self.state.global_step}")

        while self.state.epoch < self.config.total_epochs:
            t0 = time.time()
            train_result = self.train_one_epoch()

            val_result = None
            if self.state.epoch % self.config.validate_every_n_epochs == 0:
                val_result = self.validate()

            summary = EpochSummary(
                epoch=self.state.epoch,
                train_loss=train_result["avg_loss"],
                train_loss_components=train_result["avg_components"],
                val_map50=val_result["map_50"] if val_result else None,
                val_map50_95=val_result["map_50_95"] if val_result else None,
                avg_retained_tokens=train_result["avg_retained_tokens"],
                sparsity_band={"min_ratio": train_result["band"].min_ratio,
                               "max_ratio": train_result["band"].max_ratio, "stage": train_result["band"].stage},
                n_unmatched_gt_fraction=train_result["unmatched_gt_fraction"],
                lr=self.optimizer.param_groups[-1]["lr"],
                epoch_time_s=time.time() - t0,
            )
            self.history.append(summary)

            val_map_str = f"{summary.val_map50:.4f}" if summary.val_map50 is not None else "N/A"
            val_map95_str = f"{summary.val_map50_95:.4f}" if summary.val_map50_95 is not None else "N/A"
            print("\n" + "=" * 70)
            print(f"  EPOCH {self.state.epoch + 1}/{self.config.total_epochs} SUMMARY ({self.config.checkpoint_dir})")
            print(f"  Train Loss : {summary.train_loss:.4f} (cls={summary.train_loss_components.get('cls', 0):.4f}, giou={summary.train_loss_components.get('giou', 0):.4f}, obj={summary.train_loss_components.get('objectness', 0):.4f})")
            print(f"  Val mAP@50 : {val_map_str} | Val mAP@50:95: {val_map95_str}")
            print(f"  Retained Tokens: {summary.avg_retained_tokens:.1f} tokens/sample")
            print(f"  Learning Rate  : {summary.lr:.6f} | Epoch Time: {summary.epoch_time_s:.2f}s")
            print("=" * 70 + "\n", flush=True)

            improved = False
            if val_result is not None and val_result["map_50"] is not None:
                if val_result["map_50"] > self.state.best_metric + self.config.early_stopping_min_delta:
                    self.state.best_metric = val_result["map_50"]
                    self.state.best_epoch = self.state.epoch
                    improved = True
                    self._epochs_without_improvement = 0
                    save_checkpoint(
                        best_checkpoint_path(self.config.checkpoint_dir),
                        self.embedding_module, self.edps_module, self.encoder, self.detection_head,
                        self.optimizer, self.scheduler, self.scaler, self.state, self.config,
                    )
                else:
                    self._epochs_without_improvement += 1

            if self.state.epoch % self.config.checkpoint_every_n_epochs == 0:
                save_checkpoint(
                    periodic_checkpoint_path(self.config.checkpoint_dir, self.state.epoch),
                    self.embedding_module, self.edps_module, self.encoder, self.detection_head,
                    self.optimizer, self.scheduler, self.scaler, self.state, self.config,
                )

            if self._epochs_without_improvement >= self.config.early_stopping_patience:
                break

            self.state.epoch += 1

        return self.history


if __name__ == "__main__":
    import argparse
    from dataset_split import create_split, resolve_dataset_root
    from gen1_dataset import EDPSGen1Dataset
    from patch_embedding_config import PatchEmbeddingConfig
    from patch_embedding_module import PatchEmbeddingModule
    from edps_config import EDPSConfig
    from edps_module import EDPSModule
    from transformer_config import TransformerEncoderConfig
    from transformer_encoder import TransformerEncoder
    from detection_head_config import DetectionHeadConfig
    from sparse_detection_head import SparseDetectionHead
    from training_collate import build_training_dataloader

    parser = argparse.ArgumentParser(description="M10 End-to-End Detector Trainer")
    parser.add_argument("--dataset_root", type=str, default="/content/gen1_local", help="Path to Gen1 dataset root")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--selection_mode", type=str, default="normal", choices=["normal", "baseline_no_pruning"])
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--total_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    actual_root = resolve_dataset_root(args.dataset_root)
    manifest = create_split(actual_root)

    num_bins = 10
    height, width = 240, 304
    patch_size = 16
    embedding_dim = 256
    n_rows, n_cols = height // patch_size, width // patch_size

    train_ds = EDPSGen1Dataset(actual_root, manifest, split="train", drop_empty_windows=True)
    val_ds = EDPSGen1Dataset(actual_root, manifest, split="val", drop_empty_windows=True) if len(manifest.val) > 0 else train_ds

    config = TrainingConfig(
        checkpoint_dir=args.checkpoint_dir,
        total_epochs=args.total_epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
    )

    embed_cfg = PatchEmbeddingConfig(embedding_dim=embedding_dim)
    embedding_module = PatchEmbeddingModule(embed_cfg, in_channels=num_bins, patch_size=patch_size, n_rows=n_rows, n_cols=n_cols, polarity_encoding="signed")

    edps_cfg = EDPSConfig(gate_threshold=args.gate_threshold, selection_mode=args.selection_mode)
    edps_module = EDPSModule(edps_cfg, embedding_dim=embedding_dim, num_bins=num_bins)

    encoder_cfg = TransformerEncoderConfig(embedding_dim=embedding_dim, num_heads=4, num_layers=2)
    encoder = TransformerEncoder(encoder_cfg)

    head_cfg = DetectionHeadConfig(embedding_dim=embedding_dim, num_classes=2)
    detection_head = SparseDetectionHead(head_cfg)

    target_dev = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"

    train_loader = build_training_dataloader(
        train_ds, actual_root, n_rows, n_cols, embedding_module, edps_module, encoder, detection_head, config, is_train=True, device=target_dev
    )
    val_loader = build_training_dataloader(
        val_ds, actual_root, n_rows, n_cols, embedding_module, edps_module, encoder, detection_head, config, is_train=False, device=target_dev
    )

    trainer = Trainer(
        embedding_module=embedding_module, edps_module=edps_module, encoder=encoder, detection_head=detection_head,
        train_loader=train_loader, val_loader=val_loader, config=config, num_classes=2, device=target_dev,
    )

    print(f"\n[Trainer] Starting training ({args.selection_mode}): epochs={args.total_epochs}, batch_size={args.batch_size}, device={target_dev}")
    history = trainer.fit(resume=not args.no_resume)
    print(f"[Trainer] Training completed! History recorded {len(history)} epochs.")
