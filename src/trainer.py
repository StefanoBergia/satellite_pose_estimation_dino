from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys

from tqdm import tqdm

from .losses import visibility_weighted_mse, combined_pose_loss, heatmap_mse_loss
from .utils import (
    compute_pixel_error,
    compute_pck,
    compute_rotation_error,
    compute_translation_error,
    compute_slab_score,
)


class Trainer:
    """Training loop for keypoint + pose estimation model.

    Supports two modes:
        - keypoint_only: only keypoint MSE loss
        - keypoint_pnp: keypoint + differentiable PnP pose loss (with warmup)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        eval_loaders: dict[str, DataLoader],
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loaders = eval_loaders
        self.config = config
        self.device = device

        self.mode = config["model"]["mode"]
        self.epochs = config["train"]["epochs"]
        self.output_dir = Path(config["train"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = config["train"].get("log_interval", 50)
        self.occluded_weight = config["train"].get("occluded_weight", 0.5)
        self.pck_threshold = config["eval"].get("pck_threshold", 0.05)

        # Loss weights
        pose_cfg = config.get("pose", {})
        self.lambda_kp = pose_cfg.get("lambda_keypoint", 1.0)
        self.lambda_pnp = pose_cfg.get("lambda_pnp_pose", 0.5)
        self.rot_weight = pose_cfg.get("rotation_weight", 1.0)
        self.trans_weight = pose_cfg.get("translation_weight", 1.0)

        # Heatmap loss
        self.lambda_heatmap = pose_cfg.get("lambda_heatmap", 0.0)
        self.heatmap_size = pose_cfg.get("heatmap_size", 64)
        self.heatmap_sigma = pose_cfg.get("heatmap_sigma", 1.5)

        # PnP warmup: start PnP loss after this many epochs
        self.pnp_warmup_epochs = pose_cfg.get("pnp_warmup_epochs", 10)
        self.pnp_rampup_epochs = pose_cfg.get("pnp_rampup_epochs", 10)

        # Optimizer with separate param groups
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = list(model.keypoint_head.parameters())

        param_groups = [
            {"params": head_params, "lr": config["train"]["lr"]},
        ]
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": config["train"]["lr_backbone"],
            })

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config["train"]["weight_decay"],
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
        )

        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        self.best_val_loss = float("inf")

    def _get_pnp_weight(self, epoch: int) -> float:
        """Compute PnP loss weight with warmup + linear ramp-up."""
        if self.mode != "keypoint_pnp":
            return 0.0
        if epoch <= self.pnp_warmup_epochs:
            return 0.0
        ramp = min(1.0, (epoch - self.pnp_warmup_epochs) / max(self.pnp_rampup_epochs, 1))
        return self.lambda_pnp * ramp

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_losses = self._train_epoch(epoch)
            self.scheduler.step()

            # Log training losses
            for name, val in train_losses.items():
                self.writer.add_scalar(f"train/{name}", val, epoch)
            self.writer.add_scalar("train/lr_head", self.optimizer.param_groups[0]["lr"], epoch)
            if self.mode == "keypoint_pnp":
                self.writer.add_scalar("train/pnp_weight", self._get_pnp_weight(epoch), epoch)

            # Evaluate on all splits
            all_metrics = {}
            for split_name, loader in self.eval_loaders.items():
                metrics = self._evaluate(loader)
                all_metrics[split_name] = metrics
                for name, val in metrics.items():
                    self.writer.add_scalar(f"{split_name}/{name}", val, epoch)

                msg = (
                    f"  [{split_name}] loss={metrics['loss']:.5f}  "
                    f"px_err={metrics['pixel_error']:.2f}  "
                    f"pck={metrics['pck']:.4f}"
                )
                if "pnp_rot_error_deg" in metrics:
                    msg += f"  rot={metrics['pnp_rot_error_deg']:.2f}deg  t_err={metrics['pnp_trans_error']:.4f}"
                if "pnp_slab_score" in metrics:
                    msg += f"  SLAB={metrics['pnp_slab_score']:.4f}"
                print(msg)

            # Save best model (reuse metrics already computed above)
            primary_split = "val" if "val" in all_metrics else next(iter(all_metrics))
            val_metrics = all_metrics[primary_split]
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, is_best=True)
                print(f"  * New best model (val_loss={self.best_val_loss:.5f})")

            if epoch % 10 == 0:
                self._save_checkpoint(epoch)

        self.writer.close()
        print("Training complete.")

    def _compute_loss(
        self, model_out: dict, batch: dict, epoch: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss depending on mode. Returns (loss, loss_dict)."""
        gt_kp = batch["keypoints"].to(self.device)
        vis = batch["visibility"].to(self.device)

        log = {}

        # Keypoint loss (always)
        kp_loss = visibility_weighted_mse(
            model_out["keypoints"], gt_kp, vis, self.occluded_weight
        )
        total = self.lambda_kp * kp_loss
        log["kp_loss"] = kp_loss.item()

        # Heatmap loss (when heatmap head is used)
        if "heatmaps" in model_out and self.lambda_heatmap > 0:
            hm_loss = heatmap_mse_loss(
                model_out["heatmaps"], gt_kp, vis,
                self.heatmap_size, self.heatmap_sigma, self.occluded_weight,
            )
            total = total + self.lambda_heatmap * hm_loss
            log["heatmap_loss"] = hm_loss.item()

        # PnP pose loss (with warmup)
        if self.mode == "keypoint_pnp" and "pnp_rotation" in model_out:
            gt_q = batch["quaternion"].to(self.device)
            gt_t = batch["translation"].to(self.device)
            has_pose = batch["has_pose"].to(self.device)

            pnp_weight = self._get_pnp_weight(epoch)
            if pnp_weight > 0:
                pnp_losses = combined_pose_loss(
                    model_out["pnp_rotation"],
                    model_out["pnp_translation"],
                    gt_q, gt_t, has_pose,
                    self.rot_weight, self.trans_weight,
                )
                total = total + pnp_weight * pnp_losses["pose_loss"]
                log["pnp_rot_loss"] = pnp_losses["rotation_loss"].item()
                log["pnp_trans_loss"] = pnp_losses["translation_loss"].item()

        log["loss"] = total.item()
        return total, log

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        accum = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}", file=sys.stdout)
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)

            # Build model forward kwargs
            fwd_kwargs = {"pixel_values": images}
            if self.mode == "keypoint_pnp":
                fwd_kwargs["crop_box"] = batch["crop_box"].to(self.device)
                fwd_kwargs["img_size"] = batch["img_size"].to(self.device)
                fwd_kwargs["visibility"] = batch["visibility"].to(self.device)

            model_out = self.model(**fwd_kwargs)
            loss, log = self._compute_loss(model_out, batch, epoch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for k, v in log.items():
                accum[k] = accum.get(k, 0.0) + v
            num_batches += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(loss=f"{log['loss']:.5f}")

        return {k: v / max(num_batches, 1) for k, v in accum.items()}

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        accum = {}
        num_batches = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            gt_kp = batch["keypoints"].to(self.device)
            vis = batch["visibility"].to(self.device)
            crop_box = batch["crop_box"].to(self.device)

            fwd_kwargs = {"pixel_values": images}
            if self.mode == "keypoint_pnp":
                fwd_kwargs["crop_box"] = crop_box
                fwd_kwargs["img_size"] = batch["img_size"].to(self.device)
                fwd_kwargs["visibility"] = vis

            model_out = self.model(**fwd_kwargs)

            # Keypoint metrics
            kp_loss = visibility_weighted_mse(
                model_out["keypoints"], gt_kp, vis, self.occluded_weight
            )
            px_err = compute_pixel_error(model_out["keypoints"], gt_kp, vis, crop_box)
            pck = compute_pck(model_out["keypoints"], gt_kp, vis, crop_box, self.pck_threshold)

            accum["loss"] = accum.get("loss", 0.0) + kp_loss.item()
            accum["pixel_error"] = accum.get("pixel_error", 0.0) + px_err.item()
            accum["pck"] = accum.get("pck", 0.0) + pck.item()

            # PnP pose metrics (keypoint_pnp mode)
            has_pose = batch["has_pose"].to(self.device)
            if "pnp_rotation" in model_out and has_pose.any():
                gt_q = batch["quaternion"].to(self.device)
                gt_t = batch["translation"].to(self.device)

                pnp_rot_err = compute_rotation_error(
                    model_out["pnp_rotation"], gt_q, has_pose
                )
                pnp_trans_err = compute_translation_error(
                    model_out["pnp_translation"], gt_t, has_pose
                )
                accum["pnp_rot_error_deg"] = accum.get("pnp_rot_error_deg", 0.0) + pnp_rot_err.item()
                accum["pnp_trans_error"] = accum.get("pnp_trans_error", 0.0) + pnp_trans_err.item()

                pnp_slab = compute_slab_score(
                    model_out["pnp_rotation"],
                    model_out["pnp_translation"],
                    gt_q, gt_t, has_pose,
                )
                accum["pnp_slab_score"] = accum.get("pnp_slab_score", 0.0) + pnp_slab["slab_score"].item()

            num_batches += 1

        n = max(num_batches, 1)
        return {k: v / n for k, v in accum.items()}

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if is_best:
            path = self.output_dir / "best_model.pth"
        else:
            path = self.output_dir / f"checkpoint_epoch{epoch:03d}.pth"

        torch.save(state, path)
