"""Callbacks that log images and latent-space plots to W&B (no matplotlib).

- ReconstructionCallback: originals vs reconstructions of 16 fixed validation images.
- LatentTSNECallback: t-SNE of z for the validation set over the epochs, and for the test set at the end.
- TestReconstructionCallback: good vs defective test images, with their reconstructions and error maps.
"""
import time
from collections import defaultdict

import plotly.graph_objects as go
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from sklearn.manifold import TSNE
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

CATEGORY_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "cross"]
LABEL_COLORS = {0: "#2ca02c", 1: "#d62728"}  # good = green, defective = red
LABEL_NAMES = {0: "good", 1: "defective"}


def spread_indices(candidates, groups, n):
    """Pick n indices from candidates, spread evenly over the groups and inside each group."""
    by_group = defaultdict(list)
    for i in candidates:
        by_group[groups[i]].append(i)
    per_group = max(1, n // len(by_group))
    picked = []
    for items in by_group.values():
        step = max(1, len(items) // per_group)
        picked += items[::step][:per_group]
    return picked[:n]


@torch.no_grad()
def run_model(pl_module, dataset, indices, batch_size=64):
    """Return (images, reconstructions, z) on CPU for the given dataset indices."""
    xs, recons, zs = [], [], []
    for start in range(0, len(indices), batch_size):
        x = dataset.get_images(indices[start : start + batch_size]).to(pl_module.device)
        out = pl_module(x)
        xs.append(x.cpu())
        recons.append(out["recon"].cpu())
        zs.append(out["z"].cpu())
    return torch.cat(xs), torch.cat(recons), torch.cat(zs)


def should_log(trainer, every_n_epochs):
    epoch = trainer.current_epoch
    return epoch == 0 or (epoch + 1) % every_n_epochs == 0 or epoch + 1 == trainer.max_epochs


def tsne_figure(z, groups, colors, symbols, title, seed=42):
    """2D t-SNE of z as a plotly scatter (one trace per group)."""
    perplexity = min(30.0, (len(z) - 1) / 3)
    points = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(z.numpy())
    fig = go.Figure()
    for group in dict.fromkeys(groups):  # keeps first-seen order
        idx = [i for i, g in enumerate(groups) if g == group]
        fig.add_trace(
            go.Scatter(
                x=points[idx, 0],
                y=points[idx, 1],
                mode="markers",
                name=group,
                marker=dict(size=7, opacity=0.75, color=colors[group], symbol=symbols[group]),
            )
        )
    fig.update_layout(title=title, xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", template="plotly_white")
    return fig


class ReconstructionCallback(Callback):
    def __init__(self, num_images=16, every_n_epochs=5, nrow=8):
        self.num_images = num_images
        self.every_n_epochs = every_n_epochs
        self.nrow = nrow
        self.indices = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not should_log(trainer, self.every_n_epochs):
            return
        dataset = trainer.datamodule.val_dataset
        if self.indices is None:  # same images every epoch, 4 per category
            self.indices = spread_indices(range(len(dataset)), dataset.categories, self.num_images)
        x, recon, _ = run_model(pl_module, dataset, self.indices)

        # Rows alternate: originals, reconstructions, originals, reconstructions...
        tiles = []
        for start in range(0, len(x), self.nrow):
            tiles += [x[start : start + self.nrow], recon[start : start + self.nrow]]
        grid = make_grid(torch.cat(tiles), nrow=self.nrow, padding=2, pad_value=1.0)
        trainer.logger.log_image(
            key="val/reconstructions",
            images=[to_pil_image(grid.clamp(0, 1))],  # PIL: wandb no longer scales [0, 1] floats
            caption=[f"Epoch {trainer.current_epoch}: odd rows = original, even rows = reconstruction"],
        )


class LatentTSNECallback(Callback):
    def __init__(self, every_n_epochs=10, seed=42):
        self.every_n_epochs = every_n_epochs
        self.seed = seed

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not should_log(trainer, self.every_n_epochs):
            return
        dataset = trainer.datamodule.val_dataset
        _, _, z = run_model(pl_module, dataset, list(range(len(dataset))))

        categories = sorted(set(dataset.categories))
        palette = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
        colors = {c: palette[i % len(palette)] for i, c in enumerate(categories)}
        symbols = {c: "circle" for c in categories}
        fig = tsne_figure(
            z, dataset.categories, colors, symbols, f"Latent space (t-SNE) - validation - epoch {trainer.current_epoch}", self.seed
        )
        trainer.logger.experiment.log({"val/tsne": wandb.Plotly(fig), "trainer/global_step": trainer.global_step})

    def on_test_epoch_end(self, trainer, pl_module):
        dataset = trainer.datamodule.test_dataset
        _, _, z = run_model(pl_module, dataset, list(range(len(dataset))))

        categories = sorted(set(dataset.categories))
        groups = [f"{c} - {LABEL_NAMES[l]}" for c, l in zip(dataset.categories, dataset.labels)]
        colors, symbols = {}, {}
        for i, c in enumerate(categories):
            for label, name in LABEL_NAMES.items():
                colors[f"{c} - {name}"] = LABEL_COLORS[label]
                symbols[f"{c} - {name}"] = CATEGORY_SYMBOLS[i % len(CATEGORY_SYMBOLS)]
        fig = tsne_figure(z, groups, colors, symbols, "Latent space (t-SNE) - test - good vs defective", self.seed)
        trainer.logger.experiment.log({"test/tsne": wandb.Plotly(fig)})


class TestReconstructionCallback(Callback):
    def __init__(self, num_images=16):
        self.num_images = num_images

    def on_test_epoch_end(self, trainer, pl_module):
        dataset = trainer.datamodule.test_dataset
        half = self.num_images // 2
        good = [i for i, label in enumerate(dataset.labels) if label == 0]
        defective = [i for i, label in enumerate(dataset.labels) if label == 1]
        indices = spread_indices(good, dataset.categories, half) + spread_indices(defective, dataset.categories, half)
        x, recon, _ = run_model(pl_module, dataset, indices)

        # Error map |x - recon|, normalized with one shared max so good and defective are comparable
        err = (x - recon).abs().mean(1, keepdim=True)
        err = (err / err.max().clamp_min(1e-8)).expand_as(x)

        g, d = slice(0, half), slice(half, 2 * half)
        tiles = torch.cat([x[g], recon[g], err[g], x[d], recon[d], err[d]])
        grid = make_grid(tiles, nrow=half, padding=2, pad_value=1.0)
        captions = [dataset.defects[i] for i in indices[half:]]
        trainer.logger.log_image(
            key="test/reconstructions",
            images=[to_pil_image(grid.clamp(0, 1))],  # PIL: wandb no longer scales [0, 1] floats
            caption=[
                "Rows 1-3: good (original, reconstruction, error). "
                f"Rows 4-6: defective (original, reconstruction, error). Defects: {', '.join(captions)}"
            ],
        )


class EpochSummaryCallback(Callback):
    """One compact line per epoch when the tqdm bar is disabled (readable output inside notebook cells).

    Prints the first and last epoch and every `every_n_epochs`; * marks a new best val/loss.
    Also keeps the best epoch, which train.py reports at the end of the run.
    """

    def __init__(self, run_name="", every_n_epochs=5):
        self.run_name = run_name
        self.every_n_epochs = every_n_epochs
        self.best_epoch, self.best_val_loss = None, float("inf")
        self.epoch_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._tick = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        # Runs after the epoch's validation, so train and val metrics are from the same epoch
        self.epoch_times.append(time.perf_counter() - self._tick)
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch + 1
        val_loss = float(metrics.get("val/loss", float("inf")))
        improved = val_loss < self.best_val_loss
        if improved:
            self.best_epoch, self.best_val_loss = epoch, val_loss

        if trainer.progress_bar_callback is not None:
            return
        if not (epoch == 1 or epoch == trainer.max_epochs or epoch % self.every_n_epochs == 0):
            return
        mean_time = sum(self.epoch_times) / len(self.epoch_times)
        eta = mean_time * (trainer.max_epochs - epoch)
        parts = [
            f"{self.run_name:<14} epoch {epoch:>3}/{trainer.max_epochs}",
            f"train {float(metrics.get('train/loss', float('nan'))):.4f}",
            f"val {val_loss:.4f}",
            f"val_ssim {float(metrics.get('val/ssim', float('nan'))):.3f}",
            f"{self.epoch_times[-1]:4.1f}s/época",
            f"ETA {int(eta // 60)}m{int(eta % 60):02d}s",
        ]
        print(" | ".join(parts) + (" *" if improved else ""), flush=True)
