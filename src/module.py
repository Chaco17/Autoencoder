"""LightningModule shared by both autoencoders (the network and loss are injected by Hydra)."""
import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional.classification import binary_auroc


def kl_divergence(mu, logvar):
    """KL(q(z|x) || N(0, I)), summed over latent dims and averaged over the batch."""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class AutoencoderModule(L.LightningModule):
    def __init__(self, net, loss_fn, optimizer, beta=0.0, scheduler=None):
        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss_fn", "optimizer", "scheduler"])
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer  # functools.partial from Hydra (_partial_: true)
        self.scheduler_factory = scheduler  # optional lr scheduler, also partial (receives the optimizer)
        self.beta = beta
        self.test_scores = []

    def forward(self, x):
        return self.net(x)

    def shared_step(self, batch, stage):
        x = batch["image"]
        out = self(x)
        recon = out["recon"]

        recon_loss = self.loss_fn(recon, x)
        loss = recon_loss
        logs = {f"{stage}/recon_loss": recon_loss}
        if out["mu"] is not None:  # VAE only
            kl = kl_divergence(out["mu"], out["logvar"])
            loss = loss + self.beta * kl
            logs[f"{stage}/kl"] = kl
        logs[f"{stage}/loss"] = loss

        # Same metrics for every run, so runs trained with different losses can be compared on one scale
        with torch.no_grad():
            logs[f"{stage}/l1"] = F.l1_loss(recon, x)
            logs[f"{stage}/mse"] = F.mse_loss(recon, x)
            logs[f"{stage}/ssim"] = ssim(recon, x, data_range=1.0)

        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False, batch_size=x.size(0))
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=x.size(0))
        return loss, recon

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        _, recon = self.shared_step(batch, "test")
        x = batch["image"]
        # Per-image reconstruction error = anomaly score (used for histograms and AUROC)
        err = (recon - x).flatten(1)
        # score_max: peak of the smoothed per-pixel error map. The image mean dilutes small defects
        # (a scratch covers <1% of the pixels); the local maximum keeps them visible.
        err_map = F.avg_pool2d((recon - x).abs().mean(1, keepdim=True), 11, stride=1, padding=5, count_include_pad=False)
        scores = {
            "score_l1": err.abs().mean(1),
            "score_mse": err.pow(2).mean(1),
            "score_ssim": 1.0 - ssim(recon, x, data_range=1.0, reduction="none"),
            "score_max": err_map.flatten(1).amax(1),
        }
        for i in range(x.size(0)):
            self.test_scores.append(
                {
                    "category": batch["category"][i],
                    "defect": batch["defect"][i],
                    "label": int(batch["label"][i]),
                    **{name: float(values[i]) for name, values in scores.items()},
                }
            )

    def on_test_epoch_end(self):
        rows = self.test_scores
        labels = torch.tensor([r["label"] for r in rows])
        score_names = ["score_l1", "score_mse", "score_ssim", "score_max"]

        # AUROC: how well the reconstruction error separates good from defective images (0.5 = random).
        # The main figure is the mean of the per-class AUROCs (as in MVTec AD): pooling all classes mixes
        # objects whose errors live on different scales, which lowers the global value.
        for name in score_names:
            short = name.removeprefix("score_")
            self.log(f"test/auroc_{short}", binary_auroc(torch.tensor([r[name] for r in rows]), labels))
            per_class = []
            for category in sorted({r["category"] for r in rows}):
                cat_rows = [r for r in rows if r["category"] == category]
                cat_scores = torch.tensor([r[name] for r in cat_rows])
                cat_labels = torch.tensor([r["label"] for r in cat_rows])
                auroc = binary_auroc(cat_scores, cat_labels)
                per_class.append(auroc)
                self.log(f"test/auroc_{short}/{category}", auroc)
            self.log(f"test/auroc_{short}_mean", torch.stack(per_class).mean())

        # Raw scores as a W&B table: the notebook builds the per-defect histograms from it
        if isinstance(self.logger, WandbLogger):
            columns = ["category", "defect", "label"] + score_names
            table = wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in rows])
            self.logger.experiment.log({"test/scores": table})
        self.test_scores = []

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(params=self.parameters())
        if self.scheduler_factory is None:  # constant learning rate
            return optimizer
        scheduler = self.scheduler_factory(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
