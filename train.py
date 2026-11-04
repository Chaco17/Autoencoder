"""Training entry point (Hydra + PyTorch Lightning).

Usage:
    python train.py                         # defaults from conf/config.yaml
    python train.py model=u-net loss=ssim   # pick other config options
    python train.py trial=e200_cos          # another set of hyperparameters (conf/trial/)
    python train.py -m                      # sweep trials x 2 models x 4 losses (runs already done are skipped)
    python train.py skip_existing.enabled=false   # train even if the same experiment was already done
    python train.py console.verbose=false   # compact output (what the notebook uses)
"""
import json
import logging
import os
import time
import warnings
from pathlib import Path

# Hide torch's import-time "triton not found" message (Windows has no triton; it only affects flop counting)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import hydra
import lightning as L
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.callbacks import (
    EpochSummaryCallback,
    LatentTSNECallback,
    ReconstructionCallback,
    TestReconstructionCallback,
)
from src.data import MVTecDataModule
from src.module import AutoencoderModule
from src.registry import find_existing, run_key

ROOT = Path(__file__).resolve().parent


def quiet_mode():
    """Silence library banners/warnings so the console only shows our own summary lines."""
    os.environ["WANDB_SILENT"] = "true"
    warnings.filterwarnings("ignore")
    for name in ("lightning", "lightning.pytorch", "lightning.fabric", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.WARNING)


def format_seconds(seconds):
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes}m{seconds:02d}s"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    verbose = cfg.console.verbose
    if verbose:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    else:
        quiet_mode()
    start = time.perf_counter()

    # Skip experiments that were already done (same hyperparameters), so a sweep can be relaunched
    # without duplicating runs in W&B. Checked before the W&B run is created.
    key = run_key(cfg)
    if cfg.skip_existing.enabled:
        try:
            existing = find_existing(cfg, ROOT)
        except Exception as error:  # e.g. no connection to W&B: only the local check can be done
            print(f"   (no se pudo consultar W&B para {cfg.run_name}: {error}; solo se revisa outputs/ y multirun/)", flush=True)
            existing = find_existing(cfg, ROOT, remote={})
        if existing:
            print(f"\n>> SALTADA {cfg.run_name} [{key}]: ya existe -> {existing}", flush=True)
            return

    torch.set_float32_matmul_precision("high")  # TF32 on Tensor Core GPUs: faster, negligible precision change
    L.seed_everything(cfg.seed, workers=True, verbose=verbose)
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Data: MVTec AD (cable, capsule, screw, transistor) at 128x128x3
    datamodule = MVTecDataModule(**cfg.data)

    # Model: network, loss and optimizer are built from their _target_ in the YAML files
    model = AutoencoderModule(
        net=instantiate(cfg.model.net),
        loss_fn=instantiate(cfg.loss.fn),
        optimizer=instantiate(cfg.model.optimizer),  # partial: receives params later
        beta=cfg.model.beta,
        scheduler=instantiate(cfg.model.scheduler) if cfg.model.get("scheduler") else None,  # partial, optional
    )

    # Logger: W&B run named <model>-<loss>, with the full resolved config attached
    logger = instantiate(cfg.logger.wandb)
    logger._wandb_init.pop("anonymous", None)  # Lightning always sends it; wandb >= 0.31 warns that it is deprecated
    logger.log_hyperparams({**OmegaConf.to_container(cfg, resolve=True), "run_key": key})  # run_key: src/registry.py

    print(
        f"\n>> {cfg.run_name} [{key}] | {cfg.trial.description} | "
        f"z {cfg.model.net.latent_dim} | W&B {cfg.logger.wandb.mode}",
        flush=True,
    )
    if cfg.logger.wandb.mode == "online":
        print(f"   {logger.experiment.url}", flush=True)

    viz = cfg.logger.viz
    summary = EpochSummaryCallback(run_name=cfg.run_name, every_n_epochs=cfg.console.print_every_n_epochs)
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best",
            monitor="val/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ReconstructionCallback(num_images=viz.num_images, every_n_epochs=viz.images_every_n_epochs),
        LatentTSNECallback(every_n_epochs=viz.tsne_every_n_epochs),
        TestReconstructionCallback(num_images=viz.num_images),
        summary,
    ]

    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks, enable_model_summary=verbose)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best", verbose=verbose)

    # Compact result of the run: printed and saved as results.json (the notebook collects these files)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("test/")}
    results = {
        "run": cfg.run_name,
        "model": cfg.model.name,
        "loss": cfg.loss.name,
        "trial": cfg.trial.name,
        "run_key": key,
        "epochs": trainer.current_epoch,
        "best_epoch": summary.best_epoch,
        "best_val_loss": summary.best_val_loss,
        "best_train_loss": summary.best_train_loss,  # train/loss of the best epoch
        "minutes": round((time.perf_counter() - start) / 60, 2),
        "wandb_url": logger.experiment.url if cfg.logger.wandb.mode == "online" else None,
        "output_dir": str(output_dir),
        **metrics,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    per_class = " ".join(
        f"{c} {metrics[f'test/auroc_max/{c}']:.2f}" for c in cfg.data.classes if f"test/auroc_max/{c}" in metrics
    )
    print(
        f"OK {cfg.run_name} terminado en {format_seconds(time.perf_counter() - start)} | "
        f"mejor época {summary.best_epoch} (val/loss {summary.best_val_loss:.4f})\n"
        f"   test: L1 {metrics.get('test/l1', float('nan')):.4f} | SSIM {metrics.get('test/ssim', float('nan')):.4f} | "
        f"AUROC media por clase: L1 {metrics.get('test/auroc_l1_mean', float('nan')):.3f} | "
        f"max {metrics.get('test/auroc_max_mean', float('nan')):.3f}\n"
        f"   AUROC(max) por clase: {per_class}",
        flush=True,
    )

    # Close the run so each multirun job gets its own W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
