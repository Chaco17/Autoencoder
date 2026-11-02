"""Training entry point (Hydra + PyTorch Lightning).

Usage:
    python train.py                         # defaults from conf/config.yaml
    python train.py model=u-net loss=ssim   # pick other config options
    python train.py -m                      # sweep 2 models x 4 losses
"""
from pathlib import Path

import hydra
import lightning as L
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.callbacks import LatentTSNECallback, ReconstructionCallback, TestReconstructionCallback
from src.data import MVTecDataModule
from src.module import AutoencoderModule


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    L.seed_everything(cfg.seed, workers=True)
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Data: MVTec AD (cable, capsule, screw, transistor) at 128x128x3
    datamodule = MVTecDataModule(**cfg.data)

    # Model: network, loss and optimizer are built from their _target_ in the YAML files
    model = AutoencoderModule(
        net=instantiate(cfg.model.net),
        loss_fn=instantiate(cfg.loss.fn),
        optimizer=instantiate(cfg.model.optimizer),  # partial: receives params later
        beta=cfg.model.beta,
    )

    # Logger: W&B run named <model>-<loss>, with the full resolved config attached
    logger = instantiate(cfg.logger.wandb)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    viz = cfg.logger.viz
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
    ]

    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Close the run so each multirun job gets its own W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
