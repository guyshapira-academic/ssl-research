"""
This script is used to train the model using the VICReg training loop.
"""
import os.path
import re

import hydra
import lightning as L
import omegaconf
import torch.utils.data as tdata
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig
from ssl_research import ncc
from ssl_research.models.cnn import resnet, vanilla
from ssl_research.vicreg import VICReg
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point for the training script."""

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    run_name = f"{cfg.model.type}_w{cfg.model.width_factor}_d{cfg.model.depth_factor}"
    run_name = re.sub(r"\.", "-", run_name)

    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        name=run_name,
        project="SSL Research",
        settings=wandb.Settings(start_method="thread"),
    )

    # Create the dataset
    if cfg.dataset.name == "stl10":
        training_dataset = STL10(
            root=".", split="unlabeled", download=True, transform=transforms.ToTensor()
        )
        validation_dataset = STL10(
            root=".",
            split="train",
            download=True,
            transform=transforms.ToTensor(),
        )
        image_size = cfg.dataset.get("image_size") or 96
    elif cfg.dataset.name == "cifar10":
        training_dataset = CIFAR10(
            root=".", train=True, download=True, transform=transforms.ToTensor()
        )
        training_dataset, validation_dataset = tdata.random_split(
            training_dataset, [45000, 5000]
        )
        image_size = cfg.dataset.get("image_size") or 32
    elif cfg.dataset.name == "cifar100":
        training_dataset = CIFAR100(
            root=".", train=True, download=True, transform=transforms.ToTensor()
        )
        training_dataset, validation_dataset = tdata.random_split(
            training_dataset, [45000, 5000]
        )
        image_size = cfg.dataset.get("image_size") or 32
    else:
        raise ValueError("Unknown dataset")

    train_loader = DataLoader(
        training_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    validation_ncc_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # Create the model
    if image_size < 64:
        stem_stride = 1
    else:
        stem_stride = 2
    if cfg.model.type == "vanilla":
        model = vanilla(
            cfg.model.dim,
            stem_stride=stem_stride,
            width_factor=cfg.model.width_factor,
            depth_factor=cfg.model.depth_factor,
        )
    elif cfg.model.type == "resnet":
        model = resnet(
            cfg.model.dim,
            stem_stride=stem_stride,
            width_factor=cfg.model.width_factor,
            depth_factor=cfg.model.depth_factor,
        )
    else:
        raise ValueError("Unknown model type")

    # Create the VICReg model
    vicreg = VICReg(
        model,
        projector_features=cfg.training.projector_features,
        sim_coef=cfg.training.sim_coef,
        cov_coef=cfg.training.cov_coef,
        std_coef=cfg.training.std_coef,
        optimizer_type=cfg.training.optimizer_type,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        num_epochs=cfg.training.num_epochs,
        image_size=image_size,
    )

    # Create the logger

    wandb_logger = WandbLogger(project="SSL Research", log_model="all")
    # add your batch size to the wandb config
    wandb_logger.experiment.config["batch_size"] = cfg.training.batch_size

    csv_logger = CSVLogger(save_dir=hydra_cfg["runtime"]["output_dir"], name="vicreg")

    # Create the trainer
    trainer = L.Trainer(
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=cfg.training.num_epochs,
        callbacks=[
            ncc.SSLMetricsCallback(validation_ncc_loader),
            ModelCheckpoint(
                every_n_epochs=100,
                dirpath=os.path.join(hydra_cfg["runtime"]["output_dir"], "checkpoints"),
                filename="vicreg-{epoch:02d}",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=[wandb_logger, csv_logger],
    )
    os.makedirs(
        os.path.join(hydra_cfg["runtime"]["output_dir"], "checkpoints"), exist_ok=True
    )
    trainer.save_checkpoint(
        os.path.join(
            hydra_cfg["runtime"]["output_dir"], "checkpoints", f"{run_name}-init.ckpt"
        )
    )

    # Train the model
    trainer.fit(vicreg, train_loader, val_dataloaders=validation_loader)

    # Save the model
    trainer.save_checkpoint(
        os.path.join(
            hydra_cfg["runtime"]["output_dir"], "checkpoints", f"{run_name}-final.ckpt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    main()
