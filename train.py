"""
This script is used to train the model using the VICReg training loop.
"""
import hydra
import lightning as L
import torch.utils.data as tdata
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from ssl_research import ncc
from ssl_research.models.cnn import resnet20, vanilla20
from ssl_research.vicreg import VICReg, VICRegDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, STL10


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point for the training script."""

    # Create the dataset
    if cfg.dataset == "stl10":
        training_dataset = STL10(root=".", split="unlabeled", download=True)
        training_dataset = VICRegDataset(training_dataset)
        validation_ncc_dataset = STL10(
            root=".",
            split="train",
            download=True,
            transform=transforms.ToTensor(),
        )
    elif cfg.dataset == "cifar10":
        training_dataset = CIFAR10(root=".", train=True, download=True)
        training_dataset = VICRegDataset(training_dataset)
        validation_ncc_dataset = CIFAR10(
            root=".",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    else:
        raise ValueError("Unknown dataset")

    # split the training_dataset into train and validation
    train_length = int(len(training_dataset) * 0.8)
    validation_length = len(training_dataset) - train_length
    training_dataset, validation_dataset = tdata.random_split(
        training_dataset, [train_length, validation_length]
    )
    train_loader = DataLoader(
        training_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    validation_ncc_loader = DataLoader(
        validation_ncc_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # Create the model
    if cfg.model.type == "vanilla20":
        model = vanilla20(cfg.model.dim)
    elif cfg.model.type == "resnet20":
        model = resnet20(cfg.model.dim)
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
    )

    # Create the logger
    wandb_logger = WandbLogger(project="SSL Research")

    # add your batch size to the wandb config
    wandb_logger.experiment.config["batch_size"] = cfg.training.batch_size

    # Create the trainer
    trainer = L.Trainer(
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=cfg.training.num_epochs,
        callbacks=[ncc.NCCAccuracyCallback(validation_ncc_loader)],
        logger=wandb_logger,
    )

    # Train the model
    trainer.fit(vicreg, train_loader, val_dataloaders=validation_loader)


if __name__ == "__main__":
    main()
