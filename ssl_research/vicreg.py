"""This module implements the VICReg algorithm for SSL.

Based on the paper:
    "VICReg: Variance-Invariance-Covariance Regularization for
    Self-Supervised Learning", https://arxiv.org/abs/2105.04906
"""
from typing import Callable, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from ssl_research.lars import LARS
from ssl_research.models.cnn import CNN
from torch import Tensor


class VICReg(L.LightningModule):
    """This class implements the VICReg algorithm for SSL, for training
    using Lightning.

    Parameters:
        model (nn.Module): The model to train.
        projector_features (int): Number of features in the projection head.
            Default: 128.
        sim_coef (float): Coefficient for the similarity loss. Default: 25.0.
        std_coef (float): Coefficient for the standard deviation loss.
            Default: 25.0.
        cov_coef (float): Coefficient for the covariance loss. Default: 1.0.
        lr (float): Learning rate. Default: 1e-3.
        weight_decay (float): Weight decay. Default: 1e-6.
        batch_size (int): Batch size. Default: 256.
        num_workers (int): Number of workers for the data loader. Default: 4.
        num_epochs (int): Number of epochs to train for. Default: 100.
    """

    def __init__(
        self,
        model: CNN,
        projector_features: int = 128,
        sim_coef: float = 25.0,
        std_coef: float = 25.0,
        cov_coef: float = 1.0,
        optimizer_type: str = "lars",
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 256,
        num_workers: int = 4,
        num_epochs: int = 100,
        image_size: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "image_size"])

        self.model = model
        self.projector = nn.Sequential(
            nn.Linear(model.output_dim, projector_features),
            nn.BatchNorm1d(projector_features),
            nn.ReLU(inplace=True),
            nn.Linear(projector_features, projector_features),
            nn.BatchNorm1d(projector_features),
            nn.ReLU(inplace=True),
            nn.Linear(projector_features, projector_features, bias=False),
        )
        self.projector_features = projector_features
        self.sim_coef = sim_coef
        self.std_coef = std_coef
        self.cov_coef = cov_coef
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs

        self.transform = vicreg_transform(image_size)

    @property
    def num_features(self) -> int:
        """The number of features in the backbone."""
        return self.model.num_features

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model. Backbone + projection head.

        Parameters:
            x (torch.Tensor): The input tensor.
        """
        y = self.model(x)
        y = self.projector(y)
        return y

    def configure_optimizers(self):
        """Configure the optimizer."""
        # Should be LARS optimizer, but SGD is used for simplicity
        # optimizer = LARS(self.parameters(), lr=self.lr,
        # weight_decay=self.weight_decay)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

        if self.optimizer_type == "lars":
            optimizer = LARS(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer_type == "adam":
            optimizer = optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Optimizer type {self.optimizer_type} not supported.")

        # Cosine decay
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )
        return [optimizer], [scheduler]

    def training_step(self, batch: Tuple[Tensor, ...], *args, **kwargs):
        """
        VICReg training step.

        Parameters:
            batch (Tensor): The batch of data.
        """
        x, _ = batch
        x_a = self.transform(x)
        x_b = self.transform(x)

        y_a = self.forward(x_a)
        y_b = self.forward(x_b)

        loss, repr_loss, std_loss, cov_loss = vicreg_loss(
            y_a, y_b, self.sim_coef, self.std_coef, self.cov_coef
        )

        # Log the three components of the loss
        self.log("train/repr_loss", repr_loss)
        self.log("train/std_loss", std_loss)
        self.log("train/cov_loss", cov_loss)

        # Log the total loss
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[Tensor, ...], *args, **kwargs):
        """
        VICReg validation step.

        Parameters:
            batch (Tensor): The batch of data.
        """
        x, _ = batch
        x_a = self.transform(x)
        x_b = self.transform(x)

        y_a = self.forward(x_a)
        y_b = self.forward(x_b)

        loss, repr_loss, std_loss, cov_loss = vicreg_loss(
            y_a, y_b, self.sim_coef, self.std_coef, self.cov_coef
        )

        # Log the three components of the loss
        self.log("val/repr_loss", repr_loss)
        self.log("val/std_loss", std_loss)
        self.log("val/cov_loss", cov_loss)

        # Log the total loss
        self.log("val/loss", loss, prog_bar=True)

        return loss


def vicreg_loss(
    y_a: Tensor,
    y_b: Tensor,
    sim_coef: float = 25.0,
    std_coef: float = 25.0,
    cov_coef: float = 1.0,
) -> Tensor:
    """VICReg loss function.

    Parameters:
        y_a (Tensor): The output of the projection head for the first
            augmented image.
        y_b (Tensor): The output of the projection head for the second
            augmented image.
        sim_coef (float): The coefficient for the similarity loss.
        std_coef (float): The coefficient for the standard deviation loss.
        cov_coef (float): The coefficient for the covariance loss.
    """
    repr_loss = F.mse_loss(y_a, y_b)

    y_a = y_a - y_a.mean(dim=0)
    y_b = y_b - y_b.mean(dim=0)

    std_a = torch.sqrt(y_a.var(dim=0) + 0.0001)
    std_b = torch.sqrt(y_b.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_a)) / 2 + torch.mean(F.relu(1 - std_b)) / 2

    batch_size = y_a.shape[0]
    feature_size = y_a.shape[1]
    cov_a = (y_a.T @ y_a) / (batch_size - 1)
    cov_b = (y_b.T @ y_b) / (batch_size - 1)
    cov_loss = off_diagonal(cov_a).pow_(2).sum().div(feature_size) + off_diagonal(
        cov_b
    ).pow_(2).sum().div(feature_size)

    loss = sim_coef * repr_loss + std_coef * std_loss + cov_coef * cov_loss

    return loss, repr_loss, std_loss, cov_loss


def off_diagonal(x: Tensor):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_transform(size: int = 32) -> Callable:
    transform = T.Compose(
        [
            T.RandomResizedCrop(
                size, interpolation=T.InterpolationMode.BICUBIC, antialias=True
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=23)],
                p=0.5,
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomSolarize(threshold=0.5, p=0.1),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform
