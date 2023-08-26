"""This module contains tests for the VICReg training loop.
"""
import lightning as L
import pytest
from ssl_research import ncc
from ssl_research.models.cnn import vanilla20
from ssl_research.vicreg import VICReg
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


@pytest.fixture
def dataset():
    return datasets.FakeData(
        size=100,
        image_size=(3, 64, 64),
        num_classes=10,
        transform=transforms.ToTensor(),
    )


def test_vicreg_loop(dataset: Dataset):
    loader = DataLoader(dataset, batch_size=32)

    loader_val = DataLoader(dataset, batch_size=32)

    ncc_callback = ncc.SSLMetricsCallback(loader_val)

    model = vanilla20(32)
    vicreg = VICReg(
        model,
        projector_features=64,
        sim_coef=0.1,
        cov_coef=0.1,
        std_coef=0.1,
        lr=0.1,
        weight_decay=0.0,
        num_epochs=10,
    )

    trainer = L.Trainer(
        fast_dev_run=True,
        max_epochs=10,
        callbacks=[ncc_callback],
    )
    trainer.fit(vicreg, loader, val_dataloaders=loader)
