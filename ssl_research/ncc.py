"""This module implements the NCC-based metrics for benchmarking SSL.
"""
from typing import List, Optional, Union

import lightning as L
import sklearn.metrics as metrics
import sklearn.neighbors as neighbors
import torch
from numpy.typing import NDArray
from ssl_research.models.cnn import CNN
from torch import Tensor
from torch.utils.data import DataLoader


class NCCAccuracyCallback(L.Callback):
    """
    This callback calculates the NCC accuracy score for a model at each validation
    epoch, using the validation data.
    """

    def __init__(self, loader: DataLoader):
        super().__init__()

        self.loader = loader

    @staticmethod
    def get_features(x: Tensor, model: CNN) -> List[Tensor]:
        features = list()
        for y in model.features_forward(x):
            if len(y.shape) == 4:
                y = y.mean(dim=[2, 3])
            features.append(y)
        return features

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """
        Calculates the NCC accuracy score for the model at the end of each validation
        epoch.

        Parameters:
            trainer (Trainer): The trainer object
            pl_module (LightningModule): The module being trained
        """
        # Forward pass through the model
        pl_module.eval()

        with torch.no_grad():
            num_features = pl_module.num_features
            X = [[] for _ in range(num_features)]
            y = []

            for batch in self.loader:
                x_batch, y_batch = batch
                x_batch = x_batch.to(pl_module.device)
                y.append(y_batch)
                features = self.get_features(x_batch, pl_module.model)
                for i in range(num_features):
                    X[i].append(features[i])

            X = [torch.cat(x) for x in X]
            y = torch.cat(y)

            for i, x in enumerate(X):
                x = x.cpu().numpy()

                score = ncc_accuracy(x, y)
                trainer.logger.log_metrics({f"ncc_accuracy_layer_{i}": score})
                print(f"ncc_accuracy_layer_{i}: {score}")

            # X, y = [], []
            # for batch in self.loader:
            #     x_batch, y_batch = batch
            #     batch_size = x_batch.shape[0]
            #     x_batch = x_batch.to(pl_module.device)
            #     y.append(y_batch)
            #     X.append(pl_module(x_batch).cpu().reshape(batch_size, -1))
            # X = torch.cat(X)
            # y = torch.cat(y)

        # # Calculate the NCC accuracy score
        # score = ncc_accuracy(X, y)
        # trainer.logger.log_metrics({"ncc_accuracy": score})


def ncc_accuracy(
    X_train: Union[Tensor, NDArray],
    y_train: Union[Tensor, NDArray],
    X_test: Optional[Union[Tensor, NDArray]] = None,
    y_test: Optional[Union[Tensor, NDArray]] = None,
) -> float:
    """
    Computes the NCC accuracy score using scikit-learn's NearestCentroid class.

    Parameters:
        X_train (tensor or array): Input vector used to train the
            NearestCentroid classifier
        y_train (tensor or array): Input classifications used to train the
            NearestCentroid classifier
        X_test (tensor or array) Input vectors for calculating the accuracy.
            If None, the training data is used.
        y_test (tensor or array): Inputs classifications calculating the accuracy.
            If None, the training data is used.
    """
    if isinstance(X_train, Tensor):
        X_train = X_train.cpu().numpy()
    if isinstance(y_train, Tensor):
        y_train = y_train.cpu().numpy()
    if isinstance(X_test, Tensor):
        X_test = X_test.cpu().numpy()
    if isinstance(y_test, Tensor):
        y_test = y_test.cpu().numpy()

    clf = neighbors.NearestCentroid()
    clf.fit(X_train, y_train)

    if X_test is None or y_test is None:
        y_hat = clf.predict(X_train)
        return metrics.accuracy_score(y_train, y_hat)
    else:
        y_hat = clf.predict(X_test)
        return metrics.accuracy_score(y_test, y_hat)
