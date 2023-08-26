"""This module implements the NCC-based metrics for benchmarking SSL.
"""
import statistics
from typing import List, Union

import lightning as L
import sklearn.neighbors as neighbors
import torch
from lightning.pytorch.loggers import CSVLogger
from numpy.typing import NDArray
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from ssl_research.models.cnn import CNN
from torch import Tensor
from torch.utils.data import DataLoader


class SSLMetricsCallback(L.Callback):
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

                scores = ncc_accuracy(x, y)
                self.log_cv(f"ncc_layer_{i}", scores, trainer)
                print(f"ncc_layer_{i}: {statistics.mean(scores):.4f}")

                lp_scores = linear_probing_accuracy(x, y)
                self.log_cv(f"linear_probing_layer_{i}", lp_scores, trainer)
                print(f"linear_probing_layer_{i}: {statistics.mean(lp_scores):.4f}")

    @staticmethod
    def log_cv(name: str, scores: List[float], trainer: L.Trainer):
        """
        Logs the cross validation scores to the loggers
        For CSV loggers, logs all scores as separate columns,
        For other loggers, logs the mean of the scores.

        Parameters:
            name (str): The name of the metric
            scores (list): The list of scores
            trainer (Trainer): The trainer object
        """
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                logger.log_metrics(
                    {f"{name}_fold_{i}": score for i, score in enumerate(scores)}
                )
            else:
                logger.log_metrics({name: statistics.mean(scores)})


def ncc_accuracy(
    X: Union[Tensor, NDArray],
    y: Union[Tensor, NDArray],
) -> float:
    """
    Computes the NCC accuracy score using scikit-learn's NearestCentroid class.

    Parameters:
        X (tensor or array): Input vectors
        y (tensor or array): Input classifications
    """
    if isinstance(X, Tensor):
        X = X.cpu().numpy()
    if isinstance(y, Tensor):
        y = y.cpu().numpy()

    clf = neighbors.NearestCentroid()
    cross_val_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    return cross_val_scores


def linear_probing_accuracy(
    X: Union[Tensor, NDArray],
    y: Union[Tensor, NDArray],
) -> float:
    """
    Computes the linear probing accuracy score using scikit-learn's LinearSVC class.

    Parameters:
        X (tensor or array): Input vectors
        y (tensor or array): Input classifications
    """
    if isinstance(X, Tensor):
        X = X.cpu().numpy()
    if isinstance(y, Tensor):
        y = y.cpu().numpy()

    clf = LinearSVC(dual="auto")
    cross_val_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    return cross_val_scores
