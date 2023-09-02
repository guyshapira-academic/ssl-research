import os
import re
import statistics
from typing import List, Union

import numpy as np
import pandas as pd
import ssl_research.models.cnn as cnn
import ssl_research.vicreg as vicreg
import torch
import torchvision
import wandb
from numpy._typing import NDArray
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from torch import Tensor
from tqdm import tqdm


def parse_run_name(filename: str):
    """
    Parse run name and extract architecture type, width and depth.

    Parameters:
        filename (str): Filename of run.
    """
    # Define regular expression pattern for parsing filenames
    pattern = re.compile(
        r"(?P<type>vanilla|resnet)_w(?P<width>\d+(\-\d+)?)(_d(?P<depth>\d+(\-\d+)?))?"
    )

    # Use regular expression to match filename and extract groups
    match = pattern.match(filename)

    if match:
        # Extract groups from match object
        architecture_type = match.group("type")

        # Convert width to float
        width_str = match.group("width")
        width = float(width_str.replace("-", "."))

        # Convert depth to float, if present
        depth_str = match.group("depth")
        if depth_str:
            depth = float(depth_str.replace("-", "."))
        else:
            depth = None  # Set depth as None if not present in filename

        return architecture_type, width, depth
    else:
        # Filename did not match pattern
        return None


def process_df(df: pd.DataFrame):
    df = df.groupby("trainer/global_step", as_index=False).sum().replace(0, np.nan)
    df["trainer/global_step"] = df["trainer/global_step"].replace(np.nan, 0)
    df["epoch"] = df["trainer/global_step"] // 176
    df = df.replace(0, np.nan)
    df = df.dropna(axis=0)
    df = df.groupby("epoch", as_index=False).sum()

    df = df[
        [
            "epoch",
            "ncc_layer_0",
            "linear_probing_layer_0",
            "ncc_layer_1",
            "linear_probing_layer_1",
            "ncc_layer_2",
            "linear_probing_layer_2",
            "ncc_layer_3",
            "linear_probing_layer_3",
            "val/loss",
        ]
    ]
    return df


def model_from_checkpoint(checkpoint_path: str, use_checkpoint=True):
    basename = os.path.basename(checkpoint_path)
    run_name = os.path.splitext(basename)[0]
    architecture_type, width, depth = parse_run_name(run_name)
    if architecture_type == "vanilla":
        model = cnn.vanilla(output_dim=128, width_factor=width, depth_factor=depth)
    elif architecture_type == "resnet":
        model = cnn.resnet(output_dim=128, width_factor=width, depth_factor=depth)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
    if use_checkpoint:
        vicreg_model = vicreg.VICReg.load_from_checkpoint(
            model=model, checkpoint_path=checkpoint_path
        )
        model = vicreg_model.model
    return model


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

    clf = LinearSVC(dual="auto", verbose=1, max_iter=10000)
    cross_val_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    return cross_val_scores


def get_features(x: Tensor, model, mean=False) -> List[Tensor]:
    features = list()
    if mean:
        in_x = x.mean(dim=[2, 3])
    else:
        in_x = x.flatten(start_dim=1)
    features.append(in_x)
    for y in model.features_forward(x):
        if len(y.shape) == 4 and mean:
            y = y.mean(dim=[2, 3])
        elif len(y.shape) == 4 and not mean:
            y = y.flatten(start_dim=1)
        features.append(y)
    return features


def evaluate_model(model, loader):
    model.eval()

    with torch.no_grad():
        num_features = model.num_features
        X = [[] for _ in range(num_features + 1)]
        y = []

        for batch in tqdm(loader):
            x_batch, y_batch = batch
            x_batch = x_batch.to("cuda")
            y.append(y_batch)
            features = get_features(x_batch, model)

            for i in range(num_features + 1):
                X[i].append(features[i])

        X = [torch.cat(x) for x in X]
        y = torch.cat(y)

        score_dict = dict()
        for i, x in enumerate(tqdm(X, leave=False)):
            x = x.cpu().numpy()

            scores = ncc_accuracy(x, y)
            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores)
            if i == 0:
                score_dict["ncc_input"] = mean_score
                score_dict["ncc_input_std"] = std_score
            elif i == 4:
                score_dict["ncc_output"] = mean_score
                score_dict["ncc_output_std"] = std_score
            else:
                score_dict[f"ncc_layer_{i}"] = mean_score
                score_dict[f"ncc_layer_{i}_std"] = std_score

        return score_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Iterate over WandB runs
    api = wandb.Api()

    runs = api.runs("guyshapira-academic/SSL Research")
    checkpoints = dict()
    os.makedirs("checkpoints", exist_ok=True)
    # Download latest checkpoint from WandB
    for r in runs:
        rname = r.name
        architecture_type, width, depth = parse_run_name(rname)
        for artifact in r.logged_artifacts():
            if artifact.metadata.get("original_filename") == "vicreg-epoch=999.ckpt":
                if f"{rname}.ckpt" in os.listdir("checkpoints"):
                    checkpoints[rname] = os.path.join("checkpoints", f"{rname}.ckpt")
                    continue
                artifact.download(root="checkpoints")
                os.rename(
                    os.path.join("checkpoints", "model.ckpt"),
                    os.path.join("checkpoints", f"{rname}.ckpt"),
                )

    # Get cifar100 test dataset and loader
    cifar100_test = torchvision.datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    cifar100_test_loader = torch.utils.data.DataLoader(
        cifar100_test, batch_size=64, shuffle=False
    )

    score_dicts = list()
    for run_name, checkpoint_path in tqdm(checkpoints.items()):
        architecture_type, width, depth = parse_run_name(run_name)
        init_model = model_from_checkpoint(checkpoint_path, use_checkpoint=False)
        init_model.cuda()
        init_scores = evaluate_model(init_model, cifar100_test_loader)
        init_scores_tmp = dict()
        for k, v in init_scores.items():
            init_scores_tmp[f"init_{k}"] = v
        init_scores = init_scores_tmp
        model = model_from_checkpoint(checkpoint_path)
        model.cuda()
        scores = evaluate_model(model, cifar100_test_loader)
        scores.update(init_scores)
        scores["run_name"] = run_name
        scores["architecture_type"] = architecture_type
        scores["width"] = width
        scores["depth"] = depth
        scores["num_parameters"] = count_parameters(model)
        score_dicts.append(scores)
    df = pd.DataFrame(score_dicts)
    df.to_csv("scores.csv", index=False)


if __name__ == "__main__":
    main()
