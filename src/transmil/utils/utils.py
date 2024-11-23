from functools import reduce
from pathlib import Path

# ---->read yaml
import yaml
from addict import Dict


def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


# ---->load Loggers
from pytorch_lightning import loggers as pl_loggers


def load_loggers(cfg):
    log_path = cfg.Logs.base_dir
    Path(log_path).mkdir(exist_ok=True, parents=True)
    log_name = cfg.Logs.name
    version_name = cfg.Logs.version_name
    cfg.Logs.run_dir = (
        Path(log_path) / log_name / version_name / f"split{cfg.Data.fold}"
    )
    print(f"---->Log dir: {cfg.Logs.run_dir}")

    # ---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(
        log_path + str(log_name),
        name=version_name,
        version=f"fold{cfg.Data.fold}",
        log_graph=True,
        default_hp_metric=False,
    )
    # ---->CSV
    print("CSV LOGGER PATH: ", log_path + str(log_name))
    csv_logger = pl_loggers.CSVLogger(
        log_path + str(log_name),
        name=version_name,
        version=f"split{cfg.Data.fold}",
    )

    return csv_logger


def __split_tensor(metric: str, t):
    if t.dim() != 1:
        raise ValueError(
            "tensor {t} must be of dimension 1 but is of dimension {t.dim()}"
        )

    return {f"{metric}_{k}": x for k, x in enumerate(t)}


def split_metrics_tensors(metrics: dict) -> dict:
    scalar_metrics = [__split_tensor(key, val) for key, val in metrics.items()]
    return reduce(lambda acc, x: {**acc, **x}, scalar_metrics, {})


# ---->load Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def load_callbacks(cfg):
    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode="min",
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == "train":
        Mycallbacks.append(
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=str(cfg.log_path),
                filename="{epoch:02d}-{val_loss:.4f}",
                verbose=True,
                save_last=True,
                save_top_k=1,
                mode="min",
                save_weights_only=True,
            )
        )
    return Mycallbacks


# ---->val loss
import torch
import torch.nn.functional as F


def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i]) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(len(y))])
    loss = -torch.sum(x_log) / len(y)
    return loss
