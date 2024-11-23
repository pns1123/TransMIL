import time

from pydantic import Field
from typing import Literal, Optional
from pydantic import root_validator
from pydantic_settings import BaseSettings


class GeneralSettings(BaseSettings):
    comment: Optional[str] = None
    seed: int = 2021
    fp16: bool = True
    amp_level: str = "O2"
    precision: int = 16
    gpus: list[int] = [2]
    multi_gpu_mode: str = "dp"
    devices: int = 1
    epochs: int = Field(1, alias="epoch")  # alias to match yaml key name
    grad_acc: int = 2
    frozen_bn: bool = False
    patience: int = 10
    server: str = "train"  # "train"  # can be set as "train" or "test"
    accelerator: str = "cpu"
    path_to_eval_checkpoint: str | None = None  # needs to be set for testing

    class Config:
        env_prefix = "GENERAL_"


class LogsSettings(BaseSettings):
    base_dir: str = "logs/"
    name: str = "neudeg"
    version_name: str = str(int(time.time()))
    run_dir: str | None = None

    class Config:
        env_prefix = "LOGS_"


class DataLoaderSettings(BaseSettings):
    batch_size: int
    num_workers: int
    mode: Literal["train", "test", "validation"]


class DataSettings(BaseSettings):
    dataset_name: str = "neudeg_data"
    data_shuffle: bool = False
    data_dir: str = "test_data/pt_files/"
    label_file: str = "test_data/dev_labels.csv"
    fold: int = 0
    nfold: int = 4
    batch_size: int = 1
    num_workers: int = 2
    train_dataloader: DataLoaderSettings = None
    test_dataloader: DataLoaderSettings = None
    validation_dataloader: DataLoaderSettings = None

    @root_validator(pre=True)
    def create_dataloaders(cls, values):
        batch_size = values.get("batch_size", 1)
        num_workers = values.get("num_workers", 2)
        values["train_dataloader"] = DataLoaderSettings(
            mode="train", batch_size=batch_size, num_workers=num_workers
        )
        values["test_dataloader"] = DataLoaderSettings(
            mode="test", batch_size=batch_size, num_workers=num_workers
        )
        values["validation_dataloader"] = DataLoaderSettings(
            mode="validation", batch_size=batch_size, num_workers=num_workers
        )
        return values

    class Config:
        env_prefix = "DATA_"


class ModelSettings(BaseSettings):
    name: str = "TransMIL"
    n_classes: int = 5

    class Config:
        env_prefix = "MODEL_"


class OptimizerSettings(BaseSettings):
    opt: str = "lookahead_radam"
    lr: float = 0.0002
    opt_eps: Optional[float] = None
    opt_betas: Optional[str] = None
    momentum: Optional[float] = None
    weight_decay: float = 0.00001

    class Config:
        env_prefix = "OPTIMIZER_"


class LossSettings(BaseSettings):
    base_loss: str = "SigmoidBCESum"

    class Config:
        env_prefix = "LOSS_"


class ConfigSettings(BaseSettings):
    General: GeneralSettings = GeneralSettings()
    Logs: LogsSettings = LogsSettings()
    Data: DataSettings = DataSettings()
    Model: ModelSettings = ModelSettings()
    Optimizer: OptimizerSettings = OptimizerSettings()
    Loss: LossSettings = LossSettings()

    class Config:
        env_nested_delimiter = "__"  # enables nested environment variables
