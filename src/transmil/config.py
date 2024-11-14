from pydantic import Field
from typing import Optional
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
    epochs: int = Field(200, alias="epoch")  # alias to match yaml key name
    grad_acc: int = 2
    frozen_bn: bool = False
    patience: int = 10
    server: str = "train"  # can be set as "train" or "test"
    accelerator: str = "cpu"

    class Config:
        env_prefix = "GENERAL_"


class LogsSettings(BaseSettings):
    base_dir: str = "logs/"
    name: str = "TEST_NAME"
    version_name: str = "TEST_VERSION"
    run_dir: str | None = None

    class Config:
        env_prefix = "LOGS_"


class DataLoaderSettings(BaseSettings):
    batch_size: int = 1
    num_workers: int = 2


class DataSettings(BaseSettings):
    dataset_name: str = "neudeg_data"
    data_shuffle: bool = False
    data_dir: str = "test_data/pt_files/"
    label_dir: str = "test_data"
    fold: int = 0
    nfold: int = 4
    train_dataloader: DataLoaderSettings = DataLoaderSettings()
    test_dataloader: DataLoaderSettings = DataLoaderSettings()

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
