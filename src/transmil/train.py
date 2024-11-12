import argparse
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface
from models import ModelInterface
from config import ConfigSettings
from utils.utils import *

import pytorch_lightning as pl
from pytorch_lightning import Trainer


def main(cfg):
    # ---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    # ---->load loggers
    loggers = load_loggers(cfg)

    # ---->load callbacks
    callbacks = []  # load_callbacks(cfg)

    # ---->Define Data
    DataInterface_dict = {
        "train_batch_size": cfg.Data.train_dataloader.batch_size,
        "train_num_workers": cfg.Data.train_dataloader.num_workers,
        "test_batch_size": cfg.Data.test_dataloader.batch_size,
        "test_num_workers": cfg.Data.test_dataloader.num_workers,
        "dataset_name": cfg.Data.dataset_name,
        "dataset_cfg": cfg.Data,
    }
    dm = DataInterface(**DataInterface_dict)

    # ---->Define Model
    ModelInterface_dict = {
        "model": cfg.Model,
        "loss": cfg.Loss,
        "optimizer": cfg.Optimizer,
        "data": cfg.Data,
        "log": cfg.Logs.run_dir,
    }
    print(ModelInterface_dict)
    model = ModelInterface(**ModelInterface_dict)

    # ---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0,
        # logger=loggers,
        callbacks=callbacks,
        max_epochs=cfg.General.epochs,
        accelerator="cpu",
        # devices=cfg.General.devices,
        # gpus=cfg.General.gpus,
        # amp_level=cfg.General.amp_level,
        # precision=cfg.General.precision,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=2,
        log_every_n_steps=1,
    )

    # ---->train or test
    if cfg.General.server == "train":
        trainer.fit(model=model, datamodule=dm)
    else:
        model_paths = list(cfg.log_path.glob("*.ckpt"))
        model_paths = [
            str(model_path) for model_path in model_paths if "epoch" in str(model_path)
        ]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)


if __name__ == "__main__":
    cfg = ConfigSettings()

    main(cfg)
