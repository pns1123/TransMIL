import json
import pandas as pd
from pathlib import Path
import torch

from datasets import DataInterface
from models import ModelInterface, TransMIL
from config import ConfigSettings
from utils.utils import load_loggers

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
        logger=loggers,
        callbacks=callbacks,
        max_epochs=cfg.General.epochs,
        accelerator=cfg.General.accelerator,
        # devices=cfg.General.devices,
        # gpus=cfg.General.gpus,
        # amp_level=cfg.General.amp_level,
        # precision=cfg.General.precision,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )

    # ---->train or test
    match cfg.General.server:
        case "train":
            trainer.fit(model=model, datamodule=dm)
        case "test":
            model_path = cfg.General.path_to_eval_checkpoint

            if model_path is None:
                raise ValueError("path_to_eval_checkpoint needs to be set for testing.")

            new_model = model.load_from_checkpoint(checkpoint_path=model_path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)
        case "full_predict":
            model_path = cfg.General.path_to_eval_checkpoint
            Path(cfg.Logs.run_dir / "attention_scores").mkdir(
                parents=True, exist_ok=True
            )

            if model_path is None:
                raise ValueError("path_to_eval_checkpoint needs to be set for testing.")

            checkpoint = torch.load(model_path)  # , map_location=torch.device("cpu"))
            state_dict = {
                key.removeprefix("model."): val
                for key, val in checkpoint["state_dict"].items()
            }
            model = TransMIL.TransMIL(n_classes=cfg.Model.n_classes)
            model.load_state_dict(state_dict)

            full_data = pd.read_csv(cfg.Data.label_file, index_col=0)

            slide_ids = [image_name.split(".")[0] for image_name in full_data["image"]]
            labels = [json.loads(label_list) for label_list in full_data["label_list"]]

            full_pred = {}
            for slide_id, label in zip(slide_ids, labels):
                full_path = Path(cfg.Data.data_dir) / f"{slide_id}pyramid.pt"
                features = torch.load(full_path)[None, :, :]
                logits = model(data=features)["logits"]
                sa1 = model(data=features)["sa1"].to(torch.float16)
                sa2 = model(data=features)["sa2"].to(torch.float16)

                full_pred[slide_id] = {
                    "logits": logits.cpu().tolist(),
                    "labels": labels,
                }

                torch.save(
                    {"sa1": sa1.cpu(), "sa2": sa2.cpu()},
                    cfg.Logs.run_dir / "attention_scores" / f"{slide_id}.pt",
                )
                del sa1, sa2, features
                print("completed prediction for", slide_id)

            with open(cfg.Logs.run_dir / "full_pred.json", "w") as fp:
                json.dump(full_pred, fp)

        case _:
            raise ValueError(f"server {cfg.General.server} invalid")


if __name__ == "__main__":
    cfg = ConfigSettings()
    print(cfg)
    main(cfg)
