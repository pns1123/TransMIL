import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

# ---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import split_metrics_tensors

# ---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

# ---->
import pytorch_lightning as pl


class ModelInterface(pl.LightningModule):
    # ---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs["log"]

        # ---->Metrics
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.classification.MultilabelAccuracy(
                    num_labels=self.n_classes, threshold=0.0, average="none",
                ),
                torchmetrics.classification.MultilabelPrecision(
                    num_labels=self.n_classes, threshold=0.0, average="none",
                ),
                torchmetrics.classification.MultilabelRecall(
                    num_labels=self.n_classes, threshold=0.0, average="none",
                ),
                torchmetrics.classification.MultilabelAUROC(
                    num_labels=self.n_classes, average="none"
                ),
            ]
        )
        self.valid_metrics = metrics.clone(prefix="val_")
        self.training_metrics = metrics.clone(prefix="train_")
        self.test_metrics = metrics.clone(prefix="test_")

        # --->random
        self.shuffle = kargs["data"].data_shuffle
        self.count = 0

    # ---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits = self.model(data=data, label=label)

        loss = self.loss(logits, label)
        self.log_dict({"loss": loss}, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return {"loss": loss, "logits": logits, "label": label}

    def training_epoch_end(self, training_step_outputs):
        logits = torch.stack([x["logits"] for x in training_step_outputs], dim=0)
        target = torch.stack([x["label"] for x in training_step_outputs], dim=0).to(torch.uint8)

        metrics = self.training_metrics(logits.squeeze(), target.squeeze())

        self.log_dict(split_metrics_tensors(metrics), prog_bar=False, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        data, label = batch
        logits = self.model(data=data, label=label)
        return {"logits": logits, "label": label}

    def validation_epoch_end(self, val_step_outputs):
        logits = torch.stack([x["logits"] for x in val_step_outputs], dim=0)
        target = torch.stack([x["label"] for x in val_step_outputs], dim=0).to(torch.uint8)

        metrics = self.valid_metrics(logits.squeeze(), target.squeeze())

        self.log_dict(split_metrics_tensors(metrics), prog_bar=False, on_epoch=True, logger=True)

        if self.shuffle:
            self.count = self.count + 1
            random.seed(self.count * 50)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        data, label = batch
        logits = self.model(data=data, label=label)
        return {"logits": logits, "label": label}

    def test_epoch_end(self, test_step_outputs):
        logits = torch.stack([x["logits"] for x in test_step_outputs], dim=0)
        target = torch.stack([x["label"] for x in test_step_outputs], dim=0).to(
            torch.uint8
        )

        metrics = self.test_metrics(logits.squeeze(), target.squeeze())
        for keys, values in metrics.items():
            print(f"{keys} = {values}")
            metrics[keys] = values.cpu().numpy()

        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / "result.csv")

    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if "_" in name:
            camel_name = "".join([i.capitalize() for i in name.split("_")])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(f"models.{name}"), camel_name)
        except:
            raise ValueError("Invalid Module File Name or Invalid Class Name!")
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.model_dump().keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)
