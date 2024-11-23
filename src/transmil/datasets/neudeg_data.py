import json
import random
import os
import pandas as pd
import torch
import torch.utils.data as data

from pathlib import Path
from torch.utils.data import dataloader


class NeudegData(data.Dataset):
    def __init__(self, mode, dataset_cfg=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        # ---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        all_labels = pd.read_csv(self.dataset_cfg.label_file, index_col=0)
        self.slide_data = all_labels[all_labels[f"split{self.fold}"] == mode]

        # ---->order
        self.shuffle = self.dataset_cfg.data_shuffle

        # ---->split dataset
        # self.data = self.slide_data[self.slide_data["group"] == state]["filename"]
        # self.label = self.slide_data[self.slide_data["group"] == state]["label0"]
        self.slide_ids = [
            image_name.split(".")[0] for image_name in self.slide_data["image"]
        ]
        self.labels = [
            json.loads(label_list) for label_list in self.slide_data["label_list"]
        ]

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        full_path = Path(self.feature_dir) / f"{slide_id}pyramid.pt"
        features = torch.load(full_path)

        # ----> shuffle
        if self.shuffle:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        return features, label
