"""
Based on https://github.com/ZikangZhou/QCNet/blob/main/datamodules/argoverse_v2_datamodule.py
"""


from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import WaymoDataset


class WaymoDataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 **kwargs) -> None:
        super(WaymoDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaymoDataset(
            data_root=self.root, split_dir='processed_scenarios_training',
            info_file='processed_scenarios_training_infos.pkl',
            # interval=5,
        )
        self.val_dataset = WaymoDataset(
            data_root=self.root, split_dir='processed_scenarios_validation',
            info_file='processed_scenarios_validation_infos.pkl',
        )
        self.test_dataset = WaymoDataset(
            data_root=self.root, split_dir='processed_scenarios_testing',
            info_file='processed_scenarios_testing_infos.pkl',
            test=True,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
