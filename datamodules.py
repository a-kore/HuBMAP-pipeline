from typing import Any, Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import HuBMAPDatset
from torchvision import transforms
import os


# from albumentations import *
# import cv2


class HuBMAPDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: str = './',
            image_size: int = 256,
            num_workers: int = -1,
            batch_size: int = 32,
            pin_memory: bool = True,
            drop_last: bool = False,
            split: int = 0.8,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(self, *args, **kwargs)

        self.image_size = (image_size, image_size)
        self.dims = (3, self.image_size[0], self.image_size[1])
        self.data_dir = data_dir
        if num_workers == -1:
            self.num_workers = os.cpu_count()
        else:
            self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.split = split

    def train_dataloader(self) -> DataLoader:
        dataset = HuBMAPDatset(
            self.data_dir,
            transform=self.train_transform(),
            split=self.split
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=True
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        dataset = HuBMAPDatset(
            self.data_dir,
            train=False,
            transform=self.val_transform(),
            split=self.split
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=False
        )
        return loader

    def train_transform(self) -> Callable:
        preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
            # transforms.Resize(self.image_size),
            # transforms.RandomResizedCrop(self.image_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomChoice([transforms.RandomRotation(30),
            #                          transforms.RandomErasing(),
            #                          ]),
        ])
        return preprocessing

    def val_transform(self) -> Callable:
        preprocessing = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(self.image_size),
        ])
        return preprocessing
    # def train_transform(self) -> Callable:
    #     preprocessing = Compose([
    #         torchvision.transforms.ToTensor(),
    #         HorizontalFlip(),
    #         VerticalFlip(),
    #         RandomRotate90(),
    #         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
    #                          border_mode=cv2.BORDER_REFLECT),
    #         OneOf([
    #             OpticalDistortion(p=0.3),
    #             GridDistortion(p=.1),
    #             IAAPiecewiseAffine(p=0.3),
    #         ], p=0.3),
    #         OneOf([
    #             HueSaturationValue(10, 15, 10),
    #             CLAHE(clip_limit=2),
    #             RandomBrightnessContrast(),
    #         ], p=0.3),
    #     ], p=1.0)
    #     return preprocessing
    #
    # def val_transform(self) -> Callable:
    #     preprocessing = torchvision.transforms.Compose([
    #         # torchvision.transforms.Resize((self.image_size, self.image_size)),
    #         torchvision.transforms.ToTensor(),
    #     ])
    #     return preprocessing
