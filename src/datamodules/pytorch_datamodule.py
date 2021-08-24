import os
from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10, MNIST, ImageFolder
from torchvision.transforms import transforms

from ..augmentations import RandAugment


class PytorchDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 10,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_classes = num_classes

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def train_len(self):
        assert self.data_train is not None
        return len(self.data_train)

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


class MNISTDataModule(PytorchDataModule):
    def __init__(
        self,
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = os.path.join(self.data_dir, "mnist")
        self.train_val_test_split = train_val_test_split
        self.prepare_transform()

    def prepare_transform(self):
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dims = (1, 28, 28)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        trainset = MNIST(self.data_dir, train=True, transform=self.transforms)
        testset = MNIST(self.data_dir, train=False, transform=self.transforms)

        self.data_train, self.data_val = random_split(trainset, self.train_val_test_split[:2])
        self.data_test = testset


class CIFAR10DataModule(PytorchDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = os.path.join(self.data_dir, "cifar")
        self.prepare_transform()

    def prepare_transform(self):
        self.transforms_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.transforms_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.dims = (3, 32, 32)

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        trainset = CIFAR10(self.data_dir, train=True, transform=self.transforms_train)
        testset = CIFAR10(self.data_dir, train=False, transform=self.transforms_test)

        self.data_train = trainset
        self.data_val = testset
        self.data_test = testset


class IMAGENETDataModule(PytorchDataModule):
    def __init__(
        self,
        randaug_m=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.randaug_m = randaug_m

        self.data_dir = os.path.join(self.data_dir, "imagenet")
        self.data_dir_train = os.path.join(self.data_dir, "train")
        self.data_dir_test = os.path.join(self.data_dir, "val")
        self.prepare_transform()

    def prepare_transform(self):
        _IMAGENET_MEAN = [0.485, 0.456, 0.406]
        _IMAGENET_STD = [0.229, 0.224, 0.225]
        self.transforms_train = transforms.Compose(
            [
                # transforms.RandomResizedCrop(224),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )
        if self.randaug_m > 0:
            self.transforms_train.transforms.insert(0, RandAugment(2, self.randaug_m))
        self.transforms_test = transforms.Compose(
            [
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )
        self.dims = (3, 224, 224)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        trainset = ImageFolder(self.data_dir_train, transform=self.transforms_train)
        testset = ImageFolder(self.data_dir_test, transform=self.transforms_test)

        self.data_train = trainset
        self.data_val = testset
        self.data_test = testset
