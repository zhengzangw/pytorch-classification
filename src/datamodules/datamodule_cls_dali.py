import os

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from pytorch_lightning import LightningDataModule


class LightningWrapper(DALIClassificationIterator):
    def __init__(self, *kargs, **kvargs):
        super().__init__(*kargs, **kvargs)

    def __next__(self):
        out = super().__next__()
        # DDP is used so only one pipeline per process
        # also we need to transform dict returned by DALIClassificationIterator to iterable
        # and squeeze the lables
        out = out[0]
        return [out[k] if k != "label" else torch.squeeze(out[k]).long() for k in self.output_map]


@pipeline_def
def create_dali_pipeline(
    data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True
):
    images, labels = fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=is_training,
        pad_last_batch=True,
        name="Reader",
    )
    dali_device = "cpu" if dali_cpu else "gpu"
    decoder_device = "cpu" if dali_cpu else "mixed"
    device_memory_padding = 211025920 if decoder_device == "mixed" else 0
    host_memory_padding = 140544512 if decoder_device == "mixed" else 0
    if is_training:
        images = fn.decoders.image_random_crop(
            images,
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            random_aspect_ratio=[0.8, 1.25],
            random_area=[0.1, 1.0],
            num_attempts=100,
        )
        images = fn.resize(
            images,
            device=dali_device,
            resize_x=crop,
            resize_y=crop,
            interp_type=types.INTERP_TRIANGULAR,
        )
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
        images = fn.resize(
            images,
            device=dali_device,
            size=size,
            mode="not_smaller",
            interp_type=types.INTERP_TRIANGULAR,
        )
        mirror = False

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )
    labels = labels.gpu()
    return images, labels


class IMAGENET_DALI_DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 10,
        dali_cpu: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_classes = num_classes
        self.dali_cpu = dali_cpu

        self.data_dir = os.path.join(self.data_dir, "imagenet")
        self.data_dir_train = os.path.join(self.data_dir, "train")
        self.data_dir_test = os.path.join(self.data_dir, "val")

    @property
    def train_len(self):
        # 1281167
        assert self.loader_train is not None
        return self.loader_train.size

    def prepare_data(self):
        pass

    def setup(self, stage):
        device_id = self.trainer.local_rank
        shard_id = self.trainer.global_rank
        num_shards = self.trainer.world_size

        pipeline_train = create_dali_pipeline(
            batch_size=self.batch_size,
            num_threads=self.num_workers,
            device_id=device_id,
            seed=12 + device_id,
            # -----
            data_dir=self.data_dir_train,
            crop=224,
            size=256,
            shard_id=shard_id,
            num_shards=num_shards,
            dali_cpu=self.dali_cpu,
            is_training=True,
        )
        pipeline_train.build()
        self.loader_train = LightningWrapper(
            pipeline_train,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

        pipeline_test = create_dali_pipeline(
            batch_size=self.batch_size,
            num_threads=self.num_workers,
            device_id=device_id,
            seed=12 + device_id,
            # -----
            data_dir=self.data_dir_test,
            crop=224,
            size=256,
            shard_id=shard_id,
            num_shards=num_shards,
            dali_cpu=self.dali_cpu,
            is_training=False,
        )
        pipeline_test.build()
        self.loader_test = LightningWrapper(
            pipeline_test,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

    def train_dataloader(self):
        return self.loader_train

    def val_dataloader(self):
        return self.loader_test

    def test_dataloader(self):
        return self.loader_test
