import glob
import os

from ..utils import utils
from .base_seg import Cityscapes_Compatible_Dataset

log = utils.get_logger(__name__)


CLASS_INFO = {
    19: {
        "valid": [
            3,
            4,
            2,
            21,
            5,
            7,
            15,
            9,
            6,
            16,
            1,
            10,
            17,
            8,
            18,
            19,
            20,
            12,
            11,
        ],
        "names": [
            "unlabelled",
            "Road",
            "Sidewalk",
            "Building",
            "Wall",
            "Fence",
            "Pole",
            "Traffic_light",
            "Traffic_sign",
            "Vegetation",
            "Terrain",
            "sky",
            "Pedestrian",
            "Rider",
            "Car",
            "Truck",
            "Bus",
            "Train",
            "Motorcycle",
            "Bicycle",
        ],
    },
    16: {
        "valid": [
            3,
            4,
            2,
            21,
            5,
            7,
            15,
            9,
            6,
            1,
            10,
            17,
            8,
            19,
            12,
            11,
        ],
        "names": [
            "unlabelled",
            "Road",
            "Sidewalk",
            "Building",
            "Wall",
            "Fence",
            "Pole",
            "Traffic_light",
            "Traffic_sign",
            "Vegetation",
            "sky",
            "Pedestrian",
            "Rider",
            "Car",
            "Bus",
            "Motorcycle",
            "Bicycle",
        ],
    },
    13: {
        "valid": [
            3,
            4,
            2,
            15,
            9,
            6,
            1,
            10,
            17,
            8,
            19,
            12,
            11,
        ],
        "names": [
            "unlabelled",
            "Road",
            "Sidewalk",
            "Building",
            "Traffic_light",
            "Traffic_sign",
            "Vegetation",
            "sky",
            "Pedestrian",
            "Rider",
            "Car",
            "Bus",
            "Motorcycle",
            "Bicycle",
        ],
        "colors": [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 60, 100],
            [0, 0, 230],
            [119, 11, 32],
        ],
    },
}


class Synthia(Cityscapes_Compatible_Dataset):
    # 1914x1052
    # train: 9400
    def __init__(self, **kwargs):
        super().__init__(class_info=CLASS_INFO, is_imageio=True, **kwargs)

        self.image_base_path = os.path.join(self.root, "RGB/train")
        self.label_base_path = os.path.join(self.root, "GT/LABELS")
        self.ids = glob.glob(f"{self.label_base_path}/**/*.png", recursive=True)
        log.info("[synthia] Found {} images".format(len(self.ids)))

        if self.superpixel:
            self.sps_base = os.path.join(self.root, "synthia_sp")
            self.sp = glob.glob(f"{self.sps_base}/*.dat")
            assert len(self.sp) == len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ids = self.ids[index]
        img_path = os.path.join(self.image_base_path, ids.split("/")[-1])
        lbl_path = ids
        sp_path = (
            os.path.join(
                self.sps_base,
                os.path.basename(img_path).split(".")[0] + ".dat",
            )
            if self.superpixel
            else None
        )

        ret = self.post_process(img_path, lbl_path, sp_path=sp_path)
        ret["index"] = index
        return ret
