import glob
import os

from ..utils import utils
from .daseg_base import Cityscapes_Compatible_Dataset

log = utils.get_logger(__name__)

CLASS_INFO = {
    19: {
        "colors": [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ],
        "void": [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1],
        "valid": [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ],
        "names": [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ],
    },
    16: {
        "valid": [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            28,
            32,
            33,
        ],
        "names": [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "sky",
            "person",
            "rider",
            "car",
            "bus",
            "motorcycle",
            "bicycle",
        ],
    },
    13: {
        "valid": [
            7,
            8,
            11,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            28,
            32,
            33,
        ],
        "names": [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "sky",
            "person",
            "rider",
            "car",
            "bus",
            "motorcycle",
            "bicycle",
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


class Cityscapes(Cityscapes_Compatible_Dataset):
    # 2048x1024
    # train: 2975
    # val: 500
    def __init__(self, **kwargs):
        super().__init__(class_info=CLASS_INFO, **kwargs)

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        self.files = glob.glob(f"{self.images_base}/**/*.png", recursive=True)
        log.info(f"[cityscapes] Found {len(self.files)} {self.split} images")

        if self.superpixel:
            self.sps_base = f"cityscapes_{self.split}_sp"
            self.sp = glob.glob(f"cityscapes_{self.split}_sp/*.dat")
            assert len(self.sp) == len(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__"""

        img_path = self.files[index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        if self.superpixel:
            sp_path = os.path.join(
                self.sps_base,
                os.path.basename(img_path).split(".")[0] + ".dat",
            )
            img, lbl, sp = self.post_process(img_path, lbl_path, sp_path=sp_path)
            return img, lbl, sp, self.files[index]
        else:
            img, lbl = self.post_process(img_path, lbl_path)
            return img, lbl, self.files[index]
