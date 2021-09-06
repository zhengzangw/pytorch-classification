import glob
import os

from ..utils import utils
from .base_seg import Cityscapes_Compatible_Dataset

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
    }
}


class GTA5(Cityscapes_Compatible_Dataset):
    # 1914x1052
    # train: 24966
    def __init__(self, **kwargs):
        super().__init__(class_info=CLASS_INFO, **kwargs)

        self.image_base_path = os.path.join(self.root, "images")
        self.label_base_path = os.path.join(self.root, "labels")
        self.ids = glob.glob(f"{self.label_base_path}/**/*.png", recursive=True)
        log.info(f"[gta5] Found {len(self.ids)} images")

        if self.superpixel:
            self.sps_base = os.path.join(self.root, "gta5_sp")
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
