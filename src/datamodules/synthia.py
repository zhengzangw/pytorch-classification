import glob
import os

from .daseg_base import Cityscapes_Compatible_Dataset

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

        print("[synthia] Found {} images".format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        img_path = os.path.join(self.image_base_path, id.split("/")[-1])
        lbl_path = id

        img, lbl = self.post_process(img_path, lbl_path)
        return img, lbl, self.ids[index]
