import argparse
import glob
import os

import cv2
import numpy as np
from cv2.ximgproc import createSuperpixelLSC
from scipy import stats
from tqdm import tqdm


def superpixel(img, visualize=True, output="sp_tmp.jpg"):
    h, w, _ = img.shape

    superpix = createSuperpixelLSC(img, region_size=10, ratio=0.075)
    superpix.iterate(15)
    superpix.enforceLabelConnectivity(min_element_size=25)

    if visualize:
        print(f"Number of Pixels: {h} * {w} = {h*w}")
        print(f"Number of superpixels: {superpix.getNumberOfSuperpixels()}")

        contour = superpix.getLabelContourMask()
        contour_rgb = cv2.cvtColor(contour, cv2.COLOR_GRAY2BGR)

        img_contour = np.where(contour_rgb == 0, img, contour_rgb)
        cv2.imwrite(output, img_contour)
        return contour_rgb
    else:
        mask = superpix.getLabels()
        return mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output", default="sp_tmp")
    parser.add_argument("--mode", default="visualize")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "visualize":
        img_path = args.input
        img = cv2.imread(img_path)

        output = f"{args.output}.jpg"
        superpixel(img, output=output)
        print(f"File saved to {output}")
    elif args.mode == "generate":
        output_dir = args.output
        os.makedirs(output_dir)

        images = glob.glob(f"{args.input}/**/*.png", recursive=True)
        print(f"Total {len(images)} images to process.")
        for image in tqdm(images):
            img_id = os.path.basename(image).split(".")[0]

            img = cv2.imread(image)
            mask = superpixel(img, visualize=False)
            mask.dump(os.path.join(output_dir, f"{img_id}.dat"))

        print(f"Superpixel masks saved to {output_dir}")
    elif args.mode == "inspect":
        sps = glob.glob(f"{args.input}/**/*.dat", recursive=True)
        counts = []
        for sp in tqdm(sps):
            mask = np.load(sp, allow_pickle=True)
            label_count = len(np.unique(mask))
            counts.append(label_count)
        stats.describe(label_count)
