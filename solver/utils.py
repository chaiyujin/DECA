import os

import cv2
import numpy as np
import toml


def read_image(fpath, A):
    img = cv2.imread(fpath)[..., [2, 1, 0]]
    mask = cv2.imread(os.path.splitext(fpath)[0] + "_mask.png")[..., 0]
    with open(os.path.splitext(fpath)[0] + "_lmks.toml") as fp:
        lmks = toml.load(fp)["points"]
        lmks = np.asarray(lmks, dtype=np.float32)
        lmks[:, 0] /= img.shape[1]
        lmks[:, 1] /= img.shape[0]
        lmks = lmks * 2 - 1
    img = (cv2.resize(img, (A, A)) / 255.0).astype(np.float32)
    mask = (cv2.resize(mask, (A, A)) / 255.0).astype(np.float32)
    return img, mask, lmks
