import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from decalib.utils import util

from .config import get_default_config
from .solver import DECADecoder, Solver
from .utils import read_image
from .solve_iden import solve_identity
from .solve_video import solve_video

torch.backends.cudnn.benchmark = True

# default settings
cfg = get_default_config()

# solve identity
cfg.train.lr = 1e-2
solve_identity('', "TestSamples/obama/results", cfg, "cuda:0")
quit()

cfg.train.lr = 5e-3
solve_video(
    # "/media/chaiyujin/FE6C78966C784B81/Linux/Home/assets/CelebTalk/ProcessTasks/m000_obama/clips_cropped_aligned/trn-000-frames",
    # "/media/chaiyujin/FE6C78966C784B81/Linux/Home/assets/CelebTalk/ProcessTasks/m000_obama/clips_cropped_aligned/trn-000-lmks-ibug-68.toml",
    "./TestSamples/obama/trn-000/trn-000-frames",
    "./TestSamples/obama/trn-000/trn-000-lmks-ibug-68.toml",
    cfg, "cuda:0"
)
