import dataclasses
import math
import typing

import beartype
import einops
import torch
import torchvision
import tqdm
from beartype import beartype

from . import config, utils, video_seg_utils, vision_utils

DTYPE = torch.float64
DEVICE = utils.CUDA_DEVICE

SUBJECT_NAME = "female-3-casual"


VIDEO_SEG_DIR = config.DIR / "video_seg_2025_0520_2"

MESH_SEG_DIR = config.DIR / "mesh_seg_2025_0514_1"

TEX_AVATAR_DIR = config.DIR / "tex_avatar_2025_0520_1"


def main1():
    video_path = config.DIR / \
        f"people_snapshot_public/{SUBJECT_NAME}/{SUBJECT_NAME}.mp4"

    src_video = vision_utils.VideoReader(
        path=video_path,
        color_type=vision_utils.ColorType.RGB,
    )

    video_seg_utils.segment(
        src_video,
        VIDEO_SEG_DIR,
        {
            "HAIR":
                "The hair of the main person in the video",

            "UPPER_GARMENT":
                "The upper garment of the main person in the video",

            "LOWER_GARMENT":
                "The lower garment of the main person in the video",
        }
    )


if __name__ == "__main__":
    main1()
