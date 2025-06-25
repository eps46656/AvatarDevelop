import dataclasses
import math
import typing

import torch

from . import config, utils, video_seg_utils, vision_utils

DTYPE = torch.float64
DEVICE = utils.CUDA_DEVICE

SUBJECT_NAME = "female-4-casual"


VIDEO_SEG_DIR = config.DIR / "video_seg_f4c_2025_0623_1"


def main1():
    video_path = config.DIR / \
        f"people_snapshot_public/{SUBJECT_NAME}/{SUBJECT_NAME}.mp4"

    src_video = vision_utils.VideoReader(
        path=video_path,
        color_type="RGB",
    )

    video_seg_utils.segment(
        src_video,
        VIDEO_SEG_DIR,
        {
            "hair":
                "the hair",

            "upper_garment":
                "the upper garment",

            "lower_garment":
                "the lower garment",

            "footwear":
                "the footwear",
        },
        en_obj_mask=True,
        en_refined_obj_mask=False,
        en_skin_mask=True,
        en_refined_skin_mask=False,
    )


if __name__ == "__main__":
    main1()
