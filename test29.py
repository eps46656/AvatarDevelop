import pathlib
import time
import typing

import torch
import tqdm
from beartype import beartype

from . import config, segment_utils, smplx_utils, utils, video_utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = utils.CPU_DEVICE


def main1():
    video_path = DIR / "people_snapshot_public/female-1-casual/female-1-casual.mp4"

    video, fps = video_utils.read_video(
        video_path, video_utils.ColorType.RGB)

    segment_utils.segment(
        imgs=[i for i in video],  # [T, C, H, W]
        fps=fps,
        out_dir=DIR / f"segment_2025_0412_1",
    )


if __name__ == "__main__":
    main1()

    print("ok")
