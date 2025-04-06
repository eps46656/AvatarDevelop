import pathlib
import time
import typing

import torch
import tqdm
from beartype import beartype

from . import config, segment_utils, smplx_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = utils.CPU_DEVICE


def main1():
    video_path = DIR / "people_snapshot_public/female-1-casual/female-1-casual.mp4"

    video, fps = utils.read_video(video_path)

    segment_utils.segment(
        imgs=video,  # [T, C, H, W]
        out_dir=DIR / f"segment_2025_0406_01",
        out_fps=fps,
    )


if __name__ == "__main__":
    main1()

    print("ok")
