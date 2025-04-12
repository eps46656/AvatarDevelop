import pathlib
import time

import numpy as np
import torch
import torchvision
from beartype import beartype

from . import lang_sam_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


def main1():
    video_path = DIR / "people_snapshot_public/female-3-sport/female-3-sport.mp4"

    video, fps = utils.read_video(video_path)

    person_mask = utils.read_video(
        DIR / "out_video <the main person>.mp4")[0].mean(1, keepdim=True)

    bg = torch.ones(video.shape[1:])

    T = video.shape[0]

    object_type = "top"

    prompt = f"the {object_type} of main person"

    mask = lang_sam_utils.predict(
        lang_sam_utils.SAMType.LARGE,
        img=[
            bg * (1 - person_mask[frame_i]) +
            video[frame_i] * person_mask[frame_i]
            for frame_i in range(T)
        ],
        prompts=[prompt] * T,
        mask_strategy=lang_sam_utils.MaskStrategy.MIN_AREA,
        batch_size=1,
    )

    out_video = torch.empty_like(video)

    for frame_i in range(T):
        out_video[frame_i] = mask[frame_i]

    timestamp = int(time.time())

    utils.write_video(
        DIR / f"out_video_{timestamp}_{prompt.replace(" ", "_")}.mp4",
        out_video,
        fps,
    )


if __name__ == "__main__":
    with utils.Timer():
        main1()
