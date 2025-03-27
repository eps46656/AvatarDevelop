import pathlib

import lang_sam
import numpy as np
import torch
import torchvision
from beartype import beartype

from . import lang_sam_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


def main1():
    config = DIR / "sam2_data/sam2.1_hiera_l.yaml"
    checkpoint = DIR / "sam2_data/sam2.1_hiera_large.pt"

    video_path = DIR / "people_snapshot_public/female-3-sport/female-3-sport.mp4"

    video, fps = utils.read_video(video_path)

    person_mask = utils.image_normalize(
        utils.read_video(DIR / "out_video <the main person>.mp4")[0])

    bg = torch.ones(video.shape[1:]) * 255

    T = video.shape[0]

    prompt = "upper garment of main person"

    mask = LangSAMPredict(
        ,
        [prompt] * T
    )

    mask = lang_sam_utils.Predict(
        lang_sam_utils.SAMType.LARGE,
        img=[
            bg * (1 - person_mask[frame_i]) +
            video[frame_i] * person_mask[frame_i]
            for frame_i in range(T)
        ],
        prompt=[prompt] * T,
        mask_strategy=lang_sam_utils.MaskStrategy.MIN_AREA,
        batch_size=1,
    )

    out_video = torch.empty_like(video)

    for frame_i in range(T):
        out_video[frame_i] = mask[frame_i]

    utils.write_video(
        DIR / f"out_video_{prompt.replace(" ", "_")}.mp4",
        utils.image_denormalize(out_video).expand_as(video),
        fps,
    )


if __name__ == "__main__":
    main1()
