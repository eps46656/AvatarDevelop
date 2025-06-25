import os

import torch
import tqdm

from . import (config, dd_human_parsing_utils, people_snapshot_utils, utils,
               vision_utils)

DTYPE = torch.float64
DEVICE = utils.CPU_DEVICE

SUBJECT_NAME = "female-4-casual"
SUBJECT_SHORT_NAME = "f4c"

DD_HUMAN_PARSING_DIR = config.DIR / \
    f"people_snapshot_2d_human_parsing/{SUBJECT_SHORT_NAME}"


def main1():
    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        None,
        DTYPE,
        DEVICE,
    )

    T, C, H, W = subject_data.video.shape

    with vision_utils.VideoWriter(
        config.DIR /
            f"people_snapshot_2d_human_parsing/{SUBJECT_SHORT_NAME}.avi",
        H,
        W,
        "RGB",
        subject_data.fps,
    ) as video_writer:
        for result in dd_human_parsing_utils.predict(
                subject_data.video[t] for t in range(T)):
            video_writer.write(result)


if __name__ == "__main__":
    main1()
