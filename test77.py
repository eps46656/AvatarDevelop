import os

import torch
import tqdm

from . import (config, people_snapshot_utils, utils, video_seg_utils,
               vision_utils)

DTYPE = torch.float64
DEVICE = utils.CPU_DEVICE

SUBJECT_NAME = "female-3-casual"
SUBJECT_SHORT_NAME = "f3c"

VIDEO_SEG_DIR = config.DIR / f"video_seg_f3c_2025_0619_1"


def main1():
    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        None,
        DTYPE,
        DEVICE,
    )

    T, C, H, W = subject_data.video.shape

    obj_list = [
        "hair",
        "upper_garment",
        "lower_garment",
    ]

    person_mask = video_seg_utils.read_person_mask(
        VIDEO_SEG_DIR,
    ).read_many()

    obj_mask = {
        obj_name: video_seg_utils.read_obj_mask(
            VIDEO_SEG_DIR,
            obj_name,
            None,
        ).read_many()
        for obj_name in obj_list
    }

    frame_idx = 30

    """
    while True:
        print(f"{frame_idx=}")

        # Write video frame
        vision_utils.show_image(
            "video_frame",
            subject_data.video[frame_idx],
        )

        # Write object masks
        for obj_name, mask in obj_mask.items():
            vision_utils.show_image(
                f"{obj_name}_mask",
                mask[frame_idx],
            )

        # Write person mask
        vision_utils.show_image(
            "person_mask",
            person_mask[frame_idx],
            pause=True
        )

        if input("is ok") == T:
            break

        frame_idx += 1
    """

    vision_utils.write_image(
        config.DIR / "_images" / f"{SUBJECT_SHORT_NAME}_image.png",
        subject_data.video[frame_idx],
    )

    vision_utils.write_image(
        config.DIR / "_images" / f"{SUBJECT_SHORT_NAME}_person_mask.png",
        person_mask[frame_idx],
    )

    for obj_name, mask in obj_mask.items():
        vision_utils.write_image(
            config.DIR / "_images" /
            f"{SUBJECT_SHORT_NAME}_{obj_name}_mask.png",
            mask[frame_idx],
        )


def main2():
    pass


if __name__ == "__main__":
    main1()
