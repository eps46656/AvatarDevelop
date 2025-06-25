

import numpy as np
import torch

from . import (config, m3d_vton_utils, people_snapshot_utils, utils,
               video_seg_utils, vision_utils)

DTYPE = torch.float64
DEVICE = utils.CPU_DEVICE

BODY_SUBJECT_NAME = "female-3-casual"
BODY_SUBJECT_SHORT_NAME = "f3c"

CLOTH_SUBJECT_NAME = "female-4-casual"
CLOTH_SUBJECT_SHORT_NAME = "f4c"

BODY_SUBJECT_VIDEO_SEG_DIR = config.DIR / "video_seg_f3c_2025_0619_1"
CLOTH_SUBJECT_VIDEO_SEG_DIR = config.DIR / "video_seg_f4c_2025_0623_1"

CLOTH_FRAME_IDX = 30

MY_M3D_VTON_DATASET_DIR = config.DIR / \
    f"MyM3DVTONDataset_people_snapshot_{BODY_SUBJECT_SHORT_NAME}_{CLOTH_SUBJECT_SHORT_NAME}"

MY_M3D_VTON_RESULT_DIR = config.DIR / \
    f"MyM3DVTONNResult_people_snapshot_{BODY_SUBJECT_SHORT_NAME}_{CLOTH_SUBJECT_SHORT_NAME}"

OBJ_NAME = "upper_garment"


def get_subject_data(subject_name, video_seg_dir):
    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / subject_name,
        None,
        DTYPE,
        DEVICE,
    )

    mask = video_seg_utils.read_obj_mask(
        video_seg_dir,
        OBJ_NAME,
        None,
    ).read_many()

    return subject_data, mask


def main2():
    body_subject_data, body_cloth_mask = get_subject_data(
        BODY_SUBJECT_NAME, BODY_SUBJECT_VIDEO_SEG_DIR)

    cloth_subject_data, cloth_cloth_mask = get_subject_data(
        CLOTH_SUBJECT_NAME, CLOTH_SUBJECT_VIDEO_SEG_DIR)

    T, C, H, W = body_subject_data.video.shape

    body_parse, _ = vision_utils.read_video(
        config.DIR /
        f"people_snapshot_2d_human_parsing/{BODY_SUBJECT_SHORT_NAME}.avi",
        "GRAY",
        dtype=torch.uint8,
        device=DEVICE,
    )

    body_pose = [
        utils.read_json(
            config.DIR /
            f"people_snapshot_openpose/{BODY_SUBJECT_SHORT_NAME}/video_{t:012d}_keypoints.json"
        )

        for t in range(T)
    ]

    cloth_frame = cloth_subject_data.video[CLOTH_FRAME_IDX]

    cloth_mask_frame = cloth_cloth_mask[CLOTH_FRAME_IDX]

    vision_utils.show_image("cloth_frame", cloth_frame)
    vision_utils.show_image("cloth_mask_frame", cloth_mask_frame, pause=True)

    m3d_vton_utils.make_dataset(
        dataset_dir=MY_M3D_VTON_DATASET_DIR,
        img=body_subject_data.video,
        img_parse=body_parse,
        pose=body_pose,
        cloth=cloth_frame.expand(T, C, H, W),
        cloth_mask=cloth_mask_frame.expand(T, 1, H, W),
    )


if __name__ == "__main__":
    main2()
