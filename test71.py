

import numpy as np
import torch

from . import (cloth3d_utils, config, people_snapshot_utils,
               stable_viton_utils, transform_utils, utils, video_seg_utils,
               vision_utils)

DTYPE = torch.float64
DEVICE = utils.CPU_DEVICE

BODY_SUBJECT_NAME = "female-3-casual"
BODY_SUBJECT_SHORT_NAME = "f3c"

CLOTH_SUBJECT_NAME = "female-4-casual"
CLOTH_SUBJECT_SHORT_NAME = "f4c"

BODY_SUBJECT_VIDEO_SEG_DIR = config.DIR / "video_seg_f3c_2025_0619_1"
CLOTH_SUBJECT_VIDEO_SEG_DIR = config.DIR / "video_seg_f3c_2025_0619_1"

MY_VITON_DATASET_DIR = config.DIR / \
    f"MyStableVITONDataset_people_snapshot_{BODY_SUBJECT_SHORT_NAME}_{CLOTH_SUBJECT_SHORT_NAME}"

MY_VITON_RESULT_DIR = config.DIR / \
    f"MyStableVITONResult_people_snapshot_{BODY_SUBJECT_SHORT_NAME}_{CLOTH_SUBJECT_SHORT_NAME}"

OBJ_NAME = "upper_garment"


"""

python inference.py --config_path ./configs/VITONHD.yaml  --model_load_path ./ckpts/VITONHD.ckpt --batch_size 4  --data_root_dir ../MyStableVITONDataset_people_snapshot_f3c --repaint  --save_dir ../MyStableVITONResult_people_snapshot_f3c

"""


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

    body_img_densepose, _ = vision_utils.read_video(
        config.DIR /
        f"people_snapshot_densepose/{BODY_SUBJECT_SHORT_NAME}.avi",
        "RGB",
        device=DEVICE,
    )

    cloth_frame_idx = 30

    cloth_frame = cloth_subject_data.video[cloth_frame_idx]

    cloth_mask_frame = cloth_cloth_mask[cloth_frame_idx]

    C, H, W = body_subject_data.video.shape[-3:]

    stable_viton_utils.make_dataset(
        dataset_dir=MY_VITON_DATASET_DIR,
        img=body_subject_data.video,
        img_mask=body_cloth_mask,
        img_densepose=body_img_densepose,
        cloth=cloth_frame,
        cloth_mask=cloth_mask_frame,
    )

    def str_path(path):
        return f"\"{path.resolve()}\""

    cmd = [
        "python",
        "inference.py",
        "--config_path", "./configs/VITONHD.yaml",
        "--model_load_path", "./ckpts/VITONHD.ckpt",
        "--batch_size", "4",
        "--data_root_dir", str_path(MY_VITON_DATASET_DIR),
        "--repaint",
        "--save_dir", str_path(MY_VITON_RESULT_DIR),
        "--img_H", str(H),
        "--img_W", str(W),
    ]

    print()
    print(f"command: {" ".join(cmd)}")
    print()


if __name__ == "__main__":
    main2()
