

import subprocess

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
    subprocess.run(
        [
            "python",
            "test.py",
            "--model", "MTM",
            "--name", "MTM",
            "--dataroot", str(MY_M3D_VTON_DATASET_DIR.resolve()),
            "--datamode", "aligned",
            "--datalist", "test_pairs",
            # "--results_dir", str(MY_M3D_VTON_RESULT_DIR.resolve()),
        ],
        cwd=config.M3D_VTON_DIR,
        check=True,
    )

    subprocess.run(
        [
            "python",
            "test.py",
            "--model", "DRM",
            "--name", "DRM",
            "--dataroot", str(MY_M3D_VTON_DATASET_DIR.resolve()),
            "--datalist", "test_pairs",
            # "--results_dir", str(MY_M3D_VTON_RESULT_DIR.resolve()),
        ],
        cwd=config.M3D_VTON_DIR,
        check=True,
    )

    subprocess.run(
        [
            "python",
            "test.py",
            "--model", "TFM",
            "--name", "TFM",
            "--dataroot", str(MY_M3D_VTON_DATASET_DIR.resolve()),
            "--datalist", "test_pairs",
            # "--results_dir", str(MY_M3D_VTON_RESULT_DIR.resolve()),
        ],
        cwd=config.M3D_VTON_DIR,
        check=True,
    )


if __name__ == "__main__":
    main2()
