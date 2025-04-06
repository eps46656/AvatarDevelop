import pathlib

import torch
import h5py
import pickle
import dataclasses

from . import people_snapshot_utils, utils, smplx_utils, config

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")


def main2():
    smpl_model_data_path_dict = {
        "male": config.SMPL_MALE_MODEL_PATH,
        "female": config.SMPL_FEMALE_MODEL_PATH,
        "neutral": config.SMPL_NEUTRAL_MODEL_PATH,
    }

    smplx_model_data_path_dict = {
        "male": config.SMPLX_MALE_MODEL_PATH,
        "female": config.SMPLX_FEMALE_MODEL_PATH,
        "neutral": config.SMPLX_NEUTRAL_MODEL_PATH,
    }

    body_shapes_cnt = 10
    expr_shapes_cnt = 0
    body_joints_cnt = 24
    jaw_joints_cnt = 0
    eye_joints_cnt = 0
    hand_joints_cnt = 0

    model_data_dict = {
        key: smplx_utils.ReadModelData(
            model_data_path=value,
            body_shapes_cnt=body_shapes_cnt,
            expr_shapes_cnt=expr_shapes_cnt,
            body_joints_cnt=body_joints_cnt,
            jaw_joints_cnt=jaw_joints_cnt,
            eye_joints_cnt=eye_joints_cnt,
            hand_joints_cnt=hand_joints_cnt,
            device=DEVICE,
        )
        for key, value in smpl_model_data_path_dict.items()
    }

    people_snapshot_dir = DIR / "people_snapshot_public"

    subject_name = "female-1-casual"

    subject_dir = people_snapshot_dir / subject_name

    subject_data = people_snapshot_utils.read_subject(
        subject_dir=subject_dir,
        model_data_dict=model_data_dict,
        device=DEVICE,
    )

    print(f"{subject_data.mask.shape}")

    utils.write_image(
        DIR / "mask.jpg",
        subject_data.mask[0],
    )


if __name__ == "__main__":
    main2()
