import typing

import torch
import dataclasses
from beartype import beartype

from . import config, smplx_utils, utils, transform_utils, camera_utils

DEVICE = torch.device("cpu")


def main1():

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

    model_data_dict: dict[str, smplx_utils.ModelData] = {
        key: smplx_utils.ModelData.from_file(
            model_data_path=value,
            model_config=smplx_utils.smpl_model_config,
            device=DEVICE,
        )
        for key, value in smpl_model_data_path_dict.items()
    }

    model_data = model_data_dict["female"]

    vert_pos = model_data.vert_pos
    # [V, 3]

    tex_vert_pos = model_data.tex_vert_pos
    # [TV, 2]

    V, TV = -1, -2

    V, TV = utils.check_shapes(
        vert_pos, (V, 3),
        tex_vert_pos, (TV, 2),
    )

    tex_h = 1000
    tex_w = 1000

    tex_vert_pos_3d = torch.empty(
        (TV, 3),
        dtype=tex_vert_pos.dtype,
        device=tex_vert_pos.device,
    )

    tex_vert_pos_3d[:, 0] = (tex_vert_pos[:, 0] - 0.5) * (tex_w / 2)
    tex_vert_pos_3d[:, 1] = (tex_vert_pos[:, 1] - 0.5) * (tex_h / 2)
    tex_vert_pos_3d[:, 2] = 0


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True, True):
        main1()
