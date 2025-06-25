import copy
import itertools
import math
import pathlib
import time
import typing

import einops
import numpy as np
import torch
import tqdm
import trimesh
from beartype import beartype

from . import (avatar_utils, camera_utils, config, gom_avatar_utils,
               mesh_layer_utils, people_snapshot_utils, rendering_utils,
               sdf_utils, smplx_utils, tex_avatar_utils, training_utils,
               transform_utils, utils, vision_utils)

DTYPE = torch.float64
DEVICE = torch.device("cuda")


MODEL_DATA_PATH = config.DIR / \
    "gom_avatar_{SUBJECT_SHORT_NAME}_2025_0531_1/model_data.pkl"

TEX_AVATAR_DIR = config.DIR / "tex_avatar_2025_0531_1"


SUBJECT_NAME = "female-3-casual"


def read_subject(model_data: typing.Optional[smplx_utils.ModelData] = None):
    if model_data is None:
        model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=DTYPE,
            device=DEVICE,
        )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        model_data,
        DTYPE,
        utils.CPU_DEVICE,
    )

    return subject_data


def main1():
    subject_data = read_subject()

    model_data = smplx_utils.ModelData.from_state_dict(
        state_dict=utils.read_pickle(MODEL_DATA_PATH),
        dtype=DTYPE,
        device=DEVICE,
    )

    model_data = model_data.remesh(
        utils.ArgPack(
            epochs_cnt=10,
            lr=1e-3,
            betas=(0.5, 0.5),

            alpha_lap_diff=1000.0,
            alpha_edge_var=1.0,
        )
    )

    blending_coeff_field = smplx_utils.BlendingCoeffField.from_model_data(
        smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=DTYPE,
            device=DEVICE,
        )
    )

    model_data = blending_coeff_field.query_model_data(model_data)

    model_data.show()

    utils.write_pickle(
        TEX_AVATAR_DIR / f"remeshed_model_data.pkl",
        model_data.state_dict()
    )

    smplx_model_builder = smplx_utils.StaticModelBuilder(
        model_data=model_data,
    )

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    init_tex = torch.rand(
        (3, 1000, 1000), dtype=DTYPE, device=DEVICE
    )

    tex_avatar_trainer = training_utils.Trainer(
        TEX_AVATAR_DIR,
        tex_avatar_utils.TrainerCore(
            config=tex_avatar_utils.TrainerCoreConfig(
                proj_dir=TEX_AVATAR_DIR,
                device=DEVICE,
                batch_size=8,

                lr=1e-1,
                betas=(0.5, 0.5),
                gamma=0.95,
            ),

            avatar_blender=smplx_model_blender,

            dataset=tex_avatar_utils.Dataset(
                tex_avatar_utils.Sample(
                    camera_config=subject_data.camera_config,
                    camera_transform=subject_data.camera_transform,

                    img=subject_data.video,

                    mask=subject_data.mask,

                    blending_param=subject_data.blending_param,
                )
            ),

            init_tex=init_tex,
        )
    )

    tex_avatar_trainer.enter_cli()


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True, True):
        main1()

    print("ok")
