import copy
import itertools
import math
import pathlib
import time
import typing

import einops
import torch
import tqdm
from beartype import beartype

from . import (config, gom_avatar_utils, people_snapshot_utils, smplx_utils,
               training_utils, utils)

DEVICE = torch.device("cuda")

PROJ_DIR = config.DIR / "train_2025_0514_2"

VERT_GRAD_NORM_THRESHOLD = 1e-3

LAP_DIFF_CLAMP_NORM = 10e-3

ALPHA_IMG_DIFF = 1.0
ALPHA_LAP_DIFF = 50.0
ALPHA_NOR_SIM = 2.0
ALPHA_EDGE_VAR = 0.0
ALPHA_GP_COLOR_DIFF = 10.0
ALPHA_GP_SCALE_DIFF = 10.0

BATCH_SIZE = 4

SUBJECT_NAME = "female-1-casual"

LR = 1e-4


def read_subject(model_data: typing.Optional[smplx_utils.ModelData] = None):
    if model_data is None:
        if "female" in SUBJECT_NAME:
            model_data_path = config.SMPL_FEMALE_MODEL_PATH
        else:
            model_data_path = config.SMPL_MALE_MODEL_PATH

        model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=model_data_path,
            model_config=smplx_utils.smpl_model_config,
            dtype=utils.FLOAT,
            device=DEVICE,
        )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME, model_data, DEVICE)

    return subject_data


def load_trainer():
    subject_data = read_subject()

    # ---

    dataset = gom_avatar_utils.Dataset(gom_avatar_utils.Sample(
        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,
        img=subject_data.video,
        mask=subject_data.mask,
        blending_param=subject_data.blending_param,
    )).to(DEVICE)

    # ---

    smplx_model_builder = smplx_utils.DeformableModelBuilder(
        temp_model_data=subject_data.model_data,
        model_data=subject_data.model_data,
    ).to(DEVICE)

    smplx_model_builder.unfreeze()

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    module = gom_avatar_utils.Module(
        avatar_blender=smplx_model_blender,
        color_channels_cnt=3,
    ).to(DEVICE).train()

    # ---

    trainer_core = gom_avatar_utils.TrainerCore(
        config=gom_avatar_utils.TrainerCoreConfig(
            proj_dir=PROJ_DIR,
            device=DEVICE,
            batch_size=BATCH_SIZE,

            lr=LR,
            betas=(0.5, 0.5),
            gamma=0.95,

            vert_grad_norm_threshold=VERT_GRAD_NORM_THRESHOLD,

            lap_diff_clamp_norm=LAP_DIFF_CLAMP_NORM,

            alpha_img_diff=ALPHA_IMG_DIFF,
            alpha_lap_diff=ALPHA_LAP_DIFF,
            alpha_nor_sim=ALPHA_NOR_SIM,
            alpha_edge_var=ALPHA_EDGE_VAR,
            alpha_gp_color_diff=ALPHA_GP_COLOR_DIFF,
            alpha_gp_scale_diff=ALPHA_GP_SCALE_DIFF,
        ),
        module=module,
        dataset=dataset,
    )

    # ---

    trainer = training_utils.Trainer(
        proj_dir=PROJ_DIR,
        trainer_core=trainer_core,
    )

    return subject_data, trainer


def main1():
    subject_data, trainer = load_trainer()

    trainer.enter_cli()


def main2():
    subject_data, trainer = load_trainer()

    trainer.load_latest()

    trainer_core: gom_avatar_utils.TrainerCore = trainer.trainer_core

    model_blender: smplx_utils.ModelBlender = \
        trainer_core.module.avatar_blender

    module_builder: smplx_utils.DeformableModelBuilder = \
        model_blender.model_builder

    module_builder.model_data.show()


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True, True):
        main1()

    print("ok")
