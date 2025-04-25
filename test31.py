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

from . import (avatar_utils, camera_utils, config, dataset_utils,
               gaussian_utils, gom_avatar_training_utils,
               gom_utils, kernel_splatting_utils, people_snapshot_utils, rendering_utils, smplx_utils,
               texture_utils, training_utils, transform_utils, utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")

PROJ_DIR = DIR / "train_2025_0422_3"

VERT_GRAD_NORM_THRESHOLD = 1e-3

ALPHA_RGB = 1.0
ALPHA_LAP_SMOOTHNESS = 1000.0
ALPHA_NOR_SIM = 10.0
ALPHA_EDGE_VAR = 1.0
ALPHA_COLOR_DIFF = 1.0
ALPHA_GP_SCALE_DIFF = 1.0

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


def create_optimizer(param_groups):
    return torch.optim.Adam(
        param_groups,
        lr=LR,
        betas=(0.5, 0.5),
    )


def create_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=pow(0.1, 1/4),
        patience=5,
        threshold=0.05,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-7,
    )


def load_trainer():
    subject_data = read_subject()

    # ---

    dataset = gom_utils.Dataset(gom_utils.Sample(
        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,
        img=subject_data.video,
        mask=subject_data.mask,
        blending_param=subject_data.blending_param,
    )).to(DEVICE)

    # ---

    smplx_model_builder = smplx_utils.DeformableModelBuilder(
        model_data=subject_data.model_data,
    ).to(DEVICE)

    smplx_model_builder.unfreeze()

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    module = gom_utils.Module(
        avatar_blender=smplx_model_blender,
        color_channels_cnt=3,
    ).to(DEVICE).train()

    # ---

    training_core = gom_avatar_training_utils.TrainingCore(
        config=gom_avatar_training_utils.Config(
            proj_dir=PROJ_DIR,
            device=DEVICE,
            batch_size=BATCH_SIZE,

            lr=LR,

            vert_grad_norm_threshold=VERT_GRAD_NORM_THRESHOLD,

            alpha_rgb=ALPHA_RGB,
            alpha_lap_smoothness=ALPHA_LAP_SMOOTHNESS,
            alpha_nor_sim=ALPHA_NOR_SIM,
            alpha_edge_var=ALPHA_EDGE_VAR,
            alpha_color_diff=ALPHA_COLOR_DIFF,
            alpha_gp_scale_diff=ALPHA_GP_SCALE_DIFF,
        ),
        module=module,
        dataset=dataset,
        optimizer_factory=create_optimizer,
        scheduler_factory=create_scheduler,
    )

    # ---

    trainer = training_utils.Trainer(
        proj_dir=PROJ_DIR,
        device=DEVICE,
    )

    trainer.training_core = training_core

    return trainer


def main1():
    trainer = load_trainer()

    trainer.enter_cli()


def main2():
    trainer = load_trainer()

    trainer.load_latest()

    training_core: gom_avatar_training_utils.TrainingCore = \
        trainer.training_core

    model_blender: smplx_utils.ModelBlender = \
        training_core.module.avatar_blender

    module_builder: smplx_utils.DeformableModelBuilder = \
        model_blender.model_builder

    module_builder.model_data.show()


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True, True):
        main1()

    print("ok")
