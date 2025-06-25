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
               training_utils, utils, vision_utils)

DEVICE = torch.device("cuda")

SUBJECT_NAME = "female-3-casual"
SUBJECT_SHORT_NAME = "f3c"

GOM_AVATAR_DIR = config.DIR / f"gom_avatar_{SUBJECT_SHORT_NAME}_2025_0624_1"


SILHOUETTE_RATIO_SIGMA = [
    0.010,
    0.006,
    # 0.003,
]

SILHOUETTE_OPACITY = [
    0.001,
    0.001,
    # 0.001,
]

DILATED_MASK_DIR = config.DIR / \
    f"people_snapshot_dilated_mask/{SUBJECT_SHORT_NAME}"

VERT_GRAD_NORM_THRESHOLD = 1e6

ALPHA_IMG_DIFF = (lambda epoch: 1.0)

ALPHA_GP_MASK_DIFF = [
    (lambda epoch: 1.0),
    (lambda epoch: 1.0),
]

ALPHA_LAP_DIFF = (lambda epoch: 1000.0)
ALPHA_NOR_SIM = (lambda epoch: 0.0)  # (lambda epoch: 30.0)
ALPHA_EDGE_VAR = (lambda epoch: 10.0)  # (lambda epoch: 30.0)


BATCH_SIZE = 8


LR = (lambda epoch: 5e-3 * (0.95 ** epoch))


def read_subject(model_data: typing.Optional[smplx_utils.ModelData] = None):
    if model_data is None:
        model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=torch.float32,
            device=utils.CPU_DEVICE,
        )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        model_data,
        torch.float32,
        utils.CPU_DEVICE,
    )

    return subject_data


def load_trainer():
    subject_data = read_subject()

    dilated_mask = [
        utils.read_pickle(
            DILATED_MASK_DIR / f"dilated_mask_{sigma_idx}.pickle"
        ).to(utils.CPU_DEVICE)
        for sigma_idx in range(len(SILHOUETTE_RATIO_SIGMA))
    ]

    # ---

    dataset = gom_avatar_utils.Dataset(gom_avatar_utils.Sample(
        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,
        img=subject_data.video,

        mask=subject_data.mask,
        dilated_mask=dilated_mask,

        blending_param=subject_data.blending_param,
    ))

    # ---

    smplx_model_builder = smplx_utils.DeformableModelBuilder(
        temp_model_data=smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=torch.float64,
            device=DEVICE,
        ),
        model_data=subject_data.model_data.to(DEVICE),
    )

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    module = gom_avatar_utils.Module(
        avatar_blender=smplx_model_blender,
        color_channels_cnt=3,
        dtype=torch.float32,
        device=DEVICE,
    ).train()

    # ---

    silhouette_sigma = [
        vision_utils.get_sigma(
            subject_data.video.shape[-2], subject_data.video.shape[-1], x)
        for x in SILHOUETTE_RATIO_SIGMA
    ]

    print(f"{silhouette_sigma=}")

    trainer_core = gom_avatar_utils.TrainerCore(
        config=gom_avatar_utils.TrainerCoreConfig(
            proj_dir=GOM_AVATAR_DIR,
            device=DEVICE,
            batch_size=BATCH_SIZE,

            lr=LR,
            betas=(0.9, 0.999),

            silhouette_sigma=silhouette_sigma,
            silhouette_opacity=SILHOUETTE_OPACITY,

            vert_grad_norm_threshold=VERT_GRAD_NORM_THRESHOLD,

            alpha_img_diff=ALPHA_IMG_DIFF,
            alpha_gp_mask_diff=ALPHA_GP_MASK_DIFF,
            alpha_lap_diff=ALPHA_LAP_DIFF,
            alpha_nor_sim=ALPHA_NOR_SIM,
            alpha_edge_var=ALPHA_EDGE_VAR,
        ),
        module=module,
        dataset=dataset,
    )

    # ---

    trainer = training_utils.Trainer(
        proj_dir=GOM_AVATAR_DIR,
        trainer_core=trainer_core,
    )

    return subject_data, trainer


def main1():
    subject_data, trainer = load_trainer()

    utils.mem_clear()

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


def main3():
    subject_data = read_subject()

    T, C, H, W = subject_data.mask.shape

    for sigma_idx in range(len(SILHOUETTE_RATIO_SIGMA)):
        print(f"{sigma_idx=}")

        dilated_mask = list()

        kernel = gom_avatar_utils.make_dilate_kernel(
            img_h=H,
            img_w=W,
            ratio_sigma=SILHOUETTE_RATIO_SIGMA[sigma_idx],
            opacity=SILHOUETTE_OPACITY[sigma_idx],
        )
        # [1, 1, K, K]

        batch_size = 16

        for t in tqdm.tqdm(range(0, T, batch_size)):
            t_end = min(t + batch_size, T)

            cur_mask = subject_data.mask[t:t_end].to(
                utils.CUDA_DEVICE, torch.float64)
            # [B, 1, H, W]

            print(f"{cur_mask.shape=}")

            cur_dilated_mask = gom_avatar_utils.make_dilated_mask(
                cur_mask, kernel
            )
            # [B, 1, H, W]

            print(f"{cur_dilated_mask.shape=}")
            print(f"{cur_dilated_mask.min()=}")
            print(f"{cur_dilated_mask.max()=}")

            vision_utils.show_image(
                "cur_dilated_mask",
                utils.rct(cur_dilated_mask[0] * 255, dtype=torch.uint8),
            )

            for _ in cur_dilated_mask:
                dilated_mask.append(_)

        utils.write_pickle(
            GOM_AVATAR_DIR / f"dilated_mask_{sigma_idx}.pickle",
            torch.stack(dilated_mask, 0),
        )

    # ---

    print("done")


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(False, False):
        main1()

    print("ok")
