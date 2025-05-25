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

from . import (config, gom_avatar_utils, people_snapshot_utils, sdf_utils,
               smplx_utils, training_utils, utils)

DTYPE = torch.float64
DEVICE = torch.device("cuda")

SDF_MODULE_DIR = config.DIR / "sdf_module_2025_0522_1"


ALPHA_SIGNED_DIST = 1.0
ALPHA_EIKONAL = 1.0

BATCH_SIZE = 64
BATCHES_CNT = 32


LR = 1e-3


def load_trainer():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_NEUTRAL_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=DTYPE,
        device=DEVICE,
    )

    # ---

    dataset = sdf_utils.Dataset(
        mesh_graph=model_data.mesh_graph,
        vert_pos=model_data.vert_pos,

        std=50e-3,

        shape=torch.Size((BATCH_SIZE * BATCHES_CNT,)),
    )

    # ---

    module = sdf_utils.Module(
        range_min=(-2.0, -2.0, -2.0),
        range_max=(+2.0, +2.0, +2.0),
        dtype=DTYPE,
        device=DEVICE,
    ).train()

    # ---

    trainer_core = sdf_utils.TrainerCore(
        config=sdf_utils.TrainerCoreConfig(
            proj_dir=SDF_MODULE_DIR,
            device=DEVICE,

            batch_size=BATCH_SIZE,

            lr=LR,
            betas=(0.9, 0.99),
            gamma=0.1 ** (1 / 64),

            alpha_signed_dist=ALPHA_SIGNED_DIST,
            alpha_eikonal=ALPHA_EIKONAL,
        ),
        module=module,
        dataset=dataset,
    )

    # ---

    trainer = training_utils.Trainer(
        proj_dir=SDF_MODULE_DIR,
        trainer_core=trainer_core,
    )

    return trainer


def main1():
    trainer = load_trainer()

    trainer.enter_cli()


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(False, False):
        main1()

    print("ok")
