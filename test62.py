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

from . import (config, gom_avatar_utils, mesh_utils, people_snapshot_utils,
               sdf_utils, smplx_utils, training_utils, utils, pipeline_utils)

DTYPE = torch.float64
DEVICE = torch.device("cuda")

SDF_MODULE_DIR = config.DIR / "sdf_module_2025_0609_1"


ALPHA_SIGNED_DIST = (lambda epoch:  1e6)
ALPHA_EIKONAL = (lambda epoch: 1.0)

BATCH_SIZE = 64
BATCHES_CNT = 64


LR = (lambda epoch: 1e-3 * (0.1 ** (epoch / 256)))


def main1():
    model_data = pipeline_utils.load_smplx_model_data(
        pipeline_utils.smpl_female_model_info,
        dtype=DTYPE,
        device=DEVICE,
    )

    sdf_module_trainer = pipeline_utils.load_sdf_module_trainer(
        sdf_module_dir=SDF_MODULE_DIR,
        mesh_data=mesh_utils.MeshData(
            mesh_graph=model_data.mesh_graph,
            vert_pos=model_data.vert_pos,
        ),
        dtype=DTYPE,
        device=DEVICE,
    )

    sdf_module_trainer.enter_cli()


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(False, False):
        main1()

    print("ok")
