import pathlib
import time
import typing

import torch
import tqdm
from beartype import beartype

from . import config, smplx_utils, utils, mesh_utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = utils.CPU_DEVICE


def main1():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPLX_NEUTRAL_MODEL_PATH,
        model_config=smplx_utils.smplx_model_config,
        device=DEVICE,
    )

    model_blender = smplx_utils.ModelBlender(
        smplx_utils.StaticModelBuilder(model_data))

    result: smplx_utils.Model = model_blender(smplx_utils.BlendingParam())

    vert_pos = result.vert_pos
    # [V, 3]

    V = utils.check_shapes(vert_pos, (-1, 3))

    min_x, min_y, min_z = vert_pos.min(0)[0]
    max_x, max_y, max_z = vert_pos.max(0)[0]

    print(f"{min_x=}, {min_y=}, {min_z=}")
    print(f"{max_x=}, {max_y=}, {max_z=}")

    B = 100

    k = 0.1

    p = torch.empty((B, 3), dtype=utils.FLOAT)
    p[:, 0].uniform_(-k, k)
    p[:, 1].uniform_(-k, k)
    p[:, 2].uniform_(-k, k)

    mesh_data = mesh_utils.MeshData(
        model_data.mesh_graph,
        vert_pos,
    )

    with utils.Timer():
        sd_a = mesh_data.calc_signed_dist(p)

    with utils.Timer():
        sd_b = mesh_data.calc_signed_dist_trimesh(p)

    # print(f"{sd_a=}")
    # print(f"{sd_b=}")

    max_err = (sd_a - sd_b).abs().max()

    print(f"{max_err=}")


if __name__ == "__main__":
    main1()

    print("ok")
