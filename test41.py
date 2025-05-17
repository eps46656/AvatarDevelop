
import dataclasses
import math
import os
import pathlib
import typing

import matplotlib.pyplot as plt
import torch
import torchrbf
import tqdm
from beartype import beartype

from . import (config, rbf_utils, segment_utils, smplx_utils, utils,
               vision_utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")

SUBJECT_NAME = "female-3-casual"


def print_tensor(x: torch.Tensor):
    print(f"[")

    for val in x.flatten().tolist():
        print(f"{val:+.6e}f", end=", ")

    print(f"]")


def get_subdirs(path: os.PathLike) -> list[os.PathLike]:
    return [name for name in utils.to_pathlib_path(path).glob("*") if name.is_dir()]


def get_train_dir(train_base_dir: os.PathLike, train_name: str) -> os.PathLike:
    dirs = get_subdirs(train_base_dir / train_name)

    assert len(dirs) == 1

    return dirs[0]


def main4():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_NEUTRAL_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=torch.float64,
        device=DEVICE,
    )

    J = model_data.kin_tree.joints_cnt
    V = model_data.verts_cnt
    BS = model_data.body_shapes_cnt

    body_shape_vert_dir_interp = torchrbf.RBFInterpolator(
        y=model_data.vert_pos.cpu().to(torch.float64),  # [V, 3]
        d=model_data.body_shape_vert_dir.cpu().reshape(V, 3 * BS).to(torch.float64),
        smoothing=1.0,
        kernel="cubic",
        degree=2,
    ).to(DEVICE, torch.float64)

    my_body_shape_vert_dir_interp = \
        rbf_utils.interp_utils.RBFInterpolator.from_data_point(
            data_pos=model_data.vert_pos.to(torch.float32),  # [V, 3]
            data_val=model_data.body_shape_vert_dir.reshape(
                V, 3 * BS).to(torch.float32),
            kernel=rbf_utils.radial_func.CubicRadialFunc(),
            degree=2,
            smoothness=1.0,
        ).to(DEVICE, torch.float64)

    data_x = model_data.vert_pos
    data_y = model_data.body_shape_vert_dir.reshape(V, 3 * BS)

    re_data_y = body_shape_vert_dir_interp(data_x)
    my_re_data_y = my_body_shape_vert_dir_interp(data_x)

    print(f"m = {(re_data_y - data_y).square().mean().sqrt():.6e}")
    print(f"my_m = {(my_re_data_y - data_y).square().mean().sqrt():.6e}")

    point_pos = torch.normal(
        mean=0.0, std=1.0, size=(1000, 3), dtype=torch.float64, device=DEVICE)

    result_a = body_shape_vert_dir_interp(point_pos)
    result_b = my_body_shape_vert_dir_interp(point_pos)

    print(f"{result_a.shape=}")
    print(f"{result_b.shape=}")

    diff = result_a - result_b

    rel_diff = (diff.abs() / (result_a.abs() + 1e-2))

    print(f"max abs diff = {diff.abs().max():.6e}")
    print(f"mean abs diff = {diff.abs().mean():.6e}")
    print(f"rel_diff = {rel_diff.max():.6e}")


if __name__ == "__main__":
    main4()
