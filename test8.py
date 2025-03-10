import dataclasses
import json
import math
import pathlib
import pickle

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch

import blending_utils
import camera_utils
import config
import smplx.smplx
import utils
from kin_tree import KinTree

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


INT = torch.int32
FLOAT = torch.float32
DEVICE = torch.device("cpu")


ORIGIN = torch.tensor([[0], [0], [0]], dtype=FLOAT)
X_AXIS = torch.tensor([[1], [0], [0]], dtype=FLOAT)
Y_AXIS = torch.tensor([[0], [1], [0]], dtype=FLOAT)
Z_AXIS = torch.tensor([[0], [0], [1]], dtype=FLOAT)

SMPLX_FT = +Z_AXIS
SMPLX_BK = -Z_AXIS

SMPLX_LT = +X_AXIS
SMPLX_RT = -X_AXIS

SMPLX_UP = +Y_AXIS
SMPLX_DW = -Y_AXIS


def main1():
    B = 100

    for _ in range(10):
        axis = utils.RandUnit((B, 3), dtype=FLOAT, device=DEVICE)
        angle = torch.abs(torch.rand(B)) * (math.pi -
                                            utils.EPS * 10) + utils.EPS * 5

        rot_mat = utils.GetRotMat(axis, angle)

        print(f"{torch.det(rot_mat)=}")

        assert rot_mat.isfinite().all(), f"{axis=}\n{angle=}\n{rot_mat=}"

        re_axis, re_angle = utils.GetAxisAngle(rot_mat)

        axis_err = torch.mean(torch.sum((axis - re_axis)**2, -1)**0.5)

        angle_err_c = torch.mean(
            (torch.cos(angle) - torch.cos(re_angle))**2, -1)**0.5

        angle_err_s = torch.mean(
            (torch.sin(angle) - torch.sin(re_angle))**2, -1)**0.5

        print(f"{axis_err=}")
        print(f"{angle_err_c=}")
        print(f"{angle_err_s=}")


if __name__ == "__main__":
    main1()
