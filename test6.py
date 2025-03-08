import numpy as np
import utils
import pickle

import torch
import camera_utils

import pathlib

import json

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

INT = torch.int32
FLOAT = torch.float32
DEVICE = torch.device("cuda")


ORIGIN = torch.tensor([[0], [0], [0]], dtype=FLOAT, device=DEVICE)
X_AXIS = torch.tensor([[1], [0], [0]], dtype=FLOAT, device=DEVICE)
Y_AXIS = torch.tensor([[0], [1], [0]], dtype=FLOAT, device=DEVICE)
Z_AXIS = torch.tensor([[0], [0], [1]], dtype=FLOAT, device=DEVICE)


def main1():
    raduis = 10
    theta = 60 * utils.DEG
    phi = (180 + 45) * utils.DEG

    view_mat = camera_utils.MakeViewMat(
        origin=utils.Sph2XYZ(raduis, theta, phi, Z_AXIS, X_AXIS, Y_AXIS),
        aim=ORIGIN,
        quasi_u_dir=Y_AXIS,
        view_axes="luf",
    ).to(dtype=FLOAT, device=DEVICE)

    print(f"view_mat=")
    print(f"{view_mat}")

    print(f"torch.inverse(view_mat)=")
    print(f"{torch.inverse(view_mat)}")


if __name__ == "__main__":
    main1()
