import dataclasses
import json
import math
import pathlib
import pickle

import torch

from . import utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


INT = torch.int32
FLOAT = torch.float32
DEVICE = utils.CPU


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


def CheckIsZero(
    zero: torch.Tensor,  # [...]
):
    err = zero.square().mean().sqrt()
    assert err <= 1e-3, f"{err=}"


def CheckIsEye(
    eye: torch.Tensor,  # [..., D, D]
):
    D, = utils.CheckShapes(eye, (..., -1, -1))

    for i in range(D):
        for j in range(D):
            k = eye[..., i, j]

            if i == j:
                k -= 1

            assert k.square().max().sqrt() <= 1e-3


def CheckIsRotMat(
    rot_mat: torch.Tensor  # [..., 3, 3]
):
    utils.CheckShapes(rot_mat, (..., 3, 3))

    rot_mat_t = rot_mat.transpose(-1, -2)

    k1 = rot_mat_t @ rot_mat
    k2 = rot_mat @ rot_mat_t

    CheckIsEye(k1)
    CheckIsEye(k2)

    CheckIsZero(rot_mat.det() - 1)


def main1():
    B = 10000

    for _ in range(1):
        axis = utils.RandUnit((B, 3), dtype=FLOAT, device=DEVICE)
        angle = torch.abs(torch.rand(B)) * (math.pi -
                                            utils.EPS * 10) + utils.EPS * 5

        rot_mat = utils.AxisAngleToRotMat(axis, angle, out_shape=(3, 3))

        CheckIsRotMat(rot_mat)

        re_axis, re_angle = utils.RotMatToAxisAngle(rot_mat)

        assert re_axis.isfinite().all()
        assert re_angle.isfinite().all()

        re_rot_mat = utils.AxisAngleToRotMat(
            re_axis, re_angle, out_shape=(3, 3))

        CheckIsRotMat(re_rot_mat)

        CheckIsZero(rot_mat - re_rot_mat)


def main2():
    B = 10000

    for _ in range(1):
        q = utils.RandUnit((B, 4), dtype=FLOAT, device=DEVICE)
        # q = torch.zeros((B, 4), dtype=FLOAT, device=DEVICE)
        # q[..., 0] = 1
        # q[..., 3] = 0

        rot_mat = utils.QuaternionToRotMat(q, order="XZYW", out_shape=(3, 3))

        assert rot_mat.isfinite().all()

        CheckIsRotMat(rot_mat)

        re_q = utils.RotMatToQuaternion(rot_mat, order="XZYW")

        assert re_q.isfinite().all()

        re_rot_mat = utils.QuaternionToRotMat(
            re_q, order="XZYW", out_shape=(3, 3))

        assert re_rot_mat.isfinite().all()

        CheckIsZero(rot_mat - re_rot_mat)


def main3():
    B = 10000

    for _ in range(1):
        q = utils.RandUnit((B, 4), dtype=FLOAT, device=DEVICE)

        axis, angle = utils.QuaternionToAxisAngle(q, order="XYZW")

        assert axis.isfinite().all()
        assert angle.isfinite().all()

        re_q = utils.AxisAngleToQuaternion(axis, angle, order="XYZW")

        assert re_q.isfinite().all()

        rot_mat = utils.QuaternionToRotMat(q, order="XYZW", out_shape=(3, 3))

        assert rot_mat.isfinite().all()

        CheckIsRotMat(rot_mat)

        re_rot_mat = utils.QuaternionToRotMat(
            re_q, order="XYZW", out_shape=(3, 3))

        CheckIsRotMat(rot_mat)

        CheckIsZero(rot_mat - re_rot_mat)


def main4():
    B = 10000

    for _ in range(1):
        p = utils.RandQuaternion((B, 4), dtype=FLOAT, device=DEVICE)
        q = utils.RandQuaternion((B, 4), dtype=FLOAT, device=DEVICE)

        mat_p = utils.QuaternionToRotMat(p, order="XYZW", out_shape=(3, 3))
        mat_q = utils.QuaternionToRotMat(q, order="XYZW", out_shape=(3, 3))

        mat_pq = mat_p @ mat_q

        r = utils.QuaternionMul(
            p, q,
            order_1="XYZW", order_2="XYZW", order_out="XYZW")

        mat_r = utils.QuaternionToRotMat(r, order="XYZW", out_shape=(3, 3))

        CheckIsRotMat(mat_pq)
        CheckIsRotMat(mat_r)

        CheckIsZero(mat_pq - mat_r)


if __name__ == "__main__":
    # main1()
    main2()
    # main3()
    # main4()
    print("Finish")
