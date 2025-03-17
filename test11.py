
import math
import pathlib

import torch

import utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


INT = torch.int32
FLOAT = torch.float32
DEVICE = torch.device("cpu")


def main1():
    P = 8
    Q = 5
    N = 10

    src = utils.MakeHomo(torch.rand((N, P-1), dtype=utils.FLOAT))

    print(f"{src[:, -1].mean()}")

    ans_homo: torch.Tensor = torch.rand((Q, P), dtype=utils.FLOAT)

    dst = utils.HomoNormalize((ans_homo @ src.unsqueeze(-1)).squeeze(-1))

    H = utils.DLT(src, dst, True)

    print(src)
    print(dst)

    print(f"{ans_homo}")
    print(f"{H}")

    re_dst = utils.HomoNormalize((H @ src.unsqueeze(-1)).squeeze(-1))

    print(f"{utils.GetDiff(dst, re_dst).mean()}")


def MakeProjMat(
    fov_h: float,
    fov_w: float,

    near: float,
    far: float,
):
    assert 0 < fov_h
    assert fov_h < 180 * utils.DEG

    assert 0 < fov_w
    assert fov_w < 180 * utils.DEG

    half_ratio_h = math.tan(fov_h / 2)
    half_ratio_w = math.tan(fov_w / 2)

    near_t = near * half_ratio_h
    near_b = -near_t

    near_r = near * half_ratio_w
    near_l = -near_r

    far_t = far * half_ratio_h
    far_b = -far_t

    far_r = far * half_ratio_w
    far_l = -far_r

    H, err = utils.DLT(
        src=torch.tensor([
            [near_l, 0, near, 1],
            [near_r, 0, near, 1],
            [0, near_b, near, 1],
            [0, near_t, near, 1],

            [far_l, 0, far, 1],
            [far_r, 0, far, 1],
            [0, far_b, far, 1],
            [0, far_t, far, 1],
        ], dtype=torch.float64),

        dst=torch.tensor([
            [-1, 0, 0, 1],
            [+1, 0, 0, 1],
            [0, -1, 0, 1],
            [0, +1, 0, 1],

            [-1, 0, 1, 1],
            [+1, 0, 1, 1],
            [0, -1, 1, 1],
            [0, +1, 1, 1],
        ], dtype=torch.float64),

        normalize=False,
        calc_err=True,
    )

    assert err <= 1e-6

    H = H / H[3, 2]

    return H


def main2():
    with utils.Timer():
        proj_mat = MakeProjMat(
            60 * utils.DEG,
            60 * utils.DEG,

            0.5,
            100
        )

    print(f"{proj_mat=}")


if __name__ == "__main__":
    main2()
