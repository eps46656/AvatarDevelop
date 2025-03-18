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
from smplx import smplx
import utils
from kin_tree import KinTree

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


INT = torch.int32
FLOAT = torch.float32
DEVICE = utils.CPU


def main1():
    a = torch.Tensor([[7.7425e-01,  1.9263e-01,  9.0920e-01],

                      [1.1747e+00,  6.5456e-01, -2.8892e-02],
                      [-6.6441e-01, -1.4415e+00,  1.1058e+00],
                      [-4.5373e-01,  3.6159e-01,  1.2116e+00],
                      [-4.2416e-01,  1.6019e+00, -4.0403e-01],
                      [9.6897e-01, -1.6320e-01, -3.8274e-01],
                      [5.8436e-01, -5.7626e-02,  3.2737e-02],
                      [1.5022e+00,  2.2498e-01, -3.9384e-01],
                      [-1.2859e+00, -1.4285e+00, -5.6178e-01],
                      [-1.2131e+00,  7.1641e-01,  2.9181e-02],
                      [9.1014e-01, -8.5048e-01,  8.7226e-02],
                      [1.3568e+00,  4.4146e-01,  1.1991e+00],
                      [1.5482e+00, -1.7184e-01, -1.0939e+00],
                      [-4.0317e-01, -7.3292e-01,  7.8577e-01],
                      [-1.0873e+00, -2.2605e-01, -1.5409e+00],
                      [1.4715e-02, -2.2618e+00,  7.4548e-01],
                      [4.6987e-01,  2.3414e-01, -9.7909e-01],
                      [1.0398e+00, -1.1405e+00,  1.0432e+00],
                      [-9.0847e-01, -2.1580e+00,  4.0132e-01],
                      [1.4254e+00,  2.0190e-01,  2.6088e-01],
                      [-1.2765e+00, -8.7553e-02,  2.1422e-01],
                      [2.5826e-01, -2.7427e-01, -2.4013e-01],

                      [1.5056e+00,  3.6265e-01,  8.1563e-02],

                      [-8.5771e-01,  1.8175e+00, -8.0225e-04],

                      [-1.5625e+00,  1.5438e-01, -2.5362e-01],

                      [-4.6422e-01, -1.6305e+00,  5.4571e-01],
                      [-3.7817e-01,  1.8244e+00, -1.4465e+00],
                      [-6.0242e-01,  2.3057e+00, -8.2980e-01],
                      [1.6659e-01, -1.4321e+00, -1.3059e+00],
                      [-1.4090e+00, -7.7208e-01,  5.4545e-01],
                      [1.4295e+00,  1.4329e+00, -3.3356e-01],
                      [-4.3606e-01,  2.6801e+00, -2.4376e-01],
                      [-1.1102e+00,  1.7275e-01,  1.1187e+00],
                      [1.4944e+00, -1.1220e+00, -3.4284e-02],
                      [-1.1319e+00, -7.9925e-01, -6.9658e-01],
                      [1.4410e+00, -1.2805e+00, -4.7618e-01],
                      [1.2627e+00, -4.3487e-01, -6.2023e-01],
                      [7.8081e-02,  4.8656e-01,  1.3201e+00],
                      [-5.6046e-01,  2.0106e+00,  1.3286e-02],
                      [5.7282e-01,  6.3706e-01,  1.4714e-01],
                      [-1.6600e+00, -7.3612e-01, -1.2175e+00],
                      [5.7674e-02, -1.3832e+00,  5.1549e-01],
                      [-6.5004e-02, -1.3407e-01,  1.0006e+00],
                      [2.4917e+00,  7.3063e-01,  5.5096e-01],
                      [-5.0876e-01, -5.7246e-01,  7.4094e-01],
                      [1.6182e-01,  9.0239e-02,  1.5857e-02],
                      [-1.9774e+00,  1.0435e+00,  6.5800e-01],
                      [-4.0360e-01,  4.8181e-01,  4.6201e-01],
                      [-1.4202e+00,  6.2614e-02, -1.4706e+00],
                      [1.5990e-02, -2.0624e+00,  2.5377e-01],
                      [1.3001e-01,  1.2131e-01,  1.3025e+00],
                      [-4.0719e-01,  8.2725e-01,  4.6333e-01],
                      [7.2170e-01, -3.6642e-01, -2.6026e+00],
                      [-1.0394e+00,  1.2380e+00, -6.7899e-01],
                      [4.7335e-01, -2.4216e-01,  1.6914e+00]])

    b = torch.Tensor([[0.7743,  0.1926,  0.9092],  # 0

                      [1.1747,  0.6546, -0.0289],
                      [-0.6644, -1.4415,  1.1058],
                      [-0.4537,  0.3616,  1.2116],
                      [-0.4242,  1.6019, -0.4040],
                      [0.9690, -0.1632, -0.3827],
                      [0.5844, -0.0576,  0.0327],
                      [1.5022,  0.2250, -0.3938],
                      [-1.2859, -1.4285, -0.5618],
                      [-1.2131,  0.7164,  0.0292],
                      [0.9101, -0.8505,  0.0872],
                      [1.3568,  0.4415,  1.1991],
                      [1.5482, -0.1718, -1.0939],
                      [-0.4032, -0.7329,  0.7858],
                      [-1.0873, -0.2261, -1.5409],
                      [0.0147, -2.2618,  0.7455],
                      [0.4699,  0.2341, -0.9791],
                      [1.0398, -1.1405,  1.0432],
                      [-0.9085, -2.1580,  0.4013],
                      [1.4254,  0.2019,  0.2609],
                      [-1.2765, -0.0876,  0.2142],
                      [0.2583, -0.2743, -0.2401],

                      [0.0000,  0.0000,  0.0000],

                      [0.0000,  0.0000,  0.0000],

                      [0.0000,  0.0000,  0.0000],

                      [-0.4642, -1.6305,  0.5457],
                      [-0.3782,  1.8244, -1.4465],
                      [-0.6024,  2.3057, -0.8298],
                      [0.1666, -1.4321, -1.3059],
                      [-1.4090, -0.7721,  0.5455],
                      [1.4295,  1.4329, -0.3336],
                      [-0.4361,  2.6801, -0.2438],
                      [-1.1102,  0.1727,  1.1187],
                      [1.4944, -1.1220, -0.0343],
                      [-1.1319, -0.7993, -0.6966],
                      [1.4410, -1.2805, -0.4762],
                      [1.2627, -0.4349, -0.6202],
                      [0.0781,  0.4866,  1.3201],
                      [-0.5605,  2.0106,  0.0133],
                      [0.5728,  0.6371,  0.1471],
                      [-1.6600, -0.7361, -1.2175],
                      [0.0577, -1.3832,  0.5155],
                      [-0.0650, -0.1341,  1.0006],
                      [2.4917,  0.7306,  0.5510],
                      [-0.5088, -0.5725,  0.7409],
                      [0.1618,  0.0902,  0.0159],
                      [-1.9774,  1.0435,  0.6580],
                      [-0.4036,  0.4818,  0.4620],
                      [-1.4202,  0.0626, -1.4706],
                      [0.0160, -2.0624,  0.2538],
                      [0.1300,  0.1213,  1.3025],
                      [-0.4072,  0.8272,  0.4633],
                      [0.7217, -0.3664, -2.6026],
                      [-1.0394,  1.2380, -0.6790],
                      [0.4734, -0.2422,  1.6914]])

    print(f"{b.reshape((-1, 3))}=")

    c = list(a.flatten())
    d = list(b.flatten())

    err = ((a.flatten() - b.flatten())**2).mean()**0.5

    print(f"{err=}")

    for idx, (i, j) in enumerate(zip(c, d)):
        if 0.1 <= abs(i-j):
            print(f"{idx}, {i}, {j}")


if __name__ == "__main__":
    main1()
