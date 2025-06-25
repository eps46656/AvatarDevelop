import pathlib

import torch

from . import camera_utils, utils, transform_utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")


def main1():
    view_matrix = torch.tensor([
        [+0.2419, +0.0000, +0.9703, +0.0000,],
        [+0.0000, +1.0000, +0.0000, +0.0000,],
        [-0.9703, +0.0000, +0.2419, +0.0000,],
        [+2.7511, +0.0000, +2.0890, +1.0000,],
    ], dtype=utils.FLOAT).transpose(0, 1)

    proj_view_matrix = torch.tensor([
        [+0.6843, +0.0000, +0.9704, +0.9703,],
        [+0.0000, +2.8284, +0.0000, +0.0000,],
        [-2.7444, +0.0000, +0.2419, +0.2419,],
        [+7.7813, +0.0000, +2.0792, +2.0890,],
    ], dtype=utils.FLOAT).transpose(0, 1)

    print(view_matrix)

    print(proj_view_matrix)

    proj_matrix = proj_view_matrix @ view_matrix.inverse()

    print(f"{proj_matrix=}")


"""

proj_matrix=
tensor([[ 2.8284e+00,  0.0000e+00,  1.0601e-04, -2.1505e-04],
        [ 0.0000e+00,  2.8284e+00,  0.0000e+00,  0.0000e+00],
        [ 2.4182e-05,  0.0000e+00,  1.0001e+00, -1.0069e-02],
        [-2.6284e-09,  0.0000e+00,  1.0000e+00,  0.0000e+00]])

x 0 0 0
0 x 0 0
0 0 1 ?
0 0 1 0


(src_f * dst_f - src_n * dst_b) / src_fn
(src_f * src_n) * (dst_b - dst_f) / src_fn


src_f * dst_f - src_n * dst_b

src_f * src_n * dst_b

-src_f * src_n * dst_f




==

src_f - src_n

"""


def main2():
    f = 1 / 2.8284

    camera_config = camera_utils.CameraConfig.from_slope_udlr(
        slope_u=f,
        slope_d=f,
        slope_l=f,
        slope_r=f,
        img_h=1000,
        img_w=1000,
        depth_near=0.01,
        depth_far=200.0,
    )

    proj_mat = camera_utils.make_proj_mat_with_config(
        camera_config=camera_config,

        camera_transform=transform_utils.ObjectTransform.
        from_matching("RDF"),

        proj_config=camera_utils.ProjConfig(
            camera_proj_transform=transform_utils.ObjectTransform.
            from_matching("RDF"),
            delta_u=1.0,
            delta_d=1.0,
            delta_l=1.0,
            delta_r=1.0,
            delta_f=1.0,
            delta_b=0.0,
        ),

        dtype=utils.FLOAT,
    )

    print(f"{proj_mat=}")


if __name__ == "__main__":
    main2()

    print("ok")
