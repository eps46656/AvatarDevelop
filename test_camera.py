import pathlib

import torch

from . import camera_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")


def main1():
    img_h = utils.rand_int(200, 1000)
    img_w = utils.rand_int(200, 1000)

    depth_near, depth_far = utils.min_max(
        utils.rand_float(0.01, 100.0),
        utils.rand_float(0.01, 100.0),
    )

    camera_config = camera_utils.CameraConfig.from_slope_udlr(
        slope_u=utils.rand_float(0.0, 10.0),
        slope_d=utils.rand_float(0.0, 10.0),
        slope_l=utils.rand_float(0.0, 10.0),
        slope_r=utils.rand_float(0.0, 10.0),

        depth_near=depth_near,
        depth_far=depth_far,

        img_h=img_h,
        img_w=img_w,
    )

    proj_mat = camera_utils.make_proj_mat(
        camera_config=camera_config,
        view_dirs=utils.Dir3.FromStr("BLU"),
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
    )

    print(f"{proj_mat=}")


if __name__ == "__main__":
    main1()

    print("ok")
