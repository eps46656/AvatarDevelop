
import math
import pathlib

import torch

import camera_utils
import utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


DEVICE = utils.CPU


def main1():
    img_h, img_w = 720.0, 1280.0

    view_mat = torch.eye(4, dtype=utils.FLOAT)

    fov_diag = 45 * utils.DEG

    focal_length = camera_utils.GetFocalLengthByDiagFoV(img_h, img_w, fov_diag)

    near = 0.2
    far = 100.0

    N = 10

    view_mat = torch.eye(4, dtype=utils.FLOAT)

    view_mat[:3, :3] = utils.AxisAngleToRotMat(
        utils.RandRotVec((3,), dtype=utils.FLOAT))

    view_mat[:3, 3] = torch.rand((3,), dtype=utils.FLOAT)

    my_proto_camera = camera_utils.ProtoCamera(
        view_volume=camera_utils.Volume.FromFovDiag(
            img_h=img_h,
            img_w=img_w,
            fov_diag=fov_diag,
            depth_near=near,
            depth_far=far,
        ),

        proj_type=camera_utils.ProjType.PERS
    )

    my_proj_mat = my_proto_camera.GetProj(
        view_coord=utils.Coord3.FromStr("LUF"),
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
    )

    print(f"{my_proj_mat=}")


"""

my_proj_mat=
tensor([[ 2.7699,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  4.9243,  0.0000,  0.0000],
        [ 0.0000,  0.0000, -1.0020, -0.2004],
        [ 0.0000,  0.0000, -1.0000,  0.0000]])

"""


if __name__ == "__main__":
    main1()
