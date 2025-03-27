
import math
import pathlib

import torch

import camera_utils
import utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


DEVICE = utils.CPU_DEVICE


def main1():
    img_h, img_w = 720.0, 1280.0

    view_mat = torch.eye(4, dtype=utils.FLOAT)

    fov_diag = 90 * utils.DEG

    focal_length = camera_utils.make_focal_length_by_fov_diag(
        img_h, img_w, fov_diag)

    near = 0.2
    far = 100.0

    N = 10

    view_mat = torch.eye(4, dtype=utils.FLOAT)

    view_mat[:3, :3] = utils.axis_angle_to_rot_mat(
        utils.rand_rot_vec((3,), dtype=utils.FLOAT), homo=False)

    view_mat[:3, 3] = torch.rand((3,), dtype=utils.FLOAT)

    my_camera_config = camera_utils.CameraConfig.from_fov_diag(
        fov_diag=fov_diag,
        depth_near=near,
        depth_far=far,
        img_h=img_h,
        img_w=img_w,
    )

    my_proj_mat = camera_utils.GetProjMat(
        camera_config=my_camera_config,
        view_coord=utils.Dir3.FromStr("LUF"),
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
    )

    print(f"{my_proj_mat=}")


"""

"""


if __name__ == "__main__":
    main1()
