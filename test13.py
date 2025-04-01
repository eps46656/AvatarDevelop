
import math
import pathlib

import torch

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures

from . import camera_utils, transform_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


DEVICE = utils.CPU_DEVICE


def main1():
    pass


def main2():
    img_h, img_w = 720, 1280

    view_mat = torch.eye(4, dtype=utils.FLOAT)

    fov_diag = 90 * utils.DEG

    focal_length = camera_utils.make_focal_length_by_fov_diag(
        img_h, img_w, fov_diag)

    near = 0.2
    far = 100.0

    N = 100

    view_mat = torch.eye(4, dtype=utils.FLOAT)

    view_mat[:3, :3] = utils.axis_angle_to_rot_mat(
        utils.rand_rot_vec((3,), dtype=utils.FLOAT), out_shape=(3, 3))

    view_mat[:3, 3] = torch.rand((3,), dtype=utils.FLOAT)

    camera_config = camera_utils.CameraConfig.from_fov_diag(
        fov_diag=fov_diag,

        depth_near=near,
        depth_far=far,

        img_h=img_h,
        img_w=img_w,
    )

    my_proj_mat = camera_utils.make_proj_mat(
        camera_config=camera_config,
        camera_view_transform=transform_utils.ObjectTransform
        .from_matching("LUF"),
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
    )

    cameras = pytorch3d.renderer.PerspectiveCameras(
        R=view_mat[:3, :3].transpose(0, 1).unsqueeze(0),
        T=view_mat[:3, 3].unsqueeze(0),
        focal_length=[focal_length / min(img_h, img_w) * 2],
        principal_point=[(0, 0)],
        in_ndc=True,
        image_size=[(img_h, img_w)],
        device=DEVICE,
    )

    proj_trans = cameras.get_projection_transform()

    proj_mat = proj_trans.get_matrix()

    print(f"{proj_mat=}")

    print(f"{my_proj_mat=}")

    points = torch.rand((N, 4), dtype=utils.FLOAT)
    points[:, 3] = 1

    ndc_points_a = cameras.transform_points_ndc(
        points[:, :3], image_size=(img_h, img_w))

    ndc_points_b = utils.do_homo(
        my_proj_mat,
        (view_mat @ points.unsqueeze(-1)).squeeze(-1)
    )

    # print(f"{ndc_points_a=}")
    # print(f"{ndc_points_b=}")

    print(f"{utils.get_l2_rms(ndc_points_a - ndc_points_b[:, :-1])}")


if __name__ == "__main__":
    main2()
