import itertools

import torch
from beartype import beartype

from .. import camera_utils, transform_utils, utils


@beartype
def get_nice_camera_config(img_h: int, img_w: int) -> camera_utils.CameraConfig:
    return camera_utils.CameraConfig.from_fov_diag(
        fov_diag=50.0 * utils.DEG,
        depth_near=utils.DEPTH_NEAR,
        depth_far=utils.DEPTH_FAR,
        img_h=img_h,
        img_w=img_w,
    )


@beartype
def get_nice_camera_transform(
    model_transform: transform_utils.ObjectTransform,
    # model <-> model space

    radius: float,
) -> transform_utils.ObjectTransform:
    thetas = [
        0.75 * 90.0 * utils.DEG,
        1.10 * 90.0 * utils.DEG,
    ]

    phis = [
        (0 / 8) * 360.0 * utils.DEG,  # front
        (1 / 8) * 360.0 * utils.DEG,  # fron left
        (3 / 8) * 360.0 * utils.DEG,  # back left
        (4 / 8) * 360.0 * utils.DEG,  # back
        (5 / 8) * 360.0 * utils.DEG,  # back right
        (7 / 8) * 360.0 * utils.DEG,  # front right
    ]

    pos = list()
    l_vec = list()
    u_vec = list()
    f_vec = list()

    quasi_u_vec = torch.tensor(
        [0, 0, 1], dtype=torch.float64, device=utils.CPU_DEVICE)

    for theta, phi in itertools.product(thetas, phis):
        cur_pos = torch.tensor(utils.sph_to_cart(
            radius=radius,
            theta=theta,
            phi=phi,
        ), dtype=torch.float64, device=utils.CPU_DEVICE)

        cur_f_vec = utils.vec_normed(-cur_pos)
        cur_l_vec = utils.vec_normed(utils.vec_cross(quasi_u_vec, cur_f_vec))
        cur_u_vec = utils.vec_cross(cur_f_vec, cur_l_vec)

        pos.append(cur_pos.tolist())
        l_vec.append(cur_l_vec.tolist())
        u_vec.append(cur_u_vec.tolist())
        f_vec.append(cur_f_vec.tolist())

    camera_world_transform = transform_utils.ObjectTransform.from_matching(
        dirs="LUF",

        pos=torch.tensor(pos, dtype=torch.float64, device=utils.CPU_DEVICE),

        vecs=(
            torch.tensor(l_vec, dtype=torch.float64, device=utils.CPU_DEVICE),
            torch.tensor(u_vec, dtype=torch.float64, device=utils.CPU_DEVICE),
            torch.tensor(f_vec, dtype=torch.float64, device=utils.CPU_DEVICE),
        ),
    )
    # camera <-> world

    model_world_transform = transform_utils.ObjectTransform.from_matching(
        dirs="FLU", dtype=torch.float64, device=utils.CPU_DEVICE)
    # model <-> world

    camera_model_space_transform = camera_world_transform.collapse(
        model_world_transform.get_trans_to(
            model_transform.to(utils.CPU_DEVICE, torch.float64))
        # world -> model space
    ).to(model_transform.device, model_transform.dtype)
    # camera <-> model space

    return camera_model_space_transform
