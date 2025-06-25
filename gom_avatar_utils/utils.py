import dataclasses
import math
import typing

import torch
from beartype import beartype

from .. import mesh_utils, utils, vision_utils


@dataclasses.dataclass
class FaceCoordResult:
    r: torch.Tensor  # [..., 3, 3]
    t: torch.Tensor  # [..., 3]
    area: torch.Tensor  # [...]


@beartype
def get_face_coord(mesh_data: mesh_utils.MeshData) -> FaceCoordResult:
    vp_a = mesh_data.face_vert_pos[..., 0, :]
    vp_b = mesh_data.face_vert_pos[..., 1, :]
    vp_c = mesh_data.face_vert_pos[..., 2, :]

    g = mesh_data.face_mean
    # [..., 3]

    axis_x = g - vp_a

    axis_z = utils.vec_cross(vp_b - vp_a, vp_c - vp_a)
    axis_z_norm = utils.vec_norm(axis_z)

    axis_y = utils.vec_cross(axis_z, axis_x)

    area = axis_z_norm * 0.5
    # [...]

    normed_axis_x = utils.vec_normed(axis_x)
    normed_axis_y = utils.vec_normed(axis_y)
    normed_axis_z = axis_z / (1e-6 + axis_z_norm)[..., None]

    err = utils.vec_dot(normed_axis_z, normed_axis_x).abs().max()
    assert err <= 1e-4, err

    err = utils.vec_dot(normed_axis_x, normed_axis_y).abs().max()
    assert err <= 1e-4, err

    err = utils.vec_dot(normed_axis_y, normed_axis_z).abs().max()
    assert err <= 1e-4, err

    r = torch.stack([
        normed_axis_x,
        normed_axis_y,
        normed_axis_z,
    ], -1)
    # [..., 3, 3]

    """

    [
        [axis_x[..., 0], axis_y[..., 0], axis_z[..., 0]],
        [axis_x[..., 1], axis_y[..., 1], axis_z[..., 1]],
        [axis_x[..., 2], axis_y[..., 2], axis_z[..., 2]],
    ]

    """

    return FaceCoordResult(r=r, t=g, area=area)


@beartype
def make_dilate_kernel(
    *,
    img_h: typing.Optional[int],
    img_w: typing.Optional[int],

    sigma:  typing.Optional[float] = None,
    ratio_sigma:  typing.Optional[float] = None,

    opacity: float,
) -> torch.Tensor:  # [1, 1, K, K]
    if sigma is None:
        assert img_h is not None
        assert img_w is not None
        assert ratio_sigma is not None

        sigma = vision_utils.get_sigma(img_h, img_w, ratio_sigma)

    assert 0 < sigma

    kernel_radius = max(1, math.ceil(sigma * 3))

    kernel = (1 - opacity * vision_utils.make_gaussian_kernel(
        sigma=sigma,
        kernel_radius=kernel_radius,
        make_mean=False,
        dtype=torch.float64,
        device=utils.CUDA_DEVICE,
    )).log()[None, None, :, :]
    # [1, 1, K, K]

    return kernel


@beartype
def make_dilated_mask(
    mask: torch.Tensor,  # [..., H, W]
    kernel: torch.Tensor,  # [1, 1, K, K]
) -> torch.Tensor:  # [..., H, W]
    H, W, K = -1, -2, -3

    H, W, K = utils.check_shapes(
        mask, (..., H, W),
        kernel, (1, 1, K, K),
    )

    assert K % 2 == 1

    kernel_radius = K // 2

    # [-1, 1, H, W]

    dilated_mask = 1 - (
        torch.nn.functional.conv2d(
            input=mask.reshape(-1, 1, H, W),
            weight=kernel,
            padding=kernel_radius,
        ).view(mask.shape) / mask.sum((-2, -1), keepdim=True)
    ).exp()
    # [..., H, W]

    return dilated_mask
