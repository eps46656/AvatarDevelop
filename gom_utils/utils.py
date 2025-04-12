import dataclasses

import torch
from beartype import beartype

from .. import utils


@dataclasses.dataclass
class FaceCoordResult:
    r: torch.Tensor  # [..., 3, 3]
    t: torch.Tensor  # [..., 3]
    area: torch.Tensor  # [...]


@beartype
def get_face_coord(
    vert_pos_a: torch.Tensor,  # [..., 3]
    vert_pos_b: torch.Tensor,  # [..., 3]
    vert_pos_c: torch.Tensor,  # [..., 3]
) -> FaceCoordResult:
    vp_a = vert_pos_a
    vp_b = vert_pos_b
    vp_c = vert_pos_c

    utils.check_shapes(
        vp_a, (..., 3),
        vp_b, (..., 3),
        vp_c, (..., 3),
    )

    dd = (
        utils.check_devices(vp_a, vp_b, vp_c),
        utils.promote_dtypes(vp_a, vp_b, vp_c),
    )

    s = utils.broadcast_shapes(vp_a, vp_b, vp_c)

    vp_a = vp_a.to(*dd).expand(s)
    vp_b = vp_b.to(*dd).expand(s)
    vp_c = vp_c.to(*dd).expand(s)

    g = (vp_a + vp_b + vp_c) / 3
    # [..., 3]

    axis_x = g - vp_a

    axis_z = utils.vec_cross(vp_b - vp_a, vp_c - vp_a)
    axis_z_norm = utils.vec_norm(axis_z)

    axis_y = utils.vec_cross(axis_z, axis_x)

    area = axis_z_norm * 0.5
    # [...]

    normed_axis_x = utils.vec_normed(axis_x)
    normed_axis_y = utils.vec_normed(axis_y)
    normed_axis_z = axis_z / (1e-6 + axis_z_norm).unsqueeze(-1)

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
