import dataclasses

import torch
from beartype import beartype

from .. import utils


@dataclasses.dataclass
class FaceCoordResult:
    rs: torch.Tensor  # [..., 3, 3]
    ts: torch.Tensor  # [..., 3]
    areas: torch.Tensor  # [...]


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

    batch_shape = utils.broadcast_shapes(
        vp_a.shape[:-1],
        vp_b.shape[:-1],
        vp_c.shape[:-1],
    )

    vp_a = vp_a.to(*dd).expand(batch_shape + (3,))
    vp_b = vp_b.to(*dd).expand(batch_shape + (3,))
    vp_c = vp_c.to(*dd).expand(batch_shape + (3,))

    g = (vp_a + vp_b + vp_c) / 3
    # [..., 3]

    axis_x = g - vp_a

    axis_z = utils.vec_cross(vp_b - vp_a, vp_c - vp_a)

    axis_y = utils.vec_cross(axis_z, axis_x)

    area = utils.vec_norm(axis_z) * 0.5
    # [...]

    normed_axis_x = utils.vec_normed(axis_x)
    normed_axis_y = utils.vec_normed(axis_y)
    normed_axis_z = utils.vec_normed(axis_z)

    err = utils.vec_dot(normed_axis_z, normed_axis_x).abs().max()
    assert err <= 2e-3, err

    err = utils.vec_dot(normed_axis_x, normed_axis_y).abs().max()
    assert err <= 2e-3, err

    err = utils.vec_dot(normed_axis_y, normed_axis_z).abs().max()
    assert err <= 2e-3, err

    rs = torch.stack([
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

    return FaceCoordResult(rs=rs, ts=g, areas=area)
