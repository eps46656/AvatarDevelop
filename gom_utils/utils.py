import dataclasses
import math

import torch
from beartype import beartype

from .. import utils


@beartype
def get_face_coord(
    vertex_positions_a: torch.Tensor,  # [..., 3]
    vertex_positions_b: torch.Tensor,  # [..., 3]
    vertex_positions_c: torch.Tensor,  # [..., 3]
) -> tuple[
    torch.Tensor,  # [..., 3, 3]
    torch.Tensor,  # [..., 3]
]:
    vpa = vertex_positions_a
    vpb = vertex_positions_b
    vpc = vertex_positions_c

    assert vpa.isfinite().all()
    assert vpb.isfinite().all()
    assert vpc.isfinite().all()

    utils.check_shapes(
        vpa, (..., 3),
        vpb, (..., 3),
        vpc, (..., 3),
    )

    device = utils.check_devices(vpa, vpb, vpc)

    dtype = utils.promote_types(vpa, vpb, vpc)

    batch_shape = utils.broadcast_shapes(
        vpa.shape[:-1],
        vpb.shape[:-1],
        vpc.shape[:-1],
    )

    vpa = vpa.to(device, dtype).expand(batch_shape + (3,))
    vpb = vpb.to(device, dtype).expand(batch_shape + (3,))
    vpc = vpc.to(device, dtype).expand(batch_shape + (3,))

    g = (vpa + vpb + vpc) / 3
    # [..., 3]

    f1 = vpc - g
    f2 = (vpa - vpb) / math.sqrt(3)

    t0 = 0.5 * torch.atan2(2 * utils.dot(f1, f2), f1.square() - f2.square())

    cos_t0 = torch.cos(t0)
    sin_t0 = torch.sin(t0)

    axis_x = f1 * cos_t0 + f2 * sin_t0
    axis_y = f2 * cos_t0 - f1 * sin_t0
    axis_z = torch.cross(axis_x, axis_y)
    # [..., 3]

    err = (axis_z * axis_x).sum(-1).abs().max()
    assert err <= 2e-3, err

    err = (axis_x * axis_y).sum(-1).abs().max()
    assert err <= 2e-3, err

    err = (axis_y * axis_z).sum(-1).abs().max()
    assert err <= 2e-3, err

    rs = torch.stack([axis_x, axis_y, axis_z], -1)

    return rs, g
