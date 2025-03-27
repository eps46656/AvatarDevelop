import dataclasses
import math

import torch
from beartype import beartype

from .. import utils


@beartype
@dataclasses.dataclass
class FaceCoordResult:
    Ts: torch.Tensor  # [..., F, 4, 4]
    face_area: torch.Tensor  # [..., F]


@beartype
def get_face_coord(
    vertex_positions_a: torch.Tensor,  # [..., 3]
    vertex_positions_b: torch.Tensor,  # [..., 3]
    vertex_positions_c: torch.Tensor,  # [..., 3]
) -> FaceCoordResult:
    vpa = vertex_positions_a
    vpb = vertex_positions_b
    vpc = vertex_positions_c

    utils.check_shapes(
        vpa, (..., 3),
        vpb, (..., 3),
        vpc, (..., 3),
    )

    device = utils.check_device(vpa, vpb, vpc)

    batch_shape = utils.broadcast_shapes(
        vpa.shape[:-1],
        vpb.shape[:-1],
        vpc.shape[:-1],
    )

    vpa = vpa.expand(batch_shape + (3,))
    vpb = vpb.expand(batch_shape + (3,))
    vpc = vpc.expand(batch_shape + (3,))

    s = (vpa + vpb + vpc) / 3
    # [..., 3]

    f1 = vpc - s
    f2 = (vpa - vpb) / math.sqrt(3)

    f1_sq = f1.square().sum(-1)
    f2_sq = f2.square().sum(-1)
    f1_dot_f2 = (f1 * f2).sum(-1)

    """

    half_cos_2t = 0.5 / (1 + tan_2t.square()).sqrt()

    cos_2t = 1 / (1 + tan_2t.square()).sqrt()

    half_cos_2t = 0.5 / (1 + tan_2t.square()).sqrt()

    cos_t = ((1 + cos_2t) / 2).sqrt()
          = (0.5 + half_cos_2t).sqrt

    cos_t = ((utils.EPS + 0.5) + half_cos_2t).sqrt().unsqueeze(-1)
    sin_t = ((utils.EPS + 0.5) - half_cos_2t).sqrt().unsqueeze(-1)
    """

    t = torch.atan2(2 * f1_dot_f2, f1_sq - f2_sq) * 0.5

    cos_t = torch.cos(t).unsqueeze(-1)
    sin_t = torch.sin(t).unsqueeze(-1)

    Ts = torch.empty(
        batch_shape + (4, 4), dtype=utils.FLOAT, device=device)

    axis_x = f1 * cos_t + f2 * sin_t
    axis_y = f2 * cos_t - f1 * sin_t
    axis_z = torch.linalg.cross(axis_x, axis_y)

    normalized_axis_x = Ts[..., :3, 0] = utils.normalized(axis_x)
    normalized_axis_y = Ts[..., :3, 1] = utils.normalized(axis_y)
    normalized_axis_z = Ts[..., :3, 2] = \
        torch.linalg.cross(normalized_axis_x, normalized_axis_y)

    err = (Ts[..., :3, 0] * Ts[..., :3, 1]).sum(-1).abs().max()

    assert err <= 2e-3, err

    Ts[..., :3, 3] = s
    Ts[..., 3, :3] = 0
    Ts[..., 3, 3] = 1

    ret = FaceCoordResult(
        Ts=Ts,
        face_area=utils.vector_norm(axis_z) * (3 * math.sqrt(3) / 4),
    )

    return ret
