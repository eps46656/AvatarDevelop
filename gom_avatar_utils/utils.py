import dataclasses
import math

import torch
from beartype import beartype

from .. import utils


@beartype
@dataclasses.dataclass
class FaceCoordResult:
    Ts: torch.Tensor  # [..., F, 4, 4]
    normalized_Ts: torch.Tensor  # [..., F, 4, 4]
    face_area: torch.Tensor  # [..., F]


@beartype
def GetFaceCoord(
    vertex_positions_a: torch.Tensor,  # [..., 3]
    vertex_positions_b: torch.Tensor,  # [..., 3]
    vertex_positions_c: torch.Tensor,  # [..., 3]
):
    vpa = vertex_positions_a
    vpb = vertex_positions_b
    vpc = vertex_positions_c

    utils.CheckShapes(
        vpa, (..., 3),
        vpb, (..., 3),
        vpc, (..., 3),
    )

    device = utils.CheckDevice(vpa, vpb, vpc)

    batch_shapes = utils.BroadcastShapes(
        vpa.shape[:-1],
        vpb.shape[:-1],
        vpc.shape[:-1],
    )

    vpa = vpa.expand(batch_shapes + (3,))
    vpb = vpb.expand(batch_shapes + (3,))
    vpc = vpc.expand(batch_shapes + (3,))

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
        batch_shapes + (4, 4), dtype=utils.FLOAT, device=device)

    normalized_Ts = torch.empty(
        batch_shapes + (4, 4), dtype=utils.FLOAT, device=device)

    axis_x = Ts[..., :3, 0] = f1 * cos_t + f2 * sin_t
    axis_y = Ts[..., :3, 1] = f2 * cos_t - f1 * sin_t
    axis_z = Ts[..., :3, 2] = torch.linalg.cross(axis_x, axis_y)

    z_norm = utils.EPS + utils.VectorNorm(axis_z)

    normalized_Ts[..., :3, 0] = utils.Normalized(axis_x)
    normalized_Ts[..., :3, 1] = utils.Normalized(axis_y)
    normalized_Ts[..., :3, 2] = axis_z / z_norm.unsqueeze(-1)

    err = (normalized_Ts[..., :3, 0] *
           normalized_Ts[..., :3, 1]).sum(-1).abs().max()

    assert err <= 2e-3, err

    Ts[..., :3, 3] = s
    Ts[..., 3, :3] = 0
    Ts[..., 3, 3] = 1

    normalized_Ts[..., :3, 3] = s
    normalized_Ts[..., 3, :3] = 0
    normalized_Ts[..., 3, 3] = 1

    ret = FaceCoordResult(
        Ts=Ts,
        normalized_Ts=normalized_Ts,
        face_area=z_norm * 3,
    )

    return ret
