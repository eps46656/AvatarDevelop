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

    assert vpa.isfinite().all()
    assert vpb.isfinite().all()
    assert vpc.isfinite().all()

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

    ab = vpb - vpa
    bc = vpc - vpb
    ca = vpa - vpc

    ab_norm = utils.vector_norm(ab)
    bc_norm = utils.vector_norm(ab)
    ca_norm = utils.vector_norm(ab)

    min_edge_length = min(ab_norm.min(), bc_norm.min(), ca_norm.min())

    axb = torch.linalg.cross(vpa, vpb)
    bxc = torch.linalg.cross(vpb, vpc)
    cxa = torch.linalg.cross(vpc, vpa)

    double_face_area_vec = axb + bxc + cxa
    double_face_area = utils.vector_norm(double_face_area_vec)

    min_double_face_area = double_face_area.min()

    assert 2e-4 <= min_double_face_area, f"{min_double_face_area=}"

    sa = vpa - s

    Ts = torch.empty(
        batch_shape + (4, 4), dtype=utils.FLOAT, device=device)

    Ts[..., :3, 0] = utils.normalized(sa)
    axis_x = Ts[..., :3, 0]

    Ts[..., :3, 2] = double_face_area_vec / double_face_area.unsqueeze(-1)
    axis_z = Ts[..., :3, 2]

    Ts[..., :3, 1] = torch.linalg.cross(axis_z, axis_x)
    axis_y = Ts[..., :3, 1]

    err = (axis_z * axis_x).sum(-1).abs().max()
    assert err <= 2e-3, err

    err = (axis_x * axis_y).sum(-1).abs().max()
    assert err <= 2e-3, err

    err = (axis_y * axis_z).sum(-1).abs().max()
    assert err <= 2e-3, err

    Ts[..., :3, 3] = s
    Ts[..., 3, :3] = 0
    Ts[..., 3, 3] = 1

    ret = FaceCoordResult(
        Ts=Ts,
        face_area=double_face_area * 0.5,
    )

    return ret
