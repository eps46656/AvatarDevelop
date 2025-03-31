import dataclasses
import typing

import torch
from beartype import beartype

from . import kin_utils, utils


@beartype
@dataclasses.dataclass
class LBSResult:
    blended_vertex_positions: typing.Optional[torch.Tensor]  # [..., V, D]
    blended_vertex_directions: typing.Optional[torch.Tensor]  # [..., V, D]

    binding_joint_Ts: torch.Tensor  # [..., J, D+1, D+1]

    inv_binding_joint_Ts: torch.Tensor  # [..., J, D+1, D+1]

    target_joint_Ts: torch.Tensor  # [..., J, D+1, D+1]

    del_joint_Ts: torch.Tensor  # [..., J, D+1, D+1]


@beartype
def lbs(
    *,
    kin_tree: kin_utils.KinTree,

    lbs_weights: torch.Tensor,  # [..., V, J]

    vertex_positions: typing.Optional[torch.Tensor] = None,  # [..., V, D]
    vertex_directions: typing.Optional[torch.Tensor] = None,  # [..., V, D]

    binding_pose_rs: torch.Tensor,  # [..., J, D, D]
    binding_pose_ts: torch.Tensor,  # [..., J, D]

    target_pose_rs: torch.Tensor,  # [..., J, D, D]
    target_pose_ts: torch.Tensor,  # [..., J, D]
) -> LBSResult:
    J = kin_tree.joints_cnt

    V, D = -1, -2

    V, D = utils.check_shapes(
        lbs_weights, (..., V, J),
        binding_pose_rs, (..., J, D, D),
        binding_pose_ts, (..., J, D),
        target_pose_rs, (..., J, D, D),
        target_pose_ts, (..., J, D),
    )

    assert lbs_weights.isfinite().all()
    assert binding_pose_rs.isfinite().all()
    assert binding_pose_ts.isfinite().all()
    assert target_pose_rs.isfinite().all()
    assert target_pose_ts.isfinite().all()

    if vertex_positions is not None:
        utils.check_shapes(vertex_positions, (..., V, D))
        assert vertex_positions.isfinite().all()

    if vertex_directions is not None:
        utils.check_shapes(vertex_directions, (..., V, D))
        assert vertex_directions.isfinite().all()

    device = lbs_weights.device

    binding_joint_Ts = kin_tree.get_joint_rts(binding_pose_rs, binding_pose_ts)
    # binding_joint_rs[..., J, D+1, D+1]

    inv_binding_joint_Ts = binding_joint_Ts.inverse()
    # inv_binding_joint_rs[..., J, D+1, D+1]

    target_joint_Ts = kin_tree.get_joint_rts(target_pose_rs, target_pose_ts)
    # target_joint_rs[..., J, D+1, D+1]

    del_joint_Ts = target_joint_Ts @ inv_binding_joint_Ts
    # del_joint_rs[..., J, D+1, D+1]

    del_joint_Ts = del_joint_Ts.to(device=device)

    assert del_joint_Ts.isfinite().all()

    lbs_weights = lbs_weights.to(device=device)

    if vertex_positions is None:
        ret_vps = None
    else:
        ret_vps_a = torch.einsum(
            "...vj,...jdk,...vk->...vd",
            lbs_weights,
            del_joint_Ts[..., :D, :D],
            vertex_positions,
        )  # [..., V, D]

        ret_vps_b = torch.einsum(
            "...vj,...jd->...vd",
            lbs_weights,
            del_joint_Ts[..., :D, D],
        )  # [..., V, D]

        ret_vps = ret_vps_a + ret_vps_b

    if vertex_directions is None:
        ret_vds = None
    else:
        ret_vds = torch.einsum(
            "...vj,...jdk,...vk->...vd",
            lbs_weights,
            del_joint_Ts[..., :D, :D],
            vertex_directions,
        )  # [..., V, D]

    return LBSResult(
        blended_vertex_positions=ret_vps,
        blended_vertex_directions=ret_vds,

        binding_joint_Ts=binding_joint_Ts,

        inv_binding_joint_Ts=inv_binding_joint_Ts,

        target_joint_Ts=target_joint_Ts,

        del_joint_Ts=del_joint_Ts,
    )
