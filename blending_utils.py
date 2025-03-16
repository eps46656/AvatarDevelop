import dataclasses
import typing

import torch
from beartype import beartype

import utils
from kin_tree import KinTree


@beartype
def GetJointRTs(
    kin_tree: KinTree,
    pose_rs: torch.Tensor,  # [..., J, D, D]
    pose_ts: torch.Tensor,  # [..., J, D]
) -> tuple[
    torch.Tensor,  # joint_rs[..., J, D, D]
    torch.Tensor,  # joint_ts[..., J, D]
]:
    J = kin_tree.joints_cnt

    D, = utils.CheckShapes(
        pose_rs, (..., J, -1, -1),
        pose_ts, (..., J, -1),
    )

    joint_rs = torch.empty_like(pose_rs)
    joint_rs[..., kin_tree.root, :, :] = pose_rs[..., kin_tree.root, :, :]
    # [..., D, D]

    joint_ts = torch.empty_like(pose_ts)
    joint_ts[..., kin_tree.root, :] = pose_ts[..., kin_tree.root, :]
    # [..., D]

    for u in kin_tree.joints_tp[1:]:
        p = kin_tree.parents[u]

        joint_rs[..., u, :, :], joint_ts[..., u, :] = utils.MergeRT(
            joint_rs[..., p, :, :],
            joint_ts[..., p, :],
            pose_rs[..., u, :, :],
            pose_ts[..., u, :],
        )

    return joint_rs, joint_ts


@dataclasses.dataclass
class LBSResult:
    blended_vertex_positions: typing.Optional[torch.Tensor]  # [..., V, D]
    blended_vertex_directions: typing.Optional[torch.Tensor]  # [..., V, D]

    binding_joint_rs: torch.Tensor  # [..., J, D, D]
    binding_joint_ts: torch.Tensor  # [..., J, D]

    inv_binding_joint_rs: torch.Tensor  # [..., J, D, D]
    inv_binding_joint_ts: torch.Tensor  # [..., J, D]

    target_joint_rs: torch.Tensor  # [..., J, D, D]
    target_joint_ts: torch.Tensor  # [..., J, D]

    del_joint_rs: torch.Tensor  # [..., J, D, D]
    del_joint_ts: torch.Tensor  # [..., J, D]


@beartype
def LBS(
    *,
    kin_tree: KinTree,

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

    V, D = utils.CheckShapes(
        lbs_weights, (..., V, J),
        binding_pose_rs, (..., J, D, D),
        binding_pose_ts, (..., J, D),
        target_pose_rs, (..., J, D, D),
        target_pose_ts, (..., J, D),
    )

    if vertex_positions is not None:
        utils.CheckShapes(vertex_positions, (..., V, D))

    if vertex_directions is not None:
        utils.CheckShapes(vertex_directions, (..., V, D))

    device = lbs_weights.device

    binding_joint_rs, binding_joint_ts = GetJointRTs(
        kin_tree, binding_pose_rs, binding_pose_ts)
    # binding_joint_rs[..., J, D, D]
    # binding_joint_ts[..., J, D]

    inv_binding_joint_rs, inv_binding_joint_ts = utils.GetInvRT(
        binding_joint_rs, binding_joint_ts)
    # inv_binding_joint_rs[..., J, D, D]
    # inv_binding_joint_ts[..., J, D]

    target_joint_rs, target_joint_ts = GetJointRTs(
        kin_tree, target_pose_rs, target_pose_ts)
    # target_joint_rs[..., J, D, D]
    # target_joint_ts[..., J, D]

    del_joint_rs, del_joint_ts = utils.MergeRT(
        target_joint_rs, target_joint_ts,
        inv_binding_joint_rs, inv_binding_joint_ts)
    # del_joint_rs[..., J, D, D]
    # del_joint_ts[..., J, D]

    del_joint_rs = del_joint_rs.to(device=device)
    del_joint_ts = del_joint_ts.to(device=device)

    lbs_weights = lbs_weights.to(device=device)

    if vertex_positions is None:
        ret_vps = None
    else:
        ret_vps_a = torch.einsum(
            "...vj,...jdk,...vk->...vd",
            lbs_weights,
            del_joint_rs,
            vertex_positions,
        )  # [..., V, D]

        ret_vps_b = torch.einsum(
            "...vj,...jd->...vd",
            lbs_weights,
            del_joint_ts,
        )  # [..., V, D]

        ret_vps = ret_vps_a + ret_vps_b

    if vertex_directions is None:
        ret_vds = None
    else:
        ret_vds = torch.einsum(
            "...vj,...jdk,...vk->...vd",
            lbs_weights,
            del_joint_rs,
            vertex_directions,
        )  # [..., V, D]

    return LBSResult(
        blended_vertex_positions=ret_vps,
        blended_vertex_directions=ret_vds,

        binding_joint_rs=binding_joint_rs,
        binding_joint_ts=binding_joint_ts,

        inv_binding_joint_rs=inv_binding_joint_rs,
        inv_binding_joint_ts=inv_binding_joint_ts,

        target_joint_rs=target_joint_rs,
        target_joint_ts=target_joint_ts,

        del_joint_rs=del_joint_rs,
        del_joint_ts=del_joint_ts,
    )
