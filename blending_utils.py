import torch
from typeguard import typechecked
import utils

from kin_tree import KinTree
from utils import GetInvRT, MergeRT, DoRT


@typechecked
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

        joint_rs[..., u, :, :], joint_ts[..., u, :] = MergeRT(
            joint_rs[..., p, :, :],
            joint_ts[..., p, :],
            pose_rs[..., u, :, :],
            pose_ts[..., u, :],
        )

    return joint_rs, joint_ts


@typechecked
def LBS(
    *,
    kin_tree: KinTree,

    vertices: torch.Tensor,  # [..., V, D]
    lbs_weights: torch.Tensor,  # [..., V, J]

    binding_pose_rs: torch.Tensor,  # [..., J, D, D]
    binding_pose_ts: torch.Tensor,  # [..., J, D]

    pose_rs: torch.Tensor,  # [..., J, D, D]
    pose_ts: torch.Tensor,  # [..., J, D]
) -> tuple[
    torch.Tensor,  # vertices[..., V, D]

    torch.Tensor,  # binding_joint_rs[..., J, D, D]
    torch.Tensor,  # binding_joint_ts[..., J, D]

    torch.Tensor,  # joint_rs[..., J, D, D]
    torch.Tensor,  # joint_ts[..., J, D]
]:
    J = kin_tree.joints_cnt

    V, D = -1, -2

    V, D = utils.CheckShapes(
        vertices, (..., V, D),
        lbs_weights, (..., V, J),
        binding_pose_rs, (..., J, D, D),
        binding_pose_ts, (..., J, D),
        pose_rs, (..., J, D, D),
        pose_ts, (..., J, D),
    )

    assert binding_pose_rs.dtype == binding_pose_ts.dtype
    assert binding_pose_rs.device == binding_pose_ts.device

    assert pose_rs.dtype == pose_ts.dtype
    assert pose_rs.device == pose_ts.device

    binding_joint_rs, binding_joint_ts = GetJointRTs(
        kin_tree, binding_pose_rs, binding_pose_ts)
    # binding_joint_rs[..., J, D, D]
    # binding_joint_ts[..., J, D]

    inv_binding_joint_rs, inv_binding_joint_ts = GetInvRT(
        binding_joint_rs, binding_joint_ts)
    # inv_binding_joint_rs[..., J, D, D]
    # inv_binding_joint_ts[..., J, D]

    joint_rs, joint_ts = GetJointRTs(kin_tree, pose_rs, pose_ts)
    # joint_rs[..., J, D, D]
    # joint_ts[..., J, D]

    m_rs, m_ts = MergeRT(
        joint_rs, joint_ts, inv_binding_joint_rs, inv_binding_joint_ts)
    # m_rs[..., J, D, D]
    # m_ts[..., J, D]

    v_dtype = vertices.dtype
    v_device = vertices.device

    m_rs = m_rs.to(dtype=v_dtype, device=v_device)
    m_ts = m_ts.to(dtype=v_dtype, device=v_device)

    lbs_weights = lbs_weights.to(dtype=v_dtype, device=v_device)

    ret_a = torch.einsum(
        "...vj,...jdk,...vk->...vd",
        lbs_weights,
        m_rs,
        vertices,
    )  # [..., V, D]

    ret_b = torch.einsum(
        "...vj,...jd->...vd",
        lbs_weights,
        m_ts,
    )  # [..., V, D]

    return ret_a + ret_b, binding_joint_rs, binding_joint_ts, joint_rs, joint_ts
