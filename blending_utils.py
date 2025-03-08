from typeguard import typechecked

import torch

from kin_tree import KinTree

from utils import MergeRT, GetInvRT, DoRT


@typechecked
def GetJointRTs(
    kin_tree: KinTree,
    pose_rs: torch.Tensor,  # [..., J, D, D]
    pose_ts: torch.Tensor,  # [..., J, D],
) -> tuple[torch.Tensor, torch.Tensor]:  # [..., J, D, D], [..., J, D]
    assert 3 <= len(pose_rs.shape)
    assert 2 <= len(pose_ts.shape)

    J, D, _ = pose_rs.shape[-3:]

    assert kin_tree.joints_cnt == J
    assert pose_rs.shape[-3:] == (J, D, D)
    assert pose_ts.shape[-2:] == (J, D)

    abs_joint_rs = torch.empty_like(pose_rs)
    abs_joint_rs[..., kin_tree.root, :, :] = pose_rs[..., kin_tree.root, :, :]
    # [..., D, D]

    abs_joint_ts = torch.empty_like(pose_ts)
    abs_joint_ts[..., kin_tree.root, :] = pose_ts[..., kin_tree.root, :]
    # [..., D]

    for u in kin_tree.joints_tp:
        parent = kin_tree.parents[u]

        if parent == -1:
            continue

        abs_joint_rs[..., u, :, :], abs_joint_ts[..., u, :] = MergeRT(
            abs_joint_rs[..., parent, :, :],
            abs_joint_ts[..., parent, :],
            pose_rs[..., u, :, :],
            pose_ts[..., u, :],
        )

    return abs_joint_rs, abs_joint_ts


@typechecked
def LBS(
    kin_tree: KinTree,
    vertices: torch.Tensor,  # [..., V, D]
    lbs_weights: torch.Tensor,  # [..., V, J]
    binding_pose_rs: torch.Tensor,  # [..., J, D, D]
    binding_pose_ts: torch.Tensor,  # [..., J, D]
    pose_rs: torch.Tensor,  # [..., J, D, D]
    pose_ts: torch.Tensor,  # [..., J, D]
) -> torch.Tensor:  # [..., V, D]
    assert 2 <= len(vertices.shape)
    assert 2 <= len(lbs_weights.shape)
    assert 3 <= len(binding_pose_rs.shape)
    assert 2 <= len(binding_pose_ts.shape)
    assert 3 <= len(pose_rs.shape)
    assert 2 <= len(pose_ts.shape)

    J = kin_tree.joints_cnt
    V, D = vertices.shape[-2:]

    assert kin_tree.joints_cnt == J
    assert vertices.shape[-2:] == (V, D)
    assert lbs_weights.shape[-2:] == (V, J)
    assert binding_pose_rs.shape[-3:] == (J, D, D)
    assert binding_pose_ts.shape[-2:] == (J, D)
    assert pose_rs.shape[-3:] == (J, D, D)
    assert pose_ts.shape[-2:] == (J, D)

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

    vs = DoRT(
        m_rs[..., None, :, :, :].to(dtype=v_dtype, device=v_device),
        m_ts[..., None, :, :].to(dtype=v_dtype, device=v_device),
        vertices[..., :, None, :]
    )

    ret = torch.einsum(
        "...vjd,...vj->...vd",
        vs,
        lbs_weights.to(dtype=v_dtype, device=v_device)
    )  # [..., V, D]

    return ret
