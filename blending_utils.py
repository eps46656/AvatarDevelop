import dataclasses
import typing

import torch
from beartype import beartype

from . import kin_utils, utils


@beartype
@dataclasses.dataclass
class LBSResult:
    blended_vert_pos: typing.Optional[torch.Tensor]  # [..., V, D]
    blended_vert_dir: typing.Optional[torch.Tensor]  # [..., V, D]

    binding_joint_Ts: torch.Tensor  # [..., J, D+1, D+1]

    inv_binding_joint_Ts: torch.Tensor  # [..., J, D+1, D+1]

    target_joint_Ts: torch.Tensor  # [..., J, D+1, D+1]

    del_joint_Ts: torch.Tensor  # [..., J, D+1, D+1]


@beartype
def lbs(
    *,
    kin_tree: kin_utils.KinTree,

    lbs_weights: torch.Tensor,  # [..., V, J]

    vert_pos: typing.Optional[torch.Tensor] = None,  # [..., V, D]
    vert_dir: typing.Optional[torch.Tensor] = None,  # [..., V, D]

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

    device = utils.check_devices(
        lbs_weights,
        vert_pos,
        vert_dir,
        binding_pose_rs,
        binding_pose_ts,
        target_pose_rs,
        target_pose_ts,
    )

    dtype = utils.promote_dtypes(
        lbs_weights,
        vert_pos,
        vert_dir,
        binding_pose_rs,
        binding_pose_ts,
        target_pose_rs,
        target_pose_ts,
    )

    dd = (device, dtype)

    if vert_pos is not None:
        utils.check_shapes(vert_pos, (..., V, D))
        assert vert_pos.isfinite().all()
        vert_pos = vert_pos.to(*dd)

    if vert_dir is not None:
        utils.check_shapes(vert_dir, (..., V, D))
        assert vert_dir.isfinite().all()
        vert_dir = vert_dir.to(*dd)

    binding_joint_Ts = kin_tree.get_joint_rts(binding_pose_rs, binding_pose_ts)
    # binding_joint_rs[..., J, D+1, D+1]

    inv_binding_joint_Ts = binding_joint_Ts.inverse()
    # inv_binding_joint_rs[..., J, D+1, D+1]

    target_joint_Ts = kin_tree.get_joint_rts(target_pose_rs, target_pose_ts)
    # target_joint_rs[..., J, D+1, D+1]

    del_joint_Ts = target_joint_Ts @ inv_binding_joint_Ts
    # del_joint_rs[..., J, D+1, D+1]

    del_joint_Ts = del_joint_Ts.to(*dd)

    assert del_joint_Ts.isfinite().all()

    lbs_weights = lbs_weights.to(*dd)

    if vert_pos is None:
        ret_vp = None
    else:
        ret_vp_a = torch.einsum(
            "...vj,...jdk,...vk->...vd",
            lbs_weights,
            del_joint_Ts[..., :D, :D],
            vert_pos,
        )  # [..., V, D]

        ret_vp_b = torch.einsum(
            "...vj,...jd->...vd",
            lbs_weights,
            del_joint_Ts[..., :D, D],
        )  # [..., V, D]

        ret_vp = ret_vp_a + ret_vp_b

    if vert_dir is None:
        ret_vds = None
    else:
        ret_vds = torch.einsum(
            "...vj,...jdk,...vk->...vd",
            lbs_weights,
            del_joint_Ts[..., :D, :D],
            vert_dir,
        )  # [..., V, D]

    return LBSResult(
        blended_vert_pos=ret_vp,
        blended_vert_dir=ret_vds,

        binding_joint_Ts=binding_joint_Ts,

        inv_binding_joint_Ts=inv_binding_joint_Ts,

        target_joint_Ts=target_joint_Ts,

        del_joint_Ts=del_joint_Ts,
    )
