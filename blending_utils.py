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

    binding_joint_T: torch.Tensor  # [..., J, D+1, D+1]

    inv_binding_joint_T: torch.Tensor  # [..., J, D+1, D+1]

    target_joint_T: torch.Tensor  # [..., J, D+1, D+1]

    del_joint_T: torch.Tensor  # [..., J, D+1, D+1]


@beartype
def lbs(
    *,
    kin_tree: kin_utils.KinTree,

    lbs_weight: torch.Tensor,  # [..., V, J]

    vert_pos: typing.Optional[torch.Tensor] = None,  # [..., V, D]
    vert_dir: typing.Optional[torch.Tensor] = None,  # [..., V, D]

    binding_pose_r: torch.Tensor,  # [..., J, D, D]
    binding_pose_t: torch.Tensor,  # [..., J, D]

    target_pose_r: torch.Tensor,  # [..., J, D, D]
    target_pose_t: torch.Tensor,  # [..., J, D]
) -> LBSResult:
    J = kin_tree.joints_cnt

    V, D = -1, -2

    V, D = utils.check_shapes(
        lbs_weight, (..., V, J),
        binding_pose_r, (..., J, D, D),
        binding_pose_t, (..., J, D),
        target_pose_r, (..., J, D, D),
        target_pose_t, (..., J, D),
    )

    assert lbs_weight.isfinite().all()
    assert binding_pose_r.isfinite().all()
    assert binding_pose_t.isfinite().all()
    assert target_pose_r.isfinite().all()
    assert target_pose_t.isfinite().all()

    device = utils.check_devices(
        lbs_weight,
        vert_pos,
        vert_dir,
        binding_pose_r,
        binding_pose_t,
        target_pose_r,
        target_pose_t,
    )

    dtype = utils.promote_dtypes(
        lbs_weight,
        vert_pos,
        vert_dir,
        binding_pose_r,
        binding_pose_t,
        target_pose_r,
        target_pose_t,
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

    binding_joint_T = kin_tree.get_joint_rt(binding_pose_r, binding_pose_t)
    # binding_joint_r[..., J, D+1, D+1]

    inv_binding_joint_T = binding_joint_T.inverse()
    # inv_binding_joint_r[..., J, D+1, D+1]

    target_joint_T = kin_tree.get_joint_rt(target_pose_r, target_pose_t)
    # target_joint_r[..., J, D+1, D+1]

    del_joint_T = target_joint_T @ inv_binding_joint_T
    # del_joint_r[..., J, D+1, D+1]

    del_joint_T = del_joint_T.to(*dd)

    assert del_joint_T.isfinite().all()

    lbs_weight = lbs_weight.to(*dd)

    if vert_pos is None:
        ret_vp = None
    else:
        ret_vp_a = torch.einsum(
            "...vj,...jdk,...vk->...vd",
            lbs_weight,
            del_joint_T[..., :D, :D],
            vert_pos,
        )  # [..., V, D]

        ret_vp_b = torch.einsum(
            "...vj,...jd->...vd",
            lbs_weight,
            del_joint_T[..., :D, D],
        )  # [..., V, D]

        ret_vp = ret_vp_a + ret_vp_b

    if vert_dir is None:
        ret_vds = None
    else:
        ret_vds = torch.einsum(
            "...vj,...jdk,...vk->...vd",
            lbs_weight,
            del_joint_T[..., :D, :D],
            vert_dir,
        )  # [..., V, D]

    return LBSResult(
        blended_vert_pos=ret_vp,
        blended_vert_dir=ret_vds,

        binding_joint_T=binding_joint_T,

        inv_binding_joint_T=inv_binding_joint_T,

        target_joint_T=target_joint_T,

        del_joint_T=del_joint_T,
    )
