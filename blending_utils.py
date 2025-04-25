from __future__ import annotations

import dataclasses
import typing

import torch
from beartype import beartype

from . import kin_utils, utils


@beartype
class LBSOperator:
    def __init__(
        self,
        binding_pose_r: typing.Optional[torch.Tensor],  # [..., J, D, D]
        binding_pose_t: typing.Optional[torch.Tensor],  # [..., J, D]

        target_pose_r: typing.Optional[torch.Tensor],  # [..., J, D, D]
        target_pose_t: typing.Optional[torch.Tensor],  # [..., J, D]

        binding_joint_T: typing.Optional[torch.Tensor],
        # [..., J, D + 1, D + 1]

        inv_binding_joint_T: typing.Optional[torch.Tensor],
        # [..., J, D + 1, D + 1]

        target_joint_T: typing.Optional[torch.Tensor],
        # [..., J, D + 1, D + 1]

        del_joint_T: torch.Tensor,  # [..., J, D + 1, D + 1]
    ):
        J, D, D_ = -1, -2, -3

        J, D, D_ = utils.check_shapes(
            binding_pose_r, (..., J, D, D),
            binding_pose_t, (..., J, D),

            target_pose_r, (..., J, D, D),
            target_pose_t, (..., J, D),

            binding_joint_T, (..., J, D_, D_),
            inv_binding_joint_T, (..., J, D_, D_),

            target_joint_T, (..., J, D_, D_),

            del_joint_T, (..., J, D_, D_),
        )

        assert D + 1 == D_

        self.binding_pose_r = binding_pose_r
        self.binding_pose_t = binding_pose_t

        self.target_pose_r = target_pose_r
        self.target_pose_t = target_pose_t

        self.binding_joint_T = binding_joint_T
        self.inv_binding_joint_T = inv_binding_joint_T

        self.target_joint_T = target_joint_T

        self.del_joint_T = del_joint_T

    @staticmethod
    def from_binding_and_target(
        kin_tree: kin_utils.KinTree,

        binding_pose_r: torch.Tensor,  # [..., J, D, D]
        binding_pose_t: torch.Tensor,  # [..., J, D]

        target_pose_r: torch.Tensor,  # [..., J, D, D]
        target_pose_t: torch.Tensor,  # [..., J, D]
    ) -> LBSOperator:
        J, D = kin_tree.joints_cnt, -1

        D = utils.check_shapes(
            binding_pose_r, (..., J, D, D),
            binding_pose_t, (..., J, D),

            target_pose_r, (..., J, D, D),
            target_pose_t, (..., J, D),
        )

        assert binding_pose_r.isfinite().all()
        assert binding_pose_t.isfinite().all()

        assert target_pose_r.isfinite().all()
        assert target_pose_t.isfinite().all()

        device = utils.check_devices(
            binding_pose_r, binding_pose_t,
            target_pose_r, target_pose_t,
        )

        dtype = utils.promote_dtypes(
            binding_pose_r, binding_pose_t,
            target_pose_r, target_pose_t,
        )

        dd = (device, dtype)

        binding_pose_r = binding_pose_r.to(*dd)  # [..., J, D, D]
        binding_pose_t = binding_pose_t.to(*dd)  # [..., J, D]

        target_pose_r = target_pose_r.to(*dd)  # [..., J, D, D]
        target_pose_t = target_pose_t.to(*dd)  # [..., J, D]

        binding_joint_T = \
            kin_tree.get_joint_rt(binding_pose_r, binding_pose_t)
        # binding_joint_r[..., J, D+1, D+1]

        inv_binding_joint_T = \
            binding_joint_T.inverse()
        # inv_binding_joint_r[..., J, D+1, D+1]

        target_joint_T = kin_tree.get_joint_rt(target_pose_r, target_pose_t)
        # target_joint_r[..., J, D+1, D+1]

        del_joint_T = target_joint_T @ inv_binding_joint_T
        # del_joint_r[..., J, D+1, D+1]

        return LBSOperator(
            binding_pose_r=binding_pose_r,
            binding_pose_t=binding_pose_t,

            target_pose_r=target_pose_r,
            target_pose_t=target_pose_t,

            binding_joint_T=binding_joint_T,
            inv_binding_joint_T=inv_binding_joint_T,

            target_joint_T=target_joint_T,

            del_joint_T=del_joint_T.to(*dd)
        )

    @property
    def shape(self) -> torch.Size:
        return self.del_joint_T.shape[:-3]

    @property
    def dtype(self) -> torch.dtype:
        return self.del_joint_T.dtype

    @property
    def device(self) -> torch.device:
        return self.del_joint_T.device

    def to(self, *args, **kwargs) -> LBSOperator:
        def f(x):
            return None if x is None else x.to(*args, **kwargs)

        return LBSOperator(
            binding_pose_r=f(self.binding_pose_r),
            binding_pose_t=f(self.binding_pose_t),

            target_pose_r=f(self.target_pose_r),
            target_pose_t=f(self.target_pose_t),

            binding_joint_T=f(self.binding_joint_T),
            inv_binding_joint_T=f(self.inv_binding_joint_T),

            target_joint_T=f(self.target_joint_T),

            del_joint_T=f(self.del_joint_T)
        )

    def blend(
        self,
        vec: torch.Tensor,  # [..., V, D]
        lbs_weight: torch.Tensor,  # [..., V, J]
        calc_trans: bool,
        calc_linear_part: bool,
        calc_const_part: bool,
    ) -> tuple[
        typing.Optional[torch.Tensor],  # ret_trans[..., V, D + 1, D + 1]
        typing.Optional[torch.Tensor],  # ret_linear[..., V, D]
        typing.Optional[torch.Tensor],  # ret_const[..., V, D]
    ]:
        J, V, D, D_ = -1, -2, -3, -4

        J, V, D, D_ = utils.check_shapes(
            self.del_joint_T, (..., J, D_, D_),
            vec, (..., V, D),
            lbs_weight, (..., V, J),
        )

        assert D + 1 == D_

        utils.check_devices(self.del_joint_T, vec, lbs_weight)

        dtype = utils.promote_dtypes(
            self.del_joint_T,
            vec,
            lbs_weight,
        )

        cur_del_joint_T = self.del_joint_T.to(dtype)
        # [..., J, D+1, D+1]

        lbs_weight = lbs_weight.to(dtype)
        # [..., V, J]

        vec = vec.to(dtype)
        # [..., V, D]

        if not calc_trans:
            ret_trans = None
        else:
            ret_trans = torch.empty((
                utils.broadcast_shapes(
                    lbs_weight.shape[:-2], cur_del_joint_T.shape[:-3]
                ), V, D + 1, D + 1,
            ), dtype=dtype, device=self.device)
            # [..., V, D + 1, D + 1]

            ret_trans[..., :, :-1, :] = torch.einsum(
                "...vj,...jab->...vab",
                lbs_weight,  # [..., V, J]
                cur_del_joint_T[..., :, :-1, :],  # [..., J, D, D + 1]
            )  # [..., V, D, D + 1]

            ret_trans[..., :, -1, :-1] = 0

            ret_trans[..., :, -1, -1] = 1

        if not calc_linear_part:
            ret_linear = None
        else:
            ret_linear = torch.einsum(
                "...vj,...jdk,...vk->...vd",
                lbs_weight,  # [..., V, J]
                cur_del_joint_T[..., :-1, :-1],  # [..., J, D, D]
                vec,
            )  # [..., V, D]

        if not calc_const_part:
            ret_const = None
        else:
            ret_const = torch.einsum(
                "...vj,...jd->...vd",
                lbs_weight,
                cur_del_joint_T[..., :-1, -1],
            )  # [..., V, D]

        return ret_trans, ret_linear, ret_const

    def blend_pos(
        self,
        pos: torch.Tensor,  # [..., V, D]
        lbs_weight: torch.Tensor,  # [..., V, J]
    ) -> torch.Tensor:  # [..., V, D]
        ret_trans, ret_linear, ret_const = self.blend(
            vec=pos,
            lbs_weight=lbs_weight,
            calc_trans=False,
            calc_linear_part=True,
            calc_const_part=True,
        )

        return ret_linear + ret_const

    def blend_dir(
        self,
        dir: torch.Tensor,  # [..., V, D]
        lbs_weight: torch.Tensor,  # [..., V, J]
    ) -> torch.Tensor:  # [..., V, D]
        ret_trans, ret_linear, ret_const = self.blend(
            vec=dir,
            lbs_weight=lbs_weight,
            calc_trans=False,
            calc_linear_part=True,
            calc_const_part=False,
        )

        return ret_linear


@beartype
@dataclasses.dataclass
class LBSResult:
    lbs_opr: LBSOperator

    blended_vert_pos: typing.Optional[torch.Tensor]  # [..., V, D]
    blended_vert_dir: typing.Optional[torch.Tensor]  # [..., V, D]


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
    lbs_opr = LBSOperator.from_binding_and_target(
        kin_tree,
        binding_pose_r,
        binding_pose_t,
        target_pose_r,
        target_pose_t,
    )

    ret_vp = None if vert_pos is None else \
        lbs_opr.blend_pos(vert_pos, lbs_weight)

    ret_vd = None if vert_dir is None else \
        lbs_opr.blend_dir(vert_dir, lbs_weight)

    return LBSResult(
        lbs_opr=lbs_opr,
        blended_vert_pos=ret_vp,
        blended_vert_dir=ret_vd,
    )
