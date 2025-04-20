from __future__ import annotations

import typing

import torch
from beartype import beartype

from .. import blending_utils, utils
from .Model import Model
from .ModelData import ModelData


@beartype
class BlendingParam:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = torch.Size(),

        body_shape: typing.Optional[torch.Tensor] = None,  # [..., BS]
        expr_shape: typing.Optional[torch.Tensor] = None,  # [..., ES]

        global_transl: typing.Optional[torch.Tensor] = None,  # [..., 3]
        global_rot: typing.Optional[torch.Tensor] = None,  # [..., 3]

        body_pose: typing.Optional[torch.Tensor] = None,  # [..., BJ - 1, 3]
        jaw_pose: typing.Optional[torch.Tensor] = None,  # [..., JJ, 3]
        leye_pose: typing.Optional[torch.Tensor] = None,  # [..., EYEJ, 3]
        reye_pose: typing.Optional[torch.Tensor] = None,  # [..., EYEJ, 3]

        lhand_pose: typing.Optional[torch.Tensor] = None,  # [..., HANDJ, 3]
        rhand_pose: typing.Optional[torch.Tensor] = None,  # [..., HANDJ, 3]

        dtype: typing.Optional[torch.dtype] = None,
        device: typing.Optional[torch.device] = None,
    ):
        BS, ES, BODYJ_, JAWJ, EYEJ, HANDJ = -1, -2, -3, -4, -5, -6

        utils.check_shapes(
            body_shape, (..., BS),
            expr_shape, (..., ES),

            global_transl, (..., 3),
            global_rot, (..., 3),

            body_pose, (..., BODYJ_, 3),
            jaw_pose, (..., JAWJ, 3),
            leye_pose, (..., EYEJ, 3),
            reye_pose, (..., EYEJ, 3),
            lhand_pose, (..., HANDJ, 3),
            rhand_pose, (..., HANDJ, 3),
        )

        BODYJ = BODYJ_ + 1

        # ---

        self.shape = utils.broadcast_shapes(
            shape,

            utils.try_get_batch_shape(body_shape, -1),
            utils.try_get_batch_shape(expr_shape, -1),

            utils.try_get_batch_shape(global_transl, -1),
            utils.try_get_batch_shape(global_rot, -1),

            utils.try_get_batch_shape(body_pose, -2),
            utils.try_get_batch_shape(jaw_pose, -2),
            utils.try_get_batch_shape(leye_pose, -2),
            utils.try_get_batch_shape(reye_pose, -2),

            utils.try_get_batch_shape(lhand_pose, -2),
            utils.try_get_batch_shape(rhand_pose, -2),
        )

        def f(obj):
            return None if obj is None else obj.to(device, dtype)

        self.raw_body_shape = f(body_shape)
        self.raw_expr_shape = f(expr_shape)

        self.raw_global_transl = f(global_transl)
        self.raw_global_rot = f(global_rot)

        self.raw_body_pose = f(body_pose)
        self.raw_jaw_pose = f(jaw_pose)
        self.raw_leye_pose = f(leye_pose)
        self.raw_reye_pose = f(reye_pose)

        self.raw_lhand_pose = f(lhand_pose)
        self.raw_rhand_pose = f(rhand_pose)

    @property
    def body_shape(self) -> typing.Optional[torch.Tensor]:
        return utils.try_batch_expand(self.raw_body_shape, self.shape, -1)

    @property
    def expr_shape(self) -> typing.Optional[torch.Tensor]:
        return utils.try_batch_expand(self.raw_expr_shape, self.shape, -1)

    @property
    def global_transl(self) -> typing.Optional[torch.Tensor]:
        return utils.try_batch_expand(self.raw_global_transl, self.shape, -1)

    @property
    def global_rot(self) -> typing.Optional[torch.Tensor]:
        return utils.try_batch_expand(self.raw_global_rot, self.shape, -1)

    @property
    def body_pose(self) -> typing.Optional[torch.Tensor]:
        return utils.try_batch_expand(self.raw_body_pose, self.shape, -2)

    @property
    def jaw_pose(self) -> typing.Optional[torch.Tensor]:
        return utils.try_batch_expand(self.raw_jaw_pose, self.shape, -2)

    @property
    def leye_pose(self) -> typing.Optional[torch.Tensor]:
        return utils.try_batch_expand(self.raw_leye_pose, self.shape, -2)

    @property
    def reye_pose(self) -> typing.Optional[torch.Tensor]:
        return utils.try_batch_expand(self.raw_reye_pose, self.shape, -2)

    @property
    def lhand_pose(self) -> typing.Optional[torch.Tensor]:
        return utils.try_batch_expand(self.raw_lhand_pose, self.shape, -2)

    @property
    def rhand_pose(self) -> typing.Optional[torch.Tensor]:
        return utils.try_batch_expand(self.raw_rhand_pose, self.shape, -2)

    @property
    def device(self):
        return self.body_shape.device

    def __getitem__(self, idx) -> BlendingParam:
        return BlendingParam(
            body_shape=utils.try_batch_indexing(
                self.body_shape, self.shape, -1, idx),
            expr_shape=utils.try_batch_indexing(
                self.expr_shape, self.shape, -1, idx),

            global_transl=utils.try_batch_indexing(
                self.global_transl, self.shape, -1, idx),
            global_rot=utils.try_batch_indexing(
                self.global_rot, self.shape, -1, idx),

            body_pose=utils.try_batch_indexing(
                self.body_pose, self.shape, -2, idx),
            jaw_pose=utils.try_batch_indexing(
                self.jaw_pose, self.shape, -2, idx),
            leye_pose=utils.try_batch_indexing(
                self.leye_pose, self.shape, -2, idx),
            reye_pose=utils.try_batch_indexing(
                self.reye_pose, self.shape, -2, idx),

            lhand_pose=utils.try_batch_indexing(
                self.lhand_pose, self.shape, -2, idx),
            rhand_pose=utils.try_batch_indexing(
                self.rhand_pose, self.shape, -2, idx),
        )

    def expand(self, shape: tuple[int, ...]) -> BlendingParam:
        return BlendingParam(
            shape=shape,

            body_shape=self.body_shape,
            expr_shape=self.expr_shape,

            global_transl=self.global_transl,
            global_rot=self.global_rot,

            body_pose=self.body_pose,
            jaw_pose=self.jaw_pose,
            leye_pose=self.leye_pose,
            reye_pose=self.reye_pose,

            lhand_pose=self.lhand_pose,
            rhand_pose=self.rhand_pose,
        )

    def to(self, *args, **kwargs) -> BlendingParam:
        def f(x): return None if x is None else x.to(*args, **kwargs)

        return BlendingParam(
            body_shape=f(self.body_shape),
            expr_shape=f(self.expr_shape),

            global_transl=f(self.global_transl),
            global_rot=f(self.global_rot),

            body_pose=f(self.body_pose),
            jaw_pose=f(self.jaw_pose),
            leye_pose=f(self.leye_pose),
            reye_pose=f(self.reye_pose),

            lhand_pose=f(self.lhand_pose),
            rhand_pose=f(self.rhand_pose),
        )

    def get_poses(self, model_data: ModelData) -> torch.Tensor:
        assert self.global_rot is not None
        assert self.body_pose is not None
        assert self.jaw_pose is not None
        assert self.leye_pose is not None
        assert self.reye_pose is not None
        assert self.lhand_pose is not None
        assert self.rhand_pose is not None

        poses = [
            utils.try_batch_expand(
                self.global_rot, self.shape, -1).unsqueeze(-2),
            utils.try_batch_expand(self.body_pose, self.shape, -2),
            utils.try_batch_expand(self.jaw_pose, self.shape, -2),
            utils.try_batch_expand(self.leye_pose, self.shape, -2),
            utils.try_batch_expand(self.reye_pose, self.shape, -2),
        ]

        if model_data.lhand_pose_mean is not None:
            poses.append(utils.try_batch_expand(
                self.lhand_pose + model_data.lhand_pose_mean, self.shape, -2))

        if model_data.rhand_pose_mean is not None:
            poses.append(utils.try_batch_expand(
                self.rhand_pose + model_data.rhand_pose_mean, self.shape, -2))

        ret = torch.cat(poses, -2)
        # [..., ?, 3]

        return ret

    def combine(self, obj: BlendingParam) -> BlendingParam:
        table = [
            "body_shape",
            "expr_shape",
            "global_transl",
            "global_rot",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
        ]

        ret = dict()

        for field_name in table:
            field_val = getattr(self, field_name)

            ret[field_name] = getattr(obj, field_name) \
                if field_val is None else field_val

        return BlendingParam(**ret)


@beartype
def blending(
    model_data: ModelData,
    blending_param: BlendingParam,
    device: torch.device,
) -> Model:
    vp = model_data.vert_pos

    if blending_param.body_shape is not None:
        vp = vp + torch.einsum("...vxb,...b->...vx",
                               model_data.body_shape_vert_dir,
                               blending_param.body_shape)

    if model_data.expr_shape_vert_dir is not None and \
       blending_param.expr_shape is not None:
        vp = vp + torch.einsum("...vxb,...b->...vx",
                               model_data.expr_shape_vert_dir,
                               blending_param.expr_shape)

    # [..., V, 3]

    binding_joint_ts = model_data.joint_t_mean.clone()

    if blending_param.body_shape is not None:
        binding_joint_ts = binding_joint_ts + torch.einsum(
            "...jxb,...b->...jx",
            model_data.body_shape_joint_dir,
            blending_param.body_shape,
        )

    if model_data.expr_shape_joint_dir is not None and \
            blending_param.expr_shape is not None:
        binding_joint_ts = binding_joint_ts + torch.einsum(
            "...jxb,...b->...jx",
            model_data.expr_shape_joint_dir,
            blending_param.expr_shape,
        )

    # [..., J, 3]

    J = model_data.kin_tree.joints_cnt

    binding_pose_rs = utils.batch_eye(
        (J, 3, 3), dtype=utils.FLOAT, device=device)

    binding_pose_ts = utils.empty_like(binding_joint_ts)
    binding_pose_ts[..., model_data.kin_tree.root,
                    :] = binding_joint_ts[..., model_data.kin_tree.root, :]
    # [..., J, 3]

    for u in model_data.kin_tree.joints_tp[1:]:
        p = model_data.kin_tree.parents[u]

        binding_pose_ts[..., u, :] = binding_joint_ts[...,
                                                      u, :] - binding_joint_ts[..., p, :]

    target_pose_rs = utils.axis_angle_to_rot_mat(
        blending_param.get_poses(model_data), out_shape=(3, 3))
    # [..., J, 3, 3]

    identity = torch.eye(
        3, dtype=target_pose_rs.dtype, device=target_pose_rs.device)

    pose_feature = target_pose_rs[..., 1:, :, :] - identity
    # [..., J - 1, 3, 3]

    pose_feature = pose_feature.view(
        *pose_feature.shape[:-3], (J - 1) * 3 * 3)
    # [..., (J - 1) * 3 * 3]

    vp = vp + torch.einsum(
        "...vxp,...p->...vx",
        model_data.pose_vert_dir,  # [..., V, 3, (J - 1) * 3 * 3]
        pose_feature,  # [..., (J - 1) * 3 * 3]
    )

    lbs_result = blending_utils.lbs(
        kin_tree=model_data.kin_tree,
        lbs_weight=model_data.lbs_weight,

        vert_pos=vp,

        binding_pose_r=binding_pose_rs,
        binding_pose_t=binding_pose_ts,
        target_pose_r=target_pose_rs,
        target_pose_t=binding_pose_ts,
    )

    if blending_param.global_transl is not None:
        lbs_result.target_joint_T[..., :, :3, 3] += \
            blending_param.global_transl.unsqueeze(-2)

        vp = blending_param.global_transl.unsqueeze(-2) + \
            lbs_result.blended_vert_pos

    return Model(
        kin_tree=model_data.kin_tree,

        mesh_graph=model_data.mesh_graph,
        tex_mesh_graph=model_data.tex_mesh_graph,

        vert_pos=vp,

        tex_vert_pos=model_data.tex_vert_pos,

        joint_T=lbs_result.target_joint_T,
    )
