import dataclasses
import typing

import torch
from beartype import beartype

from .. import blending_utils, utils
from .Model import Model
from .ModelData import ModelData


@beartype
class BlendingParam:
    batch_dim_table = {
        "body_shapes": -1,
        "expr_shapes": -1,
        "global_transl": -1,
        "global_rot": -1,
        "body_poses": -2,
        "jaw_poses": -2,
        "leye_poses": -2,
        "reye_poses": -2,
        "lhand_poses": -2,
        "rhand_poses": -2,
    }

    def __init__(
        self,
        *,
        body_shapes: typing.Optional[torch.Tensor] = None,  # [..., BS]
        expr_shapes: typing.Optional[torch.Tensor] = None,  # [..., ES]

        global_transl: typing.Optional[torch.Tensor] = None,  # [..., 3]
        global_rot: typing.Optional[torch.Tensor] = None,  # [..., 3]

        body_poses: typing.Optional[torch.Tensor] = None,  # [..., BJ - 1, 3]
        jaw_poses: typing.Optional[torch.Tensor] = None,  # [..., JJ, 3]
        leye_poses: typing.Optional[torch.Tensor] = None,  # [..., EYEJ, 3]
        reye_poses: typing.Optional[torch.Tensor] = None,  # [..., EYEJ, 3]

        lhand_poses: typing.Optional[torch.Tensor] = None,  # [..., HANDJ, 3]
        rhand_poses: typing.Optional[torch.Tensor] = None,  # [..., HANDJ, 3]

        blending_vert_nor: bool = False,
    ):
        if body_shapes is not None:
            BS = utils.check_shapes(body_shapes, (..., -1))

        if expr_shapes is not None:
            ES = utils.check_shapes(expr_shapes, (..., -1))

        if global_transl is not None:
            utils.check_shapes(global_transl, (..., 3))

        if global_rot is not None:
            utils.check_shapes(global_rot, (..., 3))

        if body_poses is not None:
            BJ = utils.check_shapes(body_poses, (..., -1, 3)) + 1

        if jaw_poses is not None:
            JJ = utils.check_shapes(jaw_poses, (..., -1, 3))

        EYEJ = -1

        if leye_poses is not None:
            EYEJ = utils.check_shapes(leye_poses, (..., EYEJ, 3))

        if reye_poses is not None:
            EYEJ = utils.check_shapes(reye_poses, (..., EYEJ, 3))

        HANDJ = -1

        if lhand_poses is not None:
            HANDJ = utils.check_shapes(lhand_poses, (..., HANDJ, 3))

        if rhand_poses is not None:
            HANDJ = utils.check_shapes(rhand_poses, (..., HANDJ, 3))

        # ---

        self.body_shapes = body_shapes
        self.expr_shapes = expr_shapes

        self.global_transl = global_transl
        self.global_rot = global_rot

        self.body_poses = body_poses
        self.jaw_poses = jaw_poses
        self.leye_poses = leye_poses
        self.reye_poses = reye_poses

        self.lhand_poses = lhand_poses
        self.rhand_poses = rhand_poses

        self.blending_vert_nor = blending_vert_nor

    def check(
        self,
        model_data: ModelData,
        single_batch: bool,
    ) -> None:
        BS = model_data.body_shapes_cnt
        ES = model_data.expr_shapes_cnt

        BJ = model_data.body_joints_cnt
        JJ = model_data.jaw_joints_cnt
        EYEJ = model_data.eye_joints_cnt
        HANDJ = model_data.hand_joints_cnt

        tensor_shape_constraints = {
            "body_shapes": (BS,),
            "expr_shapes": (ES,),
            "global_transl": (3,),
            "global_rot": (3,),
            "body_poses": (BJ - 1, 3),
            "jaw_poses": (JJ, 3),
            "leye_poses": (EYEJ, 3),
            "reye_poses": (EYEJ, 3),
            "lhand_poses": (HANDJ, 3),
            "rhand_poses": (HANDJ, 3),
        }

        for field_name, shape_constraint in tensor_shape_constraints.items():
            value = getattr(self, field_name)

            if value is None:
                continue

            assert isinstance(value, torch.Tensor)

            if single_batch:
                assert value.shape == shape_constraint
            else:
                assert value.shape[-len(shape_constraint):] == shape_constraint

    @property
    def shape(self) -> tuple[int, ...]:
        return utils.broadcast_shapes(
            utils.try_get_batch_shape(self.body_shapes, -1),
            utils.try_get_batch_shape(self.expr_shapes, -1),
            utils.try_get_batch_shape(self.global_transl, -1),
            utils.try_get_batch_shape(self.global_rot, -1),
            utils.try_get_batch_shape(self.body_poses, -2),
            utils.try_get_batch_shape(self.jaw_poses, -2),
            utils.try_get_batch_shape(self.leye_poses, -2),
            utils.try_get_batch_shape(self.reye_poses, -2),
            utils.try_get_batch_shape(self.lhand_poses, -2),
            utils.try_get_batch_shape(self.rhand_poses, -2),
        )

    @property
    def device(self):
        return self.body_shapes.device

    def to(self, *args, **kwargs) -> typing.Self:
        d = dict()

        for key in BlendingParam.batch_dim_table.keys():
            cur_val = getattr(self, key)
            d[key] = None if cur_val is None else cur_val.to(*args, **kwargs)

        return BlendingParam(
            **d, blending_vert_nor=self.blending_vert_nor)

    def expand(self, shape) -> typing.Self:
        shape = tuple(shape)

        d = dict()

        for key, val in BlendingParam.batch_dim_table.items():
            d[key] = utils.try_batch_expand(getattr(self, key), shape, val)

        return BlendingParam(
            **d, blending_vert_nor=self.blending_vert_nor)

    def flatten(self) -> typing.Self:
        batch_shape = self.shape

        d = dict()

        for key, val in BlendingParam.batch_dim_table.items():
            cur_val = getattr(self, key)
            d[key] = None if cur_val is None else utils.try_batch_expand(
                self.body_shapes, batch_shape, val).flatten(end_dim=val)

        return BlendingParam(
            **d, blending_vert_nor=self.blending_vert_nor)

    def __getitem__(self, idx):
        batch_shape = self.shape

        d = dict()

        for key, val in BlendingParam.batch_dim_table.items():
            cur_val = getattr(self, key)

            d[key] = None if cur_val is None else \
                cur_val.expand(batch_shape + cur_val.shape[val:])[idx]

        return BlendingParam(
            **d, blending_vert_nor=self.blending_vert_nor)

    def get_poses(self, model_data: ModelData) -> torch.Tensor:
        self.check(model_data, False)

        assert self.global_rot is not None
        assert self.body_poses is not None
        assert self.jaw_poses is not None
        assert self.leye_poses is not None
        assert self.reye_poses is not None
        assert self.lhand_poses is not None
        assert self.rhand_poses is not None

        lhand_en = \
            self.lhand_poses is not None and \
            model_data.lhand_poses_mean is not None

        rhand_en = \
            self.rhand_poses is not None and \
            model_data.rhand_poses_mean is not None

        if lhand_en:
            lhand_poses = self.lhand_poses + model_data.lhand_poses_mean

        if rhand_en:
            rhand_poses = self.rhand_poses + model_data.rhand_poses_mean

        poses_batch_dims = utils.broadcast_shapes(
            self.global_rot.shape[:-1],
            self.body_poses.shape[:-2],

            utils.try_get_batch_shape(self.jaw_poses, -2),
            utils.try_get_batch_shape(self.leye_poses, -2),
            utils.try_get_batch_shape(self.leye_poses, -2),
            utils.try_get_batch_shape(self.reye_poses, -2),

            tuple() if not lhand_en else lhand_poses.shape[:-2],
            tuple() if not rhand_en else rhand_poses.shape[:-2],
        )

        poses = [
            self.global_rot.unsqueeze(-2).expand(poses_batch_dims + (-1, 3)),
            self.body_poses.expand(poses_batch_dims + (-1, 3)),
        ]

        if self.jaw_poses is not None:
            poses.append(self.jaw_poses.expand(poses_batch_dims + (-1, 3)))

        if self.leye_poses is not None:
            poses.append(self.leye_poses.expand(poses_batch_dims + (-1, 3)))

        if self.reye_poses is not None:
            poses.append(self.reye_poses.expand(poses_batch_dims + (-1, 3)))

        if lhand_en:
            poses.append(lhand_poses.expand(poses_batch_dims + (-1, 3)))

        if rhand_en:
            poses.append(rhand_poses.expand(poses_batch_dims + (-1, 3)))

        ret = torch.cat(poses, -2)
        # [..., ?, 3]

        return ret

    def combine(self, obj: typing.Self) -> typing.Self:
        ret = BlendingParam(**self.__dict__)

        for field_name in BlendingParam.batch_dim_table.keys():
            field_value = getattr(self, field_name)

            if field_value is None:
                setattr(ret, field_name, getattr(obj, field_name))

        return ret


@beartype
def blending(
    model_data: ModelData,
    blending_param: BlendingParam,
    device: torch.device,
) -> Model:
    blending_param.check(model_data, False)

    vp = model_data.vert_pos

    assert vp.isfinite().all()

    if blending_param.body_shapes is not None:
        assert model_data.body_shape_dirs.isfinite().all()
        assert blending_param.body_shapes.isfinite().all()

        vp = vp + torch.einsum("...vxb,...b->...vx",
                               model_data.body_shape_dirs,
                               blending_param.body_shapes)

    if model_data.expr_shape_dirs is not None and \
       blending_param.expr_shapes is not None:
        assert model_data.expr_shape_dirs.isfinite().all()
        assert blending_param.expr_shapes.isfinite().all()

        vp = vp + torch.einsum("...vxb,...b->...vx",
                               model_data.expr_shape_dirs,
                               blending_param.expr_shapes)

    # [..., V, 3]

    assert model_data.joint_ts_mean.isfinite().all()

    binding_joint_ts = model_data.joint_ts_mean.clone()

    if blending_param.body_shapes is not None:
        assert model_data.body_shape_joint_regressor.isfinite().all()
        assert blending_param.body_shapes.isfinite().all()

        binding_joint_ts = binding_joint_ts + torch.einsum(
            "...jxb,...b->...jx",
            model_data.body_shape_joint_regressor,
            blending_param.body_shapes,
        )

    if model_data.expr_shape_joint_regressor is not None and \
       blending_param.expr_shapes is not None:
        binding_joint_ts = binding_joint_ts + torch.einsum(
            "...jxb,...b->...jx",
            model_data.expr_shape_joint_regressor,
            blending_param.expr_shapes,
        )

    # [..., J, 3]

    J = model_data.kin_tree.joints_cnt

    binding_pose_rs = torch.eye(3, dtype=utils.FLOAT, device=device) \
        .unsqueeze(0).expand((J, 3, 3))

    binding_pose_ts = torch.empty_like(binding_joint_ts)
    binding_pose_ts[..., model_data.kin_tree.root, :] = \
        binding_joint_ts[..., model_data.kin_tree.root, :]
    # [..., J, 3]

    for u in model_data.kin_tree.joints_tp[1:]:
        p = model_data.kin_tree.parents[u]

        binding_pose_ts[..., u, :] = \
            binding_joint_ts[..., u, :] - binding_joint_ts[..., p, :]

    target_pose_rs = utils.axis_angle_to_rot_mat(
        blending_param.get_poses(model_data), out_shape=(3, 3))
    # [..., J, 3, 3]

    identity = torch.eye(
        3, dtype=target_pose_rs.dtype, device=target_pose_rs.device)

    pose_feature = \
        (target_pose_rs[..., 1:, :, :] - identity).view((-1, (J - 1) * 3 * 3))
    # [..., (J - 1) * 3 * 3]

    vp = vp + torch.einsum(
        "...vxp,...p->...vx",
        model_data.pose_dirs,  # [..., V, 3, (J - 1) * 3 * 3]
        pose_feature,  # [..., (J - 1) * 3 * 3]
    )

    lbs_result = \
        blending_utils.lbs(
            kin_tree=model_data.kin_tree,
            lbs_weights=model_data.lbs_weights,

            vert_pos=vp,
            vert_dir=model_data.vert_nor if blending_param.blending_vert_nor else None,

            binding_pose_rs=binding_pose_rs,
            binding_pose_ts=binding_pose_ts,
            target_pose_rs=target_pose_rs,
            target_pose_ts=binding_pose_ts,
        )

    if blending_param.global_transl is not None:
        lbs_result.target_joint_Ts[..., :, :3, 3] += \
            blending_param.global_transl.unsqueeze(-2)

        vp = blending_param.global_transl.unsqueeze(-2) + \
            lbs_result.blended_vert_pos

    return Model(
        kin_tree=model_data.kin_tree,

        mesh_data=model_data.mesh_data,
        tex_mesh_data=model_data.tex_mesh_data,

        vert_pos=vp,
        vert_nor=None if lbs_result.blended_vert_dir is None
        else utils.vec_normed(lbs_result.blended_vert_dir),

        tex_vert_pos=model_data.tex_vert_pos,

        joint_Ts=lbs_result.target_joint_Ts,
    )
