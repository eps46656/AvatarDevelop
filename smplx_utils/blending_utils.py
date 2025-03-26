import dataclasses
import typing

import torch
from beartype import beartype

from .. import blending_utils, utils
from .Model import Model
from .ModelData import ModelData


@beartype
@dataclasses.dataclass
class BlendingParam:
    body_shapes: typing.Optional[torch.Tensor] = None
    # [..., BS]

    expr_shapes: typing.Optional[torch.Tensor] = None
    # [..., ES]

    global_transl: typing.Optional[torch.Tensor] = None
    # [..., 3]

    global_rot: typing.Optional[torch.Tensor] = None
    # [..., 3]

    body_poses: typing.Optional[torch.Tensor] = None
    # [..., BJ - 1, 3]

    jaw_poses: typing.Optional[torch.Tensor] = None
    # [..., JJ, 3]

    leye_poses: typing.Optional[torch.Tensor] = None
    # [..., EYEJ, 3]

    reye_poses: typing.Optional[torch.Tensor] = None
    # [..., EYEJ, 3]

    lhand_poses: typing.Optional[torch.Tensor] = None
    # [..., HANDJ, 3]

    rhand_poses: typing.Optional[torch.Tensor] = None
    # [..., HANDJ, 3]

    blending_vertex_normal: bool = False

    def Check(
        self,
        model_data: ModelData,
        single_batch: bool,
    ) -> None:
        model_data.Check()

        BS = model_data.GetBodyShapesCnt()
        ES = model_data.GetExprShapesCnt()

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

        for field in dataclasses.fields(BlendingParam):
            field_name = field.name

            if field_name not in tensor_shape_constraints:
                continue

            value = getattr(self, field_name)

            if value is None:
                continue

            assert isinstance(value, torch.Tensor)

            shape_constraint = tensor_shape_constraints[field_name]

            if single_batch:
                assert value.shape == shape_constraint
            else:
                assert value.shape[-len(shape_constraint):] == shape_constraint

    def GetPoses(self, model_data: ModelData) -> torch.Tensor:
        self.Check(model_data, False)

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

        poses_batch_dims = utils.BroadcastShapes(
            self.global_rot.shape[:-1],
            self.body_poses.shape[:-2],

            tuple() if self.jaw_poses is None else self.jaw_poses.shape[:-2],
            tuple() if self.leye_poses is None else self.leye_poses.shape[:-2],
            tuple() if self.reye_poses is None else self.reye_poses.shape[:-2],
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

    def GetCombined(self, obj: typing.Self) -> typing.Self:
        ret = BlendingParam()

        for field in dataclasses.fields(BlendingParam):
            field_name = field.name

            value = getattr(self, field_name)

            setattr(ret, field_name,
                    getattr(obj, field_name)
                    if value is None else value)

        return ret


@beartype
def Blending(
    model_data: ModelData,
    blending_param: BlendingParam,
    device: torch.device,
) -> Model:
    blending_param.Check(model_data, False)

    vps = model_data.vertex_positions

    if blending_param.body_shapes is not None:
        vps = vps + torch.einsum("...vxb,...b->...vx",
                                 model_data.body_shape_dirs,
                                 blending_param.body_shapes)

    if model_data.expr_shape_dirs is not None and \
       blending_param.expr_shapes is not None:
        vps = vps + torch.einsum("...vxb,...b->...vx",
                                 model_data.expr_shape_dirs,
                                 blending_param.expr_shapes)

    # [..., V, 3]

    binding_joint_ts = model_data.joint_ts_mean.clone()

    if blending_param.body_shapes is not None:
        binding_joint_ts += torch.einsum(
            "...jxb,...b->...jx",
            model_data.body_shape_joint_regressor,
            blending_param.body_shapes,
        )

    if model_data.expr_shape_joint_regressor is not None and \
       blending_param.expr_shapes is not None:
        binding_joint_ts += torch.einsum(
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

    lbs_result = \
        blending_utils.LBS(
            kin_tree=model_data.kin_tree,
            lbs_weights=model_data.lbs_weights,

            vertex_positions=vps,
            vertex_directions=model_data.vertex_normals if blending_param.blending_vertex_normal else None,

            binding_pose_rs=binding_pose_rs,
            binding_pose_ts=binding_pose_ts,
            target_pose_rs=utils.AxisAngleToRotMat(
                blending_param.GetPoses(model_data), out_shape=(3, 3)),
            target_pose_ts=binding_pose_ts,
        )

    if blending_param.global_transl is not None:
        lbs_result.target_joint_Ts[..., :, :3, 3] += \
            blending_param.global_transl.unsqueeze(-2)

        vps = blending_param.global_transl.unsqueeze(-2) + \
            lbs_result.blended_vertex_positions

    return Model(
        kin_tree=model_data.kin_tree,

        joint_Ts=lbs_result.target_joint_Ts,

        vertex_positions=vps,
        vertex_normals=None if lbs_result.blended_vertex_directions is None else utils.Normalized(
            lbs_result.blended_vertex_directions),

        texture_vertex_positions=model_data.texture_vertex_positions,

        faces=model_data.faces,
        texture_faces=model_data.texture_faces,

        mesh_data=model_data.mesh_data,
    )
