import dataclasses
import os
import pickle
import typing
import mesh_utils

import torch
from beartype import beartype

import blending_utils
import utils
from kin_tree import KinTree

SMPLX_FT = +utils.Z_AXIS
SMPLX_BK = -utils.Z_AXIS

SMPLX_LT = +utils.X_AXIS
SMPLX_RT = -utils.X_AXIS

SMPLX_UP = +utils.Y_AXIS
SMPLX_DW = -utils.Y_AXIS


@dataclasses.dataclass
class SMPLXModel:
    joint_rs: torch.Tensor  # [..., J, 3, 3]
    joint_ts: torch.Tensor  # [..., J, 3]

    vertex_positions: torch.Tensor  # [..., V, 3]
    vertex_normals: torch.Tensor  # [..., V, 3]


BODY_SHAPES_CNT = 10
EXPR_SHAPES_CNT = 10

BODY_POSES_CNT = 21
JAW_POSES_CNT = 1
EYE_POSES_CNT = 1
HAND_POSES_CNT = 15

BODY_SHAPES_SPACE_DIM = 300


@beartype
@dataclasses.dataclass
class SMPLXModelData:
    kin_tree: KinTree

    vertex_positions: torch.Tensor  # [..., V, 3]
    vertex_normals: torch.Tensor  # [..., V, 3]

    texture_vertex_positions: typing.Optional[torch.Tensor]  # [..., VT, 2]

    faces: typing.Optional[torch.Tensor]  # [..., F, 3]
    texture_faces: typing.Optional[torch.Tensor]  # [..., F, 3]

    lbs_weights: torch.Tensor  # [..., V, J]

    body_shape_dirs: torch.Tensor  # [..., V, 3, BS]
    expr_shape_dirs: typing.Optional[torch.Tensor]  # [..., V, 3, ES]

    body_joints_cnt: int
    jaw_joints_cnt: int
    eye_joints_cnt: int
    hand_joints_cnt: int

    joint_ts_mean: torch.Tensor  # [..., J, 3]

    body_shape_joint_regressor: torch.Tensor  # [..., J, 3, BS]
    expr_shape_joint_regressor: torch.Tensor  # [..., J, 3, BS]

    lhand_poses_mean: typing.Optional[torch.Tensor]  # [..., HANDJ, 3]
    rhand_poses_mean: typing.Optional[torch.Tensor]  # [..., HANDJ, 3]

    def Check(self):
        V, = utils.CheckShapes(self.vertex_positions, (..., -1, 3))

        VT, = (0,) if self.texture_vertex_positions is None else\
            utils.CheckShapes(self.texture_vertex_positions, (..., -1, 2))

        F, = (0,) if self.faces is None else\
            utils.CheckShapes(self.texture_faces, (..., -1, 3))

        FT, = (0,) if self.texture_faces is None else\
            utils.CheckShapes(self.texture_faces, (..., -1, 3))

        BS, = (0, ) if self.body_shape_dirs is None else \
            utils.CheckShapes(self.body_shape_dirs, (..., V, 3, -1))

        ES, = (0, ) if self.expr_shape_dirs is None else \
            utils.CheckShapes(self.expr_shape_dirs, (..., V, 3, -1))

        BJ = self.body_joints_cnt
        JJ = self.jaw_joints_cnt
        EJ = self.eye_joints_cnt
        HANDJ = self.hand_joints_cnt

        assert 1 <= BJ
        assert 0 <= JJ
        assert 0 <= EJ
        assert 0 <= HANDJ

        J = self.kin_tree.joints_cnt

        assert BJ + JJ + EJ * 2 + HANDJ * 2 == J

        utils.CheckShapes(self.lbs_weights, (..., V, J))

        utils.CheckShapes(self.body_shape_joint_regressor, (..., J, 3, BS))
        utils.CheckShapes(self.expr_shape_joint_regressor, (..., J, 3, ES))

        if 0 < HANDJ:
            assert self.lhand_poses_mean is not None
            assert self.rhand_poses_mean is not None

            utils.CheckShapes(self.lhand_poses_mean, (..., HANDJ, 3))
            utils.CheckShapes(self.rhand_poses_mean, (..., HANDJ, 3))

    def GetJointsCnt(self):
        return self.kin_tree.joints_cnt

    def GetVerticesCnt(self):
        return self.vertex_positions.shape[-2]

    def GetBodyShapesCnt(self):
        return self.body_shape_dirs.shape[-1]

    def GetExprShapesCnt(self):
        return 0 if self.expr_shape_dirs is None \
            else self.expr_shape_dirs.shape[-1]


@beartype
@dataclasses.dataclass
class SMPLXBlendingParam:
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

    def Check(self,
              model_data: SMPLXModelData,
              single_batch: bool):
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

        for field in dataclasses.fields(SMPLXBlendingParam):
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

    def GetPoses(self, model_data: SMPLXModelData):
        self.Check(model_data, False)

        assert self.global_rot is not None
        assert self.body_poses is not None
        assert self.jaw_poses is not None
        assert self.leye_poses is not None
        assert self.reye_poses is not None
        assert self.lhand_poses is not None
        assert self.rhand_poses is not None

        lhand_poses = self.lhand_poses + model_data.lhand_poses_mean
        rhand_poses = self.rhand_poses + model_data.rhand_poses_mean

        poses_batch_dims = list(utils.GetCommonShape([
            self.global_rot.shape[:-1],
            self.body_poses.shape[:-2],
            self.jaw_poses.shape[:-2],
            self.leye_poses.shape[:-2],
            self.reye_poses.shape[:-2],
            lhand_poses.shape[:-2],
            rhand_poses.shape[:-2],
        ]))

        ret = torch.cat((
            self.global_rot.expand(poses_batch_dims + [-1, 3]),
            self.body_poses.expand(poses_batch_dims + [-1, 3]),
            self.jaw_poses.expand(poses_batch_dims + [-1, 3]),
            self.leye_poses.expand(poses_batch_dims + [-1, 3]),
            self.reye_poses.expand(poses_batch_dims + [-1, 3]),
            lhand_poses.expand(poses_batch_dims + [-1, 3]),
            rhand_poses.expand(poses_batch_dims + [-1, 3]),
        ), dim=-2)

        return ret


def MergeBlendingParam(
    blending_param: SMPLXBlendingParam,
    defualt_blending_param: SMPLXBlendingParam,
):
    ret = SMPLXBlendingParam()

    for field in dataclasses.fields(SMPLXBlendingParam):
        field_name = field.name

        value = getattr(blending_param, field_name)

        setattr(ret, field_name,
                getattr(defualt_blending_param, field_name)
                if value is None else value)

    return ret


def SMPLXBlending(
    model_data: SMPLXModelData,
    blending_param: SMPLXBlendingParam,
    device: torch.device,
):
    blending_param.Check(model_data, False)

    vps = model_data.vertex_positions

    if blending_param.body_shapes is not None:
        vps += torch.einsum("...vxb,...b->...vx",
                            model_data.body_shape_dirs,
                            blending_param.body_shapes)

    if blending_param.expr_shapes is not None:
        vps += torch.einsum("...vxb,...b->...vx",
                            model_data.expr_shape_dirs,
                            blending_param.expr_shapes)

    # [..., V, 3]

    binding_joint_ts = model_data.joint_ts_mean

    if blending_param.body_shapes is not None:
        binding_joint_ts += torch.einsum(
            "...jxb,...b->...jx",
            model_data.body_shape_joint_regressor,
            blending_param.body_shapes,
        )

    if blending_param.expr_shapes is not None:
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
            target_pose_rs=utils.GetRotMat(
                blending_param.GetPoses(model_data)),
            target_pose_ts=binding_pose_ts,
        )

    joint_ts = lbs_result.target_joint_ts

    if blending_param.global_transl is not None:
        joint_ts = blending_param.global_transl + joint_ts
        vps = blending_param.global_transl + lbs_result.blended_vertex_positions

    return SMPLXModel(
        joint_rs=lbs_result.target_joint_rs,
        joint_ts=joint_ts,

        vertex_positions=vps,
        vertex_normals=None if lbs_result.blended_vertex_directions is None else utils.Normalized(
            lbs_result.blended_vertex_directions),
    )


def ReadModelData(
    *,
    model_data_path: os.PathLike,
    body_shapes_cnt: int,
    expr_shapes_cnt: int,
    body_joints_cnt: int,
    jaw_joints_cnt: int,
    eye_joints_cnt: int,
    hand_joints_cnt: int,
    device: torch.device,
):
    with open(model_data_path, "rb") as f:
        model_data = pickle.load(f, encoding="latin1")

    kin_tree_table = model_data["kintree_table"]

    kin_tree_links = [
        (int(kin_tree_table[0, j]), int(kin_tree_table[1, j]))
        for j in range(kin_tree_table.shape[1])]

    kin_tree = KinTree.FromLinks(kin_tree_links, 2**32-1)
    # joints_cnt = J

    J = kin_tree.joints_cnt

    # ---

    vertex_positions = torch.from_numpy(model_data["v_template"]) \
        .to(dtype=utils.FLOAT, device=device)
    # [V, 3]

    V, = utils.CheckShapes(vertex_positions, (-1, 3))

    # ---

    lbs_weights = torch.from_numpy(model_data["weights"]) \
        .to(dtype=utils.FLOAT, device=device)

    utils.CheckShapes(lbs_weights, (V, J))

    # ---

    joint_regressor = torch.from_numpy(model_data["J_regressor"]) \
        .to(dtype=utils.FLOAT, device=device)

    utils.CheckShapes(joint_regressor, (J, V))

    # ---

    def GetShapeDirs(shape_dirs: torch.Tensor, shape_dirs_cnt: int):
        K, = utils.CheckShapes(shape_dirs, (V, 3, -1))

        assert 0 <= shape_dirs_cnt

        if shape_dirs_cnt == 0:
            return None

        if shape_dirs_cnt <= K:
            ret = shape_dirs[:, :, :shape_dirs_cnt]
        else:
            ret = torch.nn.functional.pad(
                shape_dirs,
                (0, shape_dirs_cnt - K),
                "constant",
                0
            )

        return ret.to(dtype=utils.FLOAT, device=device)

    shape_dirs = torch.from_numpy(model_data["shapedirs"])

    body_shape_dirs = GetShapeDirs(
        shape_dirs[:, :, :BODY_SHAPES_SPACE_DIM], body_shapes_cnt)

    expr_shape_dirs = GetShapeDirs(
        shape_dirs[:, :, BODY_SHAPES_SPACE_DIM:], expr_shapes_cnt)

    # ---

    joint_ts_mean = torch.einsum(
        "...jv,...vx->...jx",
        joint_regressor,
        vertex_positions,
    )

    body_shape_joint_regressor = torch.einsum(
        "...jv,...vxb->...jxb",
        joint_regressor,
        body_shape_dirs,
    )

    expr_shape_joint_regressor = torch.einsum(
        "...jv,...vxb->...jxb",
        joint_regressor,
        expr_shape_dirs,
    )

    # ---

    texture_vertex_positions = torch.from_numpy(model_data["vt"]) \
        .to(dtype=utils.FLOAT, device=device)

    VT, = utils.CheckShapes(texture_vertex_positions, (..., -1, 2))

    # ---

    faces = torch.from_numpy(model_data["f"]) \
        .to(dtype=torch.long, device=device)

    F, = utils.CheckShapes(faces, (..., -1, 3))

    # ---

    texture_faces = torch.from_numpy(model_data["ft"]) \
        .to(dtype=torch.long, device=device)

    FT, = utils.CheckShapes(faces, (..., -1, 3))

    # ---

    lhand_poses_mean = torch.from_numpy(
        model_data["hands_meanl"]) \
        .reshape((-1, 3))[-hand_joints_cnt:, :] \
        .to(dtype=utils.FLOAT, device=device)

    rhand_poses_mean = torch.from_numpy(
        model_data["hands_meanr"]) \
        .reshape((-1, 3))[-hand_joints_cnt:, :] \
        .to(dtype=utils.FLOAT, device=device)

    # ---

    return SMPLXModelData(
        kin_tree=kin_tree,

        vertex_positions=vertex_positions,
        vertex_normals=mesh_utils.GetAreaWeightedVertexNormals(
            faces=faces,
            vertex_positions=vertex_positions,
        ).to(dtype=utils.FLOAT, device=device),

        texture_vertex_positions=texture_vertex_positions,

        faces=faces,
        texture_faces=texture_faces,

        lbs_weights=lbs_weights,

        body_shape_dirs=body_shape_dirs,
        expr_shape_dirs=expr_shape_dirs,

        body_joints_cnt=body_joints_cnt,
        jaw_joints_cnt=jaw_joints_cnt,
        eye_joints_cnt=eye_joints_cnt,
        hand_joints_cnt=hand_joints_cnt,

        joint_ts_mean=joint_ts_mean,
        body_shape_joint_regressor=body_shape_joint_regressor,
        expr_shape_joint_regressor=expr_shape_joint_regressor,

        lhand_poses_mean=lhand_poses_mean,
        rhand_poses_mean=rhand_poses_mean,
    )


class SMPLXBuilder:
    def __init__(self,
                 model_data_path: os.PathLike,
                 body_shapes_cnt: int,
                 expr_shapes_cnt: int,
                 body_joints_cnt: int,
                 jaw_joints_cnt: int,
                 eye_joints_cnt: int,
                 hand_joints_cnt: int,
                 device: torch.device,
                 ):
        self.device = device

        self.model_data = ReadModelData(
            model_data_path=model_data_path,
            body_shapes_cnt=body_shapes_cnt,
            expr_shapes_cnt=expr_shapes_cnt,
            body_joints_cnt=body_joints_cnt,
            jaw_joints_cnt=jaw_joints_cnt,
            eye_joints_cnt=eye_joints_cnt,
            hand_joints_cnt=hand_joints_cnt,
            device=device,
        )

        # ---

        self.default_blending_param = SMPLXBlendingParam(
            body_shapes=torch.zeros(
                (body_shapes_cnt,), dtype=utils.FLOAT, device=self.device),

            expr_shapes=torch.zeros(
                (expr_shapes_cnt,), dtype=utils.FLOAT, device=self.device),

            global_transl=torch.zeros(
                (3,), dtype=utils.FLOAT, device=self.device),

            global_rot=torch.zeros(
                (3,), dtype=utils.FLOAT, device=self.device),

            body_poses=torch.zeros(
                (body_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            jaw_poses=torch.zeros(
                (jaw_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            leye_poses=torch.zeros(
                (eye_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            reye_poses=torch.zeros(
                (eye_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            lhand_poses=torch.zeros(
                (hand_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            rhand_poses=torch.zeros(
                (hand_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            blending_vertex_normal=False,
        )

    def GetVerticesCnt(self):
        return self.model_data.vertex_positions.shape[-2]

    def GetTextureVertexCnt(self):
        return 0 if self.model_data.texture_vertex_positions is None else self.model_data.texture_vertex_positions.shape[-2]

    def GetFacesCnt(self):
        return 0 if self.model_data.faces is None else self.model_data.faces.shape[-2]

    def GetTextureFacesCnt(self):
        return 0 if self.model_data.texture_faces is None else self.model_data.texture_faces.shape[-2]

    def GetJointsCnt(self):
        return self.model_data.kin_tree.joints_cnt

    def GetBodyShapesCnt(self):
        return self.model_data.body_shape_dirs.shape[-1]

    def GetExprShapesCnt(self):
        return 0 if self.model_data.expr_shape_dirs is None else self.model_data.expr_shape_dirs.shape[-1]

    def GetFaces(self):
        return self.model_data.faces

    def SetDefaultBlendingParam(self, blending_param: SMPLXBlendingParam):
        blending_param.Check(self.model_data, True)

        for field in dataclasses.fields(SMPLXBlendingParam):
            value = getattr(blending_param, field)

            if value is not None:
                setattr(self.default_blending_param, field, value)

    def forward(self, blending_param: SMPLXBlendingParam):
        return SMPLXBlending(
            self.model_data,
            MergeBlendingParam(blending_param, self.default_blending_param),
            device=self.device,
        )
