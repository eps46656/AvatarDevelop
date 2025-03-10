import dataclasses
import pickle

import torch
import typing

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
    joints: torch.Tensor
    vertices: torch.Tensor


BODY_SHAPES_CNT = 10
EXPR_SHAPES_CNT = 10

BODY_POSES_CNT = 21
JAW_POSES_CNT = 1
LEYE_POSES_CNT = 1
REYE_POSES_CNT = 1
LHAND_POSES_CNT = 15
RHAND_POSES_CNT = 15

BODY_SHAPES_SPACE_DIM = 300


def SMPLXBlending(
    *,
    kin_tree: KinTree,

    vertices: torch.Tensor,  # [..., V, 3]
    joint_regressor: torch.Tensor,  # [..., J, V]

    shape_dirs: typing.Optional[torch.Tensor],
    # [..., V, 3, BODY_SHAPES_CNT + EXPR_SHAPES_CNT]

    body_shapes: typing.Optional[torch.Tensor] = None,
    # [..., BODY_POSES_CNT]

    expr_shapes: typing.Optional[torch.Tensor] = None,
    # [..., EXPR_SHAPES_CNT]

    lbs_weights: torch.Tensor,  # [..., V, J]

    root_rs: torch.Tensor = None,  # [..., 3]
    root_ts: torch.Tensor = None,  # [..., 3]

    body_poses: typing.Optional[torch.Tensor] = None,
    # [..., BODY_POSES_CNT, 3]

    jaw_poses: typing.Optional[torch.Tensor] = None,
    # [..., JAW_POSES_CNT, 3]

    leye_poses: typing.Optional[torch.Tensor] = None,
    # [..., LEYE_POSES_CNT, 3]

    reye_poses: typing.Optional[torch.Tensor] = None,
    # [..., REYE_POSES_CNT, 3]

    lhand_poses: typing.Optional[torch.Tensor] = None,
    # [..., LHAND_POSES_CNT, 3]

    rhand_poses: typing.Optional[torch.Tensor] = None,
    # [..., RHAND_POSES_CNT, 3]

    lhand_poses_mean: torch.Tensor,  # [..., LHAND_POSES_CNT, 3]
    rhand_poses_mean: torch.Tensor,  # [..., RHAND_POSES_CNT, 3]

    device: torch.device
):
    assert 2 <= len(vertices)
    assert 2 <= len(joint_regressor)

    J = kin_tree.joints_cnt
    V = vertices.shape[0]

    assert vertices.shape[-2:] == (V, 3)
    assert joint_regressor.shape[-2:] == (J, V)

    # ---

    assert 3 <= len(shape_dirs)
    assert shape_dirs.shape[-3:-1] == (V, 3)

    if body_shapes is None:
        body_shapes = torch.zeros((10,), dtype=utils.FLOAT, device=device)

    if expr_shapes is None:
        expr_shapes = torch.zeros((10,), dtype=utils.FLOAT, device=device)

    assert 1 <= body_shapes.dim()
    assert 1 <= expr_shapes.dim()

    assert body_shapes.shape[-1] == 10
    assert expr_shapes.shape[-1] == 10

    shapes_batch_dims = list(utils.GetCommonShape(
        [body_shapes.shape[:-1], expr_shapes[:-1]]))

    shapes = torch.cat((
        body_shapes.expand(shapes_batch_dims + [10]),
        expr_shapes.expand(shapes_batch_dims + [10]),
    ), dim=-1)

    # ---

    assert lbs_weights.shape[-2:] == (V, J)

    if root_rs is None:
        root_rs = torch.zeros((3,), dtype=utils.FLOAT, device=device)

    if root_ts is None:
        root_ts = torch.zeros((3,), dtype=utils.FLOAT, device=device)

    if body_poses is None:
        body_poses = torch.zeros(
            (BODY_POSES_CNT, 3), dtype=utils.FLOAT, device=device)

    if jaw_poses is None:
        jaw_poses = torch.zeros(
            (JAW_POSES_CNT, 3), dtype=utils.FLOAT, device=device)

    if leye_poses is None:
        leye_poses = torch.zeros(
            (LEYE_POSES_CNT, 3), dtype=utils.FLOAT, device=device)

    if reye_poses is None:
        reye_poses = torch.zeros(
            (REYE_POSES_CNT, 3), dtype=utils.FLOAT, device=device)

    if lhand_poses is None:
        lhand_poses = torch.zeros(
            (LHAND_POSES_CNT, 3), dtype=utils.FLOAT, device=device)

    if rhand_poses is None:
        rhand_poses = torch.zeros(
            (RHAND_POSES_CNT, 3), dtype=utils.FLOAT, device=device)

    assert 1 <= root_rs.dim()
    assert 1 <= root_ts.dim()
    assert 2 <= body_poses.dim()
    assert 2 <= jaw_poses.dim()
    assert 2 <= leye_poses.dim()
    assert 2 <= reye_poses.dim()
    assert 2 <= lhand_poses.dim()
    assert 2 <= rhand_poses.dim()
    assert 2 <= lhand_poses_mean.dim()
    assert 2 <= rhand_poses_mean.dim()

    assert root_rs.shape[-1] == 3
    assert root_ts.shape[-1] == 3
    assert body_poses.shape[-2:] == (BODY_POSES_CNT, 3)
    assert jaw_poses.shape[-2:] == (JAW_POSES_CNT, 3)
    assert leye_poses.shape[-2:] == (LEYE_POSES_CNT, 3)
    assert reye_poses.shape[-2:] == (REYE_POSES_CNT, 3)
    assert lhand_poses.shape[-2:] == (LHAND_POSES_CNT, 3)
    assert rhand_poses.shape[-2:] == (RHAND_POSES_CNT, 3)
    assert lhand_poses_mean.shape[-2:] == (LHAND_POSES_CNT, 3)
    assert rhand_poses_mean.shape[-2:] == (RHAND_POSES_CNT, 3)

    lhand_poses = lhand_poses + lhand_poses_mean
    rhand_poses = rhand_poses + rhand_poses_mean

    poses_batch_dims = list(utils.GetCommonShape([
        root_rs.shape[:-1],
        body_poses.shape[:-2],
        jaw_poses.shape[:-2],
        leye_poses.shape[:-2],
        reye_poses.shape[:-2],
        lhand_poses.shape[:-2],
        rhand_poses.shape[:-2],
    ]))

    poses = torch.cat((
        root_rs.expand(poses_batch_dims + [1, 3]),
        body_poses.expand(poses_batch_dims + [BODY_POSES_CNT, 3]),
        jaw_poses.expand(poses_batch_dims + [JAW_POSES_CNT, 3]),
        leye_poses.expand(poses_batch_dims + [LEYE_POSES_CNT, 3]),
        reye_poses.expand(poses_batch_dims + [REYE_POSES_CNT, 3]),
        lhand_poses.expand(poses_batch_dims + [LHAND_POSES_CNT, 3]),
        rhand_poses.expand(poses_batch_dims + [RHAND_POSES_CNT, 3]),
    ), dim=-2)

    # ---

    vs = vertices + torch.einsum("...vxb,...b->...vx", shape_dirs, shapes)
    # [..., V, 3]

    binding_joints_ts = torch.einsum("jv,...vx->...jx", joint_regressor, vs)
    # [..., J, 3]

    binding_pose_rs = torch.eye(3, dtype=utils.FLOAT, device=device) \
        .unsqueeze(0).expand((J, 3, 3))

    binding_pose_ts = torch.empty_like(binding_joints_ts)
    binding_pose_ts[..., kin_tree.root, :] = \
        binding_joints_ts[..., kin_tree.root, :]
    # [..., J, 3]

    for u in kin_tree.joints_tp[1:]:
        p = kin_tree.parents[u]

        binding_pose_ts[..., u, :] = \
            binding_joints_ts[..., u, :] - binding_joints_ts[..., p, :]

    binding_joint_rs, binding_joints_ts, joint_rs, joint_ts, vs = \
        blending_utils.LBS(
            kin_tree=kin_tree,
            vertices=vs,
            lbs_weights=lbs_weights,
            binding_pose_rs=binding_pose_rs,
            binding_pose_ts=binding_pose_ts,
            pose_rs=utils.GetRotMat(poses),
            pose_ts=binding_pose_ts,
        )

    if root_ts is not None:
        joint_ts = root_ts + joint_ts
        vs = root_ts + vs

    return SMPLXModel(
        joints=joint_ts,
        vertices=vs,
    )


class SMPLXBuilder:
    def __init__(self,
                 model_path,
                 device: torch.device,
                 ):
        self.device = device

        with open(model_path, "rb") as f:
            self.model_data = pickle.load(f, encoding="latin1")

        kin_tree_table = self.model_data["kintree_table"]

        kin_tree_links = [(int(kin_tree_table[0, j]), int(kin_tree_table[1, j]))
                          for j in range(kin_tree_table.shape[1])]

        self.kin_tree = KinTree.FromLinks(kin_tree_links, 2**32-1)
        # joints_cnt = J

        # [V, 3]

        self.lbs_weights = torch.from_numpy(self.model_data["weights"]) \
            .to(dtype=utils.FLOAT, device=self.device)
        # [V, J]

        self.joint_regressor = torch.from_numpy(
            self.model_data["J_regressor"]) \
            .to(dtype=utils.FLOAT, device=self.device)
        # [J, V]

        shape_dirs = torch.from_numpy(self.model_data["shapedirs"]) \
            .to(dtype=utils.FLOAT, device=self.device)

        self.shape_dirs = torch.cat([
            shape_dirs[:, :, 0:BODY_SHAPES_CNT],
            shape_dirs[:, :, BODY_SHAPES_SPACE_DIM:
                       BODY_SHAPES_SPACE_DIM + EXPR_SHAPES_CNT]
        ], dim=-1)
        # [V, 3, BODY_SHAPES_CNT + EXPR_SHAPES_CNT]

        self.vertices = torch.from_numpy(self.model_data["v_template"]) \
            .to(dtype=utils.FLOAT, device=self.device)
        # [V, 3]

        self.vertex_textures = torch.from_numpy(self.model_data["vt"]) \
            .to(dtype=utils.FLOAT, device=self.device)
        # [V, 2]

        self.faces = torch.from_numpy(self.model_data["f"]) \
            .to(dtype=utils.INT, device=self.device)
        # [F, 3]

        self.face_textures = torch.from_numpy(self.model_data["ft"]) \
            .to(dtype=utils.INT, device=self.device)
        # [F, 3]

        self.lhand_poses_mean = torch.from_numpy(
            self.model_data["hands_meanl"]) \
            .reshape((-1, 3))[-LHAND_POSES_CNT:, :] \
            .to(dtype=utils.FLOAT, device=self.device)
        # [LHAND_POSES_CNT, 3]

        self.rhand_poses_mean = torch.from_numpy(
            self.model_data["hands_meanr"]) \
            .reshape((-1, 3))[-RHAND_POSES_CNT:, :] \
            .to(dtype=utils.FLOAT, device=self.device)
        # [RHAND_POSES_CNT, 3]

        J = self.kin_tree.joints_cnt
        B = self.shape_dirs.shape[2]
        V = self.vertices.shape[0]

        assert self.kin_tree.joints_cnt == J, self.kin_tree.joints_cnt
        assert self.vertices.shape == (V, 3), self.vertices.shape
        assert self.lbs_weights.shape == (V, J), self.lbs_weights.shape
        assert self.joint_regressor.shape == (J, V), self.joint_regressor.shape
        assert self.shape_dirs.shape == (V, 3, B), self.shape_dirs.shape

    def GetShapesCnt(self):
        return self.shape_dirs.shape[2]

    def GetJointsCnt(self):
        return self.kin_tree.joints_cnt

    def GetVerticesCnt(self):
        return self.vertices.shape[0]

    def GetFacesCnt(self):
        return 0
        # return self.faces

    def forward(
        self,

        body_shapes: typing.Optional[torch.Tensor] = None,
        # [..., BODY_POSES_CNT]

        expr_shapes: typing.Optional[torch.Tensor] = None,
        # [..., EXPR_SHAPES_CNT]

        root_rs: torch.Tensor = None,  # [..., 3]
        root_ts: torch.Tensor = None,  # [..., 3]

        body_poses: typing.Optional[torch.Tensor] = None,
        # [..., BODY_POSES_CNT, 3]

        jaw_poses: typing.Optional[torch.Tensor] = None,
        # [..., JAW_POSES_CNT, 3]

        leye_poses: typing.Optional[torch.Tensor] = None,
        # [..., LEYE_POSES_CNT, 3]

        reye_poses: typing.Optional[torch.Tensor] = None,
        # [..., REYE_POSES_CNT, 3]

        lhand_poses: typing.Optional[torch.Tensor] = None,
        # [..., LHAND_POSES_CNT, 3]

        rhand_poses: typing.Optional[torch.Tensor] = None,
        # [..., RHAND_POSES_CNT, 3]
    ):
        return SMPLXBlending(
            kin_tree=self.kin_tree,

            vertices=self.vertices,
            joint_regressor=self.joint_regressor,

            shape_dirs=self.shape_dirs,

            body_shapes=body_shapes,
            expr_shapes=expr_shapes,

            lbs_weights=self.lbs_weights,

            root_rs=root_rs,
            root_ts=root_ts,

            body_poses=body_poses,
            jaw_poses=jaw_poses,
            leye_poses=leye_poses,
            reye_poses=reye_poses,
            lhand_poses=lhand_poses,
            rhand_poses=rhand_poses,

            lhand_poses_mean=self.lhand_poses_mean,
            rhand_poses_mean=self.rhand_poses_mean,

            device=self.device
        )

    def GetVertices(self) -> torch.Tensor:  # [V, 3]
        return self.vertices

    def GetVertexTextures(self) -> torch.Tensor:  # [V, 2]
        return self.vertex_textures

    def GetFaces(self) -> torch.Tensor:  # [F, 3]
        return self.faces

    def GetFaceTextures(self) -> torch.Tensor:  # [F, 3]
        return self.face_textures
