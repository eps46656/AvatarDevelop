import dataclasses
import os
import typing

import torch
from beartype import beartype

from .. import kin_utils, mesh_utils, utils
from .config import BODY_SHAPES_SPACE_DIM
from .ModelConfig import ModelConfig


@beartype
@dataclasses.dataclass
class ModelData:
    kin_tree: kin_utils.KinTree

    vertex_positions: torch.Tensor  # [..., V, 3]
    vertex_normals: torch.Tensor  # [..., V, 3]

    texture_vertex_positions: typing.Optional[torch.Tensor]  # [..., TV, 2]

    faces: typing.Optional[torch.Tensor]  # [F, 3]
    texture_faces: typing.Optional[torch.Tensor]  # [F, 3]

    lbs_weights: torch.Tensor  # [..., V, J]

    body_shape_dirs: torch.Tensor  # [..., V, 3, BS]
    expr_shape_dirs: typing.Optional[torch.Tensor]  # [..., V, 3, ES]

    body_joints_cnt: int
    jaw_joints_cnt: int
    eye_joints_cnt: int
    hand_joints_cnt: int

    joint_ts_mean: torch.Tensor  # [..., J, 3]

    body_shape_joint_regressor: torch.Tensor  # [..., J, 3, BS]

    expr_shape_joint_regressor: typing.Optional[torch.Tensor]
    # [..., J, 3, BS]

    lhand_poses_mean: typing.Optional[torch.Tensor]  # [..., HANDJ, 3]
    rhand_poses_mean: typing.Optional[torch.Tensor]  # [..., HANDJ, 3]

    mesh_data: mesh_utils.MeshData

    def Check(self) -> None:
        V, = utils.CheckShapes(self.vertex_positions, (..., -1, 3))

        TV, = (0,) if self.texture_vertex_positions is None else\
            utils.CheckShapes(self.texture_vertex_positions, (..., -1, 2))

        F, = (0,) if self.faces is None else\
            utils.CheckShapes(self.faces, (-1, 3))

        TF, = (0,) if self.texture_faces is None else\
            utils.CheckShapes(self.texture_faces, (-1, 3))

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

        if self.expr_shape_joint_regressor is not None:
            utils.CheckShapes(self.expr_shape_joint_regressor, (..., J, 3, ES))

        if 0 < HANDJ:
            assert self.lhand_poses_mean is not None
            assert self.rhand_poses_mean is not None

            utils.CheckShapes(self.lhand_poses_mean, (..., HANDJ, 3))
            utils.CheckShapes(self.rhand_poses_mean, (..., HANDJ, 3))

    def GetBodyShapesCnt(self) -> int:
        return self.body_shape_dirs.shape[-1]

    def GetExprShapesCnt(self) -> int:
        return 0 if self.expr_shape_dirs is None \
            else self.expr_shape_dirs.shape[-1]

    @staticmethod
    def FromFile(
        *,
        model_data_path: os.PathLike,
        model_config: ModelConfig,
        device: torch.device,
    ) -> typing.Self:
        model_data = utils.ReadPickle(model_data_path)

        kin_tree_table = model_data["kintree_table"]

        kin_tree_links = [
            (int(kin_tree_table[0, j]), int(kin_tree_table[1, j]))
            for j in range(kin_tree_table.shape[1])]

        kin_tree = kin_utils.KinTree.FromLinks(kin_tree_links, 2**32-1)
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

        def GetShapeDirs(shape_dirs: torch.Tensor, shape_dirs_cnt: int) \
                -> torch.Tensor:
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
            shape_dirs[:, :, :BODY_SHAPES_SPACE_DIM],
            model_config.body_shapes_cnt)

        expr_shape_dirs = GetShapeDirs(
            shape_dirs[:, :, BODY_SHAPES_SPACE_DIM:],
            model_config.expr_shapes_cnt)

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

        if expr_shape_dirs is None:
            expr_shape_joint_regressor = None
        else:
            expr_shape_joint_regressor = torch.einsum(
                "...jv,...vxb->...jxb",
                joint_regressor,
                expr_shape_dirs,
            )

        # ---

        if "vt" in model_data:
            texture_vertex_positions = torch.from_numpy(model_data["vt"]) \
                .to(dtype=utils.FLOAT, device=device)

            TV, = utils.CheckShapes(texture_vertex_positions, (..., -1, 2))
        else:
            texture_vertex_positions = None
            TV = 0

        # ---

        faces = torch.from_numpy(model_data["f"]) \
            .to(dtype=torch.long, device=device)

        F, = utils.CheckShapes(faces, (..., -1, 3))

        # ---

        if "ft" in model_data:
            texture_faces = torch.from_numpy(model_data["ft"]) \
                .to(dtype=torch.long, device=device)

            TF, = utils.CheckShapes(texture_faces, (..., -1, 3))
        else:
            texture_faces = None

            TF = 0

        # ---

        if "hands_meanl" in model_data:
            lhand_poses_mean = torch.from_numpy(
                model_data["hands_meanl"]) \
                .reshape((-1, 3))[-model_config.hand_joints_cnt:, :] \
                .to(dtype=utils.FLOAT, device=device)
        else:
            lhand_poses_mean = None

        if "hands_meanr" in model_data:
            rhand_poses_mean = torch.from_numpy(
                model_data["hands_meanr"]) \
                .reshape((-1, 3))[-model_config.hand_joints_cnt:, :] \
                .to(dtype=utils.FLOAT, device=device)
        else:
            rhand_poses_mean = None

        # ---

        mesh_data = mesh_utils.MeshData.FromFaceVertexAdjList(
            V, faces, device)

        # ---

        return ModelData(
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

            body_joints_cnt=model_config.body_joints_cnt,
            jaw_joints_cnt=model_config.jaw_joints_cnt,
            eye_joints_cnt=model_config.eye_joints_cnt,
            hand_joints_cnt=model_config.hand_joints_cnt,

            joint_ts_mean=joint_ts_mean,
            body_shape_joint_regressor=body_shape_joint_regressor,
            expr_shape_joint_regressor=expr_shape_joint_regressor,

            lhand_poses_mean=lhand_poses_mean,
            rhand_poses_mean=rhand_poses_mean,

            mesh_data=mesh_data,
        )
