import os
import typing

import torch
from beartype import beartype

from .. import kin_utils, mesh_utils, utils
from .config import BODY_SHAPES_SPACE_DIM
from .ModelConfig import ModelConfig


@beartype
class ModelData:
    def __init__(
        self,
        *,
        kin_tree: kin_utils.KinTree,

        body_joints_cnt: int,  # BJ
        jaw_joints_cnt: int,  # JJ
        eye_joints_cnt: int,  # EYEJ

        vertex_positions: torch.Tensor,  # [..., V, 3]
        vertex_normals: torch.Tensor,  # [..., V, 3]

        texture_vertex_positions: typing.Optional[torch.Tensor] = None,
        # [..., TV, 2]

        faces: typing.Optional[torch.Tensor] = None,  # [F, 3]
        texture_faces: typing.Optional[torch.Tensor] = None,  # [F, 3]

        body_shape_dirs: torch.Tensor,  # [..., V, 3, BS]
        expr_shape_dirs: typing.Optional[torch.Tensor] = None,
        # [..., V, 3, ES]

        body_shape_joint_regressor: torch.Tensor,  # [..., J, 3, BS]

        expr_shape_joint_regressor: typing.Optional[torch.Tensor] = None,
        # [..., J, 3, BS]

        joint_ts_mean: torch.Tensor,  # [..., J, 3]

        pose_dirs: torch.Tensor,  # [..., V, 3, (J - 1) * 3 * 3]

        lbs_weights: torch.Tensor,  # [..., V, J]

        lhand_poses_mean: typing.Optional[torch.Tensor] = None,
        # [..., HANDJ, 3]

        rhand_poses_mean: typing.Optional[torch.Tensor] = None,
        # [..., HANDJ, 3]

        mesh_data: mesh_utils.MeshData,
    ):
        device = utils.check_device(
            vertex_positions,
            vertex_normals,
            texture_vertex_positions,
            faces,
            texture_faces,
            body_shape_dirs,
            expr_shape_dirs,
            body_shape_joint_regressor,
            expr_shape_joint_regressor,
            joint_ts_mean,
            pose_dirs,
            lbs_weights,
            lhand_poses_mean,
            rhand_poses_mean,
        )

        J = kin_tree.joints_cnt

        BJ = body_joints_cnt
        JJ = jaw_joints_cnt
        EJ = eye_joints_cnt

        V = utils.check_shapes(
            vertex_positions, (..., -1, 3),
            vertex_normals, (..., -1, 3),
        )

        TV = 0 if texture_vertex_positions is None else \
            utils.check_shapes(texture_vertex_positions, (..., -1, 2))

        F = 0 if faces is None else utils.check_shapes(faces, (-1, 3))

        if texture_faces is not None:
            utils.check_shapes(texture_faces, (F, 3))

        utils.check_shapes(pose_dirs, (..., V, 3, (J - 1) * 3 * 3))

        utils.check_shapes(lbs_weights, (..., V, J))

        BS = 0 if body_shape_dirs is None else \
            utils.check_shapes(body_shape_dirs, (..., V, 3, -1))

        ES = 0 if expr_shape_dirs is None else \
            utils.check_shapes(expr_shape_dirs, (..., V, 3, -1))

        utils.check_shapes(body_shape_joint_regressor, (..., J, 3, BS))

        if expr_shape_joint_regressor is not None:
            utils.check_shapes(expr_shape_joint_regressor, (..., J, 3, ES))

        assert 1 <= BJ
        assert 0 <= JJ
        assert 0 <= EJ

        if lhand_poses_mean is None:
            assert rhand_poses_mean is None
            HANDJ = 0
        else:
            assert rhand_poses_mean is not None
            assert lhand_poses_mean.shape == rhand_poses_mean.shape
            HANDJ = utils.check_shapes(lhand_poses_mean, (..., -1, 3))

        assert BJ + JJ + EJ * 2 + HANDJ * 2 == J

        # ---

        self.kin_tree = kin_tree

        self.body_joints_cnt = body_joints_cnt
        self.jaw_joints_cnt = jaw_joints_cnt
        self.eye_joints_cnt = eye_joints_cnt

        self.vertex_positions = vertex_positions
        self.vertex_normals = vertex_normals

        self.texture_vertex_positions = texture_vertex_positions

        self.faces = faces
        self.texture_faces = texture_faces

        self.body_shape_dirs = body_shape_dirs
        self.expr_shape_dirs = expr_shape_dirs

        self.body_shape_joint_regressor = body_shape_joint_regressor
        self.expr_shape_joint_regressor = expr_shape_joint_regressor

        self.joint_ts_mean = joint_ts_mean

        self.pose_dirs = pose_dirs

        self.lbs_weights = lbs_weights

        self.lhand_poses_mean = lhand_poses_mean
        self.rhand_poses_mean = rhand_poses_mean

        self.mesh_data = mesh_data

    @property
    def body_shapes_cnt(self) -> int:
        return self.body_shape_dirs.shape[-1]

    @property
    def expr_shapes_cnt(self) -> int:
        return 0 if self.expr_shape_dirs is None else\
            self.expr_shape_dirs.shape[-1]

    @property
    def hand_joints_cnt(self) -> int:
        return 0 if self.lhand_poses_mean is None else self.lhand_poses_mean.shape[-2]

    def get_model_config(self) -> ModelConfig:
        return ModelConfig(
            body_shapes_cnt=self.body_shapes_cnt,
            expr_shapes_cnt=self.expr_shapes_cnt,
            body_joints_cnt=self.body_joints_cnt,
            jaw_joints_cnt=self.jaw_joints_cnt,
            eye_joints_cnt=self.eye_joints_cnt,
            hand_joints_cnt=self.hand_joints_cnt,
        )

    @staticmethod
    def from_file(
        *,
        model_data_path: os.PathLike,
        model_config: ModelConfig,
        device: torch.device,
    ) -> typing.Self:
        model_data = utils.read_pickle(model_data_path)

        kin_tree_table = model_data["kintree_table"]

        kin_tree_links = [
            (int(kin_tree_table[0, j]), int(kin_tree_table[1, j]))
            for j in range(kin_tree_table.shape[1])]

        kin_tree = kin_utils.KinTree.from_links(kin_tree_links, 2**32-1)
        # joints_cnt = J

        J = kin_tree.joints_cnt

        # ---

        vertex_positions = torch.from_numpy(model_data["v_template"]) \
            .to(device, utils.FLOAT)
        # [V, 3]

        V = utils.check_shapes(vertex_positions, (-1, 3))

        # ---

        pose_dirs = torch.from_numpy(model_data["posedirs"]) \
            .to(device, utils.FLOAT)

        utils.check_shapes(pose_dirs, (V, 3, (J - 1) * 3 * 3))

        # ---

        lbs_weights = torch.from_numpy(model_data["weights"]) \
            .to(device, utils.FLOAT)

        utils.check_shapes(lbs_weights, (V, J))

        # ---

        joint_regressor = torch.from_numpy(model_data["J_regressor"]) \
            .to(device, utils.FLOAT)

        utils.check_shapes(joint_regressor, (J, V))

        # ---

        def get_shape_dirs(shape_dirs: torch.Tensor, shape_dirs_cnt: int) \
                -> torch.Tensor:
            K = utils.check_shapes(shape_dirs, (V, 3, -1))

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

            return ret.to(device, utils.FLOAT)

        shape_dirs = torch.from_numpy(model_data["shapedirs"])

        body_shape_dirs = get_shape_dirs(
            shape_dirs[:, :, :BODY_SHAPES_SPACE_DIM],
            model_config.body_shapes_cnt)

        expr_shape_dirs = get_shape_dirs(
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
                .to(device, utils.FLOAT)

            TV = utils.check_shapes(texture_vertex_positions, (..., -1, 2))
        else:
            texture_vertex_positions = None
            TV = 0

        # ---

        faces = torch.from_numpy(model_data["f"]).to(torch.long, device)

        F = utils.check_shapes(faces, (..., -1, 3))

        # ---

        if "ft" in model_data:
            texture_faces = torch.from_numpy(model_data["ft"]) \
                .to(torch.long, device)

            utils.check_shapes(texture_faces, (..., F, 3))
        else:
            texture_faces = None

        # ---

        if "hands_meanl" in model_data:
            lhand_poses_mean = torch.from_numpy(
                model_data["hands_meanl"]) \
                .reshape((-1, 3))[-model_config.hand_joints_cnt:, :] \
                .to(device, utils.FLOAT)
        else:
            lhand_poses_mean = None

        if "hands_meanr" in model_data:
            rhand_poses_mean = torch.from_numpy(
                model_data["hands_meanr"]) \
                .reshape((-1, 3))[-model_config.hand_joints_cnt:, :] \
                .to(device, utils.FLOAT)
        else:
            rhand_poses_mean = None

        # ---

        mesh_data = mesh_utils.MeshData.from_face_vertex_adj_list(
            V, faces, device)

        # ---

        return ModelData(
            kin_tree=kin_tree,

            body_joints_cnt=model_config.body_joints_cnt,
            jaw_joints_cnt=model_config.jaw_joints_cnt,
            eye_joints_cnt=model_config.eye_joints_cnt,

            vertex_positions=vertex_positions,
            vertex_normals=mesh_utils.get_area_weighted_vertex_normals(
                faces=faces,
                vertex_positions=vertex_positions,
            ).to(device, utils.FLOAT),

            texture_vertex_positions=texture_vertex_positions,

            faces=faces,
            texture_faces=texture_faces,

            body_shape_dirs=body_shape_dirs,
            expr_shape_dirs=expr_shape_dirs,

            body_shape_joint_regressor=body_shape_joint_regressor,
            expr_shape_joint_regressor=expr_shape_joint_regressor,

            joint_ts_mean=joint_ts_mean,

            pose_dirs=pose_dirs,
            lbs_weights=lbs_weights,

            lhand_poses_mean=lhand_poses_mean,
            rhand_poses_mean=rhand_poses_mean,

            mesh_data=mesh_data,
        )

    @property
    def device(self):
        return self.vertex_positions.device

    def to(self, *args, **kwargs) -> typing.Self:
        d = {
            "kin_tree": self.kin_tree,

            "vertex_positions": None,
            "vertex_normals": None,

            "texture_vertex_positions": None,

            "faces": None,
            "texture_faces": None,

            "body_shape_dirs": None,
            "expr_shape_dirs": None,

            "body_shape_joint_regressor": None,
            "expr_shape_joint_regressor": None,

            "body_joints_cnt": self.body_joints_cnt,
            "jaw_joints_cnt": self.jaw_joints_cnt,
            "eye_joints_cnt": self.eye_joints_cnt,

            "joint_ts_mean": None,

            "pose_dirs": None,

            "lbs_weights": None,

            "lhand_poses_mean": None,
            "rhand_poses_mean": None,

            "mesh_data": None,
        }

        for key, val in d.items():
            if val is None:
                cur_val = getattr(self, key)
                d[key] = None if cur_val is None else \
                    cur_val.to(*args, **kwargs)

        return ModelData(**d)
