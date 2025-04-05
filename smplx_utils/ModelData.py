from __future__ import annotations

import dataclasses
import os
import typing

import numpy as np
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

        mesh_data: mesh_utils.MeshData,
        tex_mesh_data: mesh_utils.MeshData,

        body_joints_cnt: int,  # BJ
        jaw_joints_cnt: int,  # JJ
        eye_joints_cnt: int,  # EYEJ

        vert_pos: torch.Tensor,  # [..., V, 3]
        vert_nor: torch.Tensor,  # [..., V, 3]

        tex_vert_pos: torch.Tensor,
        # [..., TV, 2]

        body_shape_vert_dir: torch.Tensor,  # [..., V, 3, BS]
        expr_shape_vert_dir: typing.Optional[torch.Tensor] = None,
        # [..., V, 3, ES]

        body_shape_joint_dir: torch.Tensor,  # [..., J, 3, BS]

        expr_shape_joint_dir: typing.Optional[torch.Tensor] = None,
        # [..., J, 3, ES]

        joint_t_mean: torch.Tensor,  # [..., J, 3]

        pose_vert_dir: torch.Tensor,  # [..., V, 3, (J - 1) * 3 * 3]

        lbs_weight: torch.Tensor,  # [..., V, J]

        lhand_pose_mean: typing.Optional[torch.Tensor] = None,
        # [..., HANDJ, 3]

        rhand_pose_mean: typing.Optional[torch.Tensor] = None,
        # [..., HANDJ, 3]
    ):
        J = kin_tree.joints_cnt

        BODYJ = body_joints_cnt
        JAWJ = jaw_joints_cnt
        EYEJ = eye_joints_cnt

        V = mesh_data.verts_cnt
        TV = tex_mesh_data.verts_cnt

        F = mesh_data.faces_cnt
        assert tex_mesh_data.faces_cnt == F

        utils.check_shapes(
            vert_pos, (..., V, 3),
            vert_nor, (..., V, 3),

            tex_vert_pos, (..., TV, 2),
            pose_vert_dir, (..., V, 3, (J - 1) * 3 * 3),

            lbs_weight, (..., V, J),
        )

        BS = 0 if body_shape_vert_dir is None else \
            utils.check_shapes(body_shape_vert_dir, (..., V, 3, -1))

        ES = 0 if expr_shape_vert_dir is None else \
            utils.check_shapes(expr_shape_vert_dir, (..., V, 3, -1))

        utils.check_shapes(body_shape_joint_dir, (..., J, 3, BS))

        if expr_shape_joint_dir is not None:
            utils.check_shapes(expr_shape_joint_dir, (..., J, 3, ES))

        assert 1 <= BODYJ
        assert 0 <= JAWJ
        assert 0 <= EYEJ

        if lhand_pose_mean is None:
            assert rhand_pose_mean is None
            HANDJ = 0
        else:
            assert rhand_pose_mean is not None
            assert lhand_pose_mean.shape == rhand_pose_mean.shape
            HANDJ = utils.check_shapes(lhand_pose_mean, (..., -1, 3))

        assert BODYJ + JAWJ + EYEJ * 2 + HANDJ * 2 == J

        # ---

        self.kin_tree = kin_tree

        self.mesh_data = mesh_data
        self.tex_mesh_data = tex_mesh_data

        self.body_joints_cnt = body_joints_cnt
        self.jaw_joints_cnt = jaw_joints_cnt
        self.eye_joints_cnt = eye_joints_cnt

        self.vert_pos = vert_pos
        self.vert_nor = vert_nor

        self.tex_vert_pos = tex_vert_pos

        self.body_shape_vert_dir = body_shape_vert_dir
        self.expr_shape_vert_dir = expr_shape_vert_dir

        self.body_shape_joint_dir = body_shape_joint_dir
        self.expr_shape_joint_dir = expr_shape_joint_dir

        self.joint_t_mean = joint_t_mean

        self.pose_vert_dir = pose_vert_dir

        self.lbs_weight = lbs_weight

        self.lhand_pose_mean = lhand_pose_mean
        self.rhand_pose_mean = rhand_pose_mean

    @property
    def hand_joints_cnt(self) -> int:
        return 0 if self.lhand_pose_mean is None else self.lhand_pose_mean.shape[-2]

    @property
    def joints_cnt(self) -> int:
        return self.kin_tree.joints_cnt

    @property
    def verts_cnt(self) -> int:
        return self.vert_pos.shape[0]

    @property
    def tex_verts_cnt(self) -> int:
        return self.tex_vert_pos.shape[0]

    @property
    def faces_cnt(self) -> int:
        return self.mesh_data.faces_cnt

    @property
    def body_shapes_cnt(self) -> int:
        return self.body_shape_vert_dir.shape[-1]

    @property
    def expr_shapes_cnt(self) -> int:
        return 0 if self.expr_shape_vert_dir is None else\
            self.expr_shape_vert_dir.shape[-1]

    @staticmethod
    def from_origin_file(
        *,
        model_data_path: os.PathLike,
        model_config: ModelConfig,
        dtype: typing.Optional[torch.dtype] = None,
        device: typing.Optional[torch.device] = None,
    ) -> ModelData:
        model_data = utils.read_pickle(model_data_path)

        assert dtype is None or dtype.is_floating_point

        # ---

        kin_tree_table = model_data["kintree_table"]

        kin_tree_links = [
            (int(kin_tree_table[0, j]), int(kin_tree_table[1, j]))
            for j in range(kin_tree_table.shape[1])]

        kin_tree = kin_utils.KinTree.from_links(kin_tree_links, 2**32-1)
        # joints_cnt = J

        J = kin_tree.joints_cnt

        # ---

        BS = model_config.body_shapes_cnt
        ES = model_config.expr_shapes_cnt

        BODYJ = model_config.body_joints_cnt
        JAWJ = model_config.jaw_joints_cnt
        EYEJ = model_config.eye_joints_cnt
        HANDJ = model_config.hand_joints_cnt

        assert BODYJ + JAWJ + EYEJ + HANDJ * 2 == J

        # ---

        def try_fetch_int(field_name: str):
            return torch.from_numpy(model_data[field_name])\
                if field_name in model_data else None

        def try_fetch_float(field_name: str):
            return torch.from_numpy(model_data[field_name]).to(torch.float64) \
                if field_name in model_data else None

        vert_pos = try_fetch_float("v_template")
        pose_vert_dir = try_fetch_float("posedirs")
        lbs_weight = try_fetch_float("weight")
        joint_regressor = try_fetch_float("J_regressor")
        shape_dirs = try_fetch_float("shapedirs")

        tex_vert_pos = try_fetch_float("vt")

        faces = try_fetch_int("f")
        tex_faces = try_fetch_int("ft")

        lhand_poses_mean = try_fetch_float("hands_meanl")[-HANDJ:, :]
        rhand_poses_mean = try_fetch_float("hands_meanr")[-HANDJ:, :]

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

            return ret.to(torch.float64)

        body_shape_dirs = get_shape_dirs(
            shape_dirs[:, :, :BODY_SHAPES_SPACE_DIM], BS)

        expr_shape_dirs = get_shape_dirs(
            shape_dirs[:, :, BODY_SHAPES_SPACE_DIM:], ES)

        # ---

        V, TV, F = -1, -2, -3, 4

        V, TV, F = utils.check_shapes(
            vert_pos, (V, 3),
            pose_vert_dir, (V, 3, (J - 1) * 3 * 3),
            lbs_weight, (V, J),
            joint_regressor, (J, V),
            tex_vert_pos, (TV, 2),
            faces, (F, 3),
            tex_faces, (F, 3),

            lhand_poses_mean, (HANDJ, 3),
            rhand_poses_mean, (HANDJ, 3),

            body_shape_dirs, (V, 3, BS),
            expr_shape_dirs, (V, 3, ES),
        )

        # ---

        joint_t_mean = torch.einsum(
            "...jv,...vx->...jx",
            joint_regressor,
            vert_pos,
        )

        body_shape_joint_dir = torch.einsum(
            "...jv,...vxb->...jxb",
            joint_regressor,
            body_shape_dirs,
        )

        if expr_shape_dirs is None:
            expr_shape_joint_dir = None
        else:
            expr_shape_joint_dir = torch.einsum(
                "...jv,...vxb->...jxb",
                joint_regressor,
                expr_shape_dirs,
            )

        # ---

        if lhand_poses_mean is not None:
            lhand_poses_mean = lhand_poses_mean \
                .reshape((-1, 3))[-model_config.hand_joints_cnt:, :]

        if rhand_poses_mean is not None:
            rhand_poses_mean = rhand_poses_mean \
                .reshape((-1, 3))[-model_config.hand_joints_cnt:, :]

        # ---

        mesh_data = mesh_utils.MeshData.from_faces(
            V, faces, device)

        tex_mesh_data = mesh_utils.MeshData.from_faces(
            TV, tex_faces, device)

        # ---

        dd = (device, dtype)

        return ModelData(
            kin_tree=kin_tree,

            mesh_data=mesh_data,
            tex_mesh_data=tex_mesh_data,

            body_joints_cnt=model_config.body_joints_cnt,
            jaw_joints_cnt=model_config.jaw_joints_cnt,
            eye_joints_cnt=model_config.eye_joints_cnt,

            vert_pos=vert_pos.to(*dd),

            vert_nor=mesh_utils.get_area_weighted_vert_nor(
                faces=faces,
                vert_pos=vert_pos,
            ).to(*dd),

            tex_vert_pos=tex_vert_pos.to(*dd),

            body_shape_vert_dir=body_shape_dirs.to(*dd),
            expr_shape_vert_dir=expr_shape_dirs.to(*dd),

            body_shape_joint_dir=body_shape_joint_dir.to(*dd),
            expr_shape_joint_dir=expr_shape_joint_dir.to(*dd),

            joint_t_mean=joint_t_mean.to(*dd),

            pose_vert_dir=pose_vert_dir.to(*dd),
            lbs_weight=lbs_weight.to(*dd),

            lhand_pose_mean=lhand_poses_mean.to(*dd),
            rhand_pose_mean=rhand_poses_mean.to(*dd),
        )

    def from_file(
        self,
        model_data_path: os.PathLike,
        *,
        dtype: typing.Optional[torch.dtype] = None,
        device: typing.Optional[torch.device] = None,
    ) -> ModelData:
        model_data = utils.read_pickle(model_data_path)

        assert dtype is None or dtype.is_floating_point

        def fetch_tensor(field_name: str):
            return torch.from_numpy(model_data[field_name]).to(device, dtype)

        return ModelData(
            kin_tree=kin_utils.KinTree.from_parents(
                model_data["joint_parents"]),

            mesh_data=mesh_utils.MeshData.from_faces(
                fetch_tensor("faces")),
            tex_mesh_data=mesh_utils.MeshData.from_faces(
                fetch_tensor("tex_faces")),

            body_joints_cnt=model_data["body_joints_cnt"],
            jaw_joints_cnt=model_data["jaw_joints_cnt"],
            eye_joints_cnt=model_data["eye_joints_cnt"],

            vert_pos=fetch_tensor("vert_pos"),
            vert_nor=fetch_tensor("vert_nor"),

            tex_vert_pos=fetch_tensor("tex_vert_pos"),

            body_shape_vert_dir=fetch_tensor("body_shape_vert_dir"),
            expr_shape_vert_dir=fetch_tensor("expr_shape_vert_dir"),

            body_shape_joint_dir=fetch_tensor("body_shape_joint_dir"),
            expr_shape_joint_dir=fetch_tensor("expr_shape_joint_dir"),

            joint_t_mean=fetch_tensor("joint_t_mean"),

            pose_vert_dir=fetch_tensor("pose_vert_dir"),

            lhand_pose_mean=fetch_tensor("lhand_pose_mean"),
            rhand_pose_mean=fetch_tensor("rhand_pose_mean"),
        )

    def save(self, path: os.PathLike):
        path = utils.to_pathlib_path(path)

        def to_int_np(x: torch.Tensor):
            return None if x is None else \
                np.array(x.numpy(force=True), dtype=np.int32, copy=True)

        def to_float_np(x: torch.Tensor):
            return None if x is None else \
                np.array(x.numpy(force=True), dtype=np.float64, copy=True)

        utils.write_pickle({
            "joint_parents": self.kin_tree.parents,

            "body_joints_cnt": self.body_joints_cnt,
            "jaw_joints_cnt": self.jaw_joints_cnt,
            "eye_joints_cnt": self.eye_joints_cnt,

            "vert_pos": to_float_np(self.vert_pos),
            "vert_nor": to_float_np(self.vert_nor),

            "tex_vert_pos": to_float_np(self.tex_vert_pos),

            "body_shape_vert_dir": to_float_np(self.body_shape_vert_dir),
            "expr_shape_vert_dir": to_float_np(self.expr_shape_vert_dir),

            "body_shape_joint_dir": to_float_np(self.body_shape_joint_dir),
            "expr_shape_joint_dir": to_float_np(self.expr_shape_joint_dir),

            "joint_t_mean": to_float_np(self.joint_t_mean),
            "pose_vert_dir": to_float_np(self.pose_vert_dir),

            "lbs_weight": to_float_np(self.lbs_weight),

            "lhand_pose_mean": to_float_np(self.lhand_pose_mean),
            "rhand_pose_mean": to_float_np(self.rhand_pose_mean),

            "faces": to_int_np(self.mesh_data.f_to_vvv),
            "tex_faces": to_int_np(self.tex_mesh_data.f_to_vvv),
        })

    @property
    def device(self):
        return self.vert_pos.device

    def to(self, *args, **kwargs) -> ModelData:
        d = {
            "kin_tree": self.kin_tree,

            "mesh_data": None,
            "tex_mesh_data": None,

            "vert_pos": None,
            "vert_nor": None,

            "tex_vert_pos": None,

            "body_shape_dirs": None,
            "expr_shape_dirs": None,

            "body_shape_joint_dir": None,
            "expr_shape_joint_dir": None,

            "body_joints_cnt": self.body_joints_cnt,
            "jaw_joints_cnt": self.jaw_joints_cnt,
            "eye_joints_cnt": self.eye_joints_cnt,

            "joint_t_mean": None,

            "pose_dirs": None,

            "lbs_weights": None,

            "lhand_poses_mean": None,
            "rhand_poses_mean": None,
        }

        for key, val in d.items():
            if val is None:
                cur_val = getattr(self, key)
                d[key] = None if cur_val is None else \
                    cur_val.to(*args, **kwargs)

        return ModelData(**d)

    @beartype
    @dataclasses.dataclass
    class MidpointSubdivisionResult:
        mesh_subdivision_result: mesh_utils.MeshData.SubdivisionResult
        tex_mesh_subdivision_result: mesh_utils.MeshData.SubdivisionResult
        model_data: ModelData

    def midpoint_subdivide(
        self,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> MidpointSubdivisionResult:
        mesh_subdivision_result = self.mesh_data.subdivide(
            target_faces=target_faces)

        tex_mesh_subdivision_result = self.mesh_data.subdivide(
            target_faces=target_faces)

        new_mesh_data = mesh_subdivision_result.mesh_data
        new_tex_mesh_data = tex_mesh_subdivision_result.mesh_data

        vert_src_table = mesh_subdivision_result.vert_src_table
        # [V_, 2]

        tex_vert_src_table = tex_mesh_subdivision_result.vert_src_table
        # [TV_, 2]

        new_vert_pos = (
            self.vert_pos[..., vert_src_table[:, 0], :] +
            self.vert_pos[..., vert_src_table[:, 1], :]) / 2
        # [..., V_, 3]

        new_vert_nor = utils.vec_normed(
            (self.vert_nor[..., vert_src_table[:, 0], :] +
             self.vert_nor[..., vert_src_table[:, 1], :]) / 2)
        # [..., V_, 3]

        new_tex_vert_pos = (
            self.tex_vert_pos[..., tex_vert_src_table[:, 0], :] +
            self.tex_vert_pos[..., tex_vert_src_table[:, 1], :]) / 2
        # [..., TV_, 2]

        new_pose_vert_dir = (
            self.pose_vert_dir[..., vert_src_table[:, 0], :] +
            self.pose_vert_dir[..., vert_src_table[:, 1], :]) / 2
        # [..., V_, 3]

        new_lbs_weight = (
            self.lbs_weight[..., vert_src_table[:, 0], :] +
            self.lbs_weight[..., vert_src_table[:, 1], :]) / 2
        # [..., V_, 3]

        model_data = ModelData(
            kin_tree=self.kin_tree,
            mesh_data=new_mesh_data,
            tex_mesh_data=new_tex_mesh_data,

            body_joints_cnt=self.body_joints_cnt,
            jaw_joints_cnt=self.jaw_joints_cnt,
            eye_joints_cnt=self.eye_joints_cnt,

            vert_pos=new_vert_pos,
            vert_nor=new_vert_nor,

            tex_vert_pos=new_tex_vert_pos,

            body_shape_vert_dir=self.body_shape_vert_dir,
            expr_shape_vert_dir=self.expr_shape_vert_dir,

            body_shape_joint_dir=self.body_shape_joint_dir,
            expr_shape_joint_dir=self.expr_shape_joint_dir,

            joint_t_mean=self.joint_t_mean,

            pose_vert_dir=new_pose_vert_dir,
            lbs_weight=new_lbs_weight,

            lhand_pose_mean=self.lhand_pose_mean,
            rhand_pose_mean=self.rhand_pose_mean,
        )

        return ModelData.MidpointSubdivisionResult(
            mesh_subdivision_result=mesh_subdivision_result,
            tex_mesh_subdivision_result=tex_mesh_subdivision_result,
            model_data=model_data,
        )

    @beartype
    @dataclasses.dataclass
    class ExtractionResult:
        mesh_data_extraction_result: mesh_utils.MeshData.ExtractionResult
        tex_mesh_data_extraction_result: mesh_utils.MeshData.ExtractionResult
        model_data: ModelData

    def extract(
        self,
        target_faces: typing.Sequence[int],
    ) -> ExtractionResult:
        mesh_data_extraction_result = self.mesh_data.extract(
            target_faces)
        # vert_src_table[V_]
        # face_src_table[F_]

        tex_mesh_data_extraction_result = self.tex_mesh_data.extract(
            target_faces)
        # vert_src_table[TV_]
        # face_src_table[F_]

        new_mesh_data = mesh_data_extraction_result.mesh_data
        new_tex_mesh_data = tex_mesh_data_extraction_result.mesh_data

        vert_src_table = mesh_data_extraction_result.vert_src_table
        # [V_]

        tex_vert_src_table = tex_mesh_data_extraction_result.vert_src_table
        # [TV_]

        new_vert_pos = self.vert_pos[..., vert_src_table, :]
        # [..., V_, 3]

        new_vert_nor = self.vert_nor[..., vert_src_table, :]
        # [..., V_, 3]

        new_tex_vert_pos = self.tex_vert_pos[..., tex_vert_src_table, :]
        # [..., TV_, 2]

        new_pose_vert_dir = self.pose_vert_dir[..., vert_src_table, :]
        # [..., V_, 3, (J - 1) * 3 * 3]

        new_lbs_weight = self.lbs_weight[..., vert_src_table, :]
        # [..., V_, J]

        model_data = ModelData(
            kin_tree=self.kin_tree,

            mesh_data=new_mesh_data,
            tex_mesh_data=new_tex_mesh_data,

            body_joints_cnt=self.body_joints_cnt,
            jaw_joints_cnt=self.jaw_joints_cnt,
            eye_joints_cnt=self.eye_joints_cnt,

            vert_pos=new_vert_pos,
            vert_nor=new_vert_nor,

            tex_vert_pos=new_tex_vert_pos,

            body_shape_vert_dir=self.body_shape_vert_dir,
            expr_shape_vert_dir=self.expr_shape_vert_dir,

            body_shape_joint_dir=self.body_shape_joint_dir,
            expr_shape_joint_dir=self.expr_shape_joint_dir,

            joint_t_mean=self.joint_t_mean,

            pose_vert_dir=new_pose_vert_dir,
            lbs_weight=new_lbs_weight,

            lhand_pose_mean=self.lhand_pose_mean,
            rhand_pose_mean=self.rhand_pose_mean,
        )

        return ModelData.ExtractionResult(
            mesh_data_extraction_result=mesh_data_extraction_result,
            tex_mesh_data_extraction_result=tex_mesh_data_extraction_result,
            model_data=model_data,
        )
