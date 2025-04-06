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
@dataclasses.dataclass
class ModelDataMidpointSubdivisionResult:
    mesh_subdivision_result: mesh_utils.MeshSubdivisionResult
    tex_mesh_subdivision_result: mesh_utils.MeshSubdivisionResult
    model_data: ModelData


@beartype
@dataclasses.dataclass
class ModelDataExtractionResult:
    mesh_data_extraction_result: mesh_utils.MeshExtractionResult
    tex_mesh_data_extraction_result: mesh_utils.MeshExtractionResult
    model_data: ModelData


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
        vert_nor: typing.Optional[torch.Tensor],  # [..., V, 3]

        tex_vert_pos: torch.Tensor,
        # [..., TV, 2]

        body_shape_vert_dir: torch.Tensor,
        # [..., V, 3, BS]

        expr_shape_vert_dir: typing.Optional[torch.Tensor] = None,
        # [..., V, 3, ES]

        body_shape_joint_dir: torch.Tensor,
        # [..., J, 3, BS]

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

        self.vert_pos = vert_pos  # [..., V, 3]
        self.vert_nor = vert_nor  # [..., V, 3]

        self.tex_vert_pos = tex_vert_pos  # [..., TV, 2]

        self.body_shape_vert_dir = body_shape_vert_dir  # [..., V, 3, BS]
        self.expr_shape_vert_dir = expr_shape_vert_dir  # [..., V, 3, ES]

        self.body_shape_joint_dir = body_shape_joint_dir  # [..., J, 3, BS]
        self.expr_shape_joint_dir = expr_shape_joint_dir  # [..., J, 3, ES]

        self.joint_t_mean = joint_t_mean  # [..., J, 3]

        self.pose_vert_dir = pose_vert_dir  # [..., V, 3, (J - 1) * 3 * 3]

        self.lbs_weight = lbs_weight  # [..., V, J]

        self.lhand_pose_mean = lhand_pose_mean  # [..., HANDJ, 3]
        self.rhand_pose_mean = rhand_pose_mean  # [..., HANDJ, 3]

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

        assert BODYJ + JAWJ + EYEJ * 2 + HANDJ * 2 == J

        # ---

        def try_fetch_int(field_name: str):
            val = model_data.get(field_name, None)

            return None if val is None else \
                torch.from_numpy(model_data[field_name]).to(torch.long)

        def try_fetch_float(field_name: str):
            val = model_data.get(field_name, None)

            return None if val is None else \
                torch.from_numpy(model_data[field_name]).to(torch.float64)

        vert_pos = try_fetch_float("v_template")
        V = utils.check_shapes(vert_pos, (-1, 3))

        pose_vert_dir = try_fetch_float("posedirs")
        utils.check_shapes(pose_vert_dir, (V, 3, (J - 1) * 3 * 3))

        lbs_weight = try_fetch_float("weights")
        utils.check_shapes(lbs_weight, (V, J))

        joint_regressor = try_fetch_float("J_regressor")
        utils.check_shapes(joint_regressor, (J, V))

        shape_dirs = try_fetch_float("shapedirs")

        tex_vert_pos = try_fetch_float("vt")
        TV = utils.check_shapes(tex_vert_pos, (-1, 2))

        faces = try_fetch_int("f")
        F = utils.check_shapes(faces, (-1, 3))

        tex_faces = try_fetch_int("ft")
        utils.check_shapes(tex_faces, (F, 3))

        lhand_pose_mean = try_fetch_float("hands_meanl")

        if lhand_pose_mean is not None:
            lhand_pose_mean = lhand_pose_mean.reshape(-1, 3)[-HANDJ:, :]
            utils.check_shapes(lhand_pose_mean, (HANDJ, 3))

        rhand_pose_mean = try_fetch_float("hands_meanr")

        if rhand_pose_mean is not None:
            rhand_pose_mean = rhand_pose_mean.reshape(-1, 3)[-HANDJ:, :]
            utils.check_shapes(rhand_pose_mean, (HANDJ, 3))

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

            return ret.to(torch.float64)

        body_shape_vert_dir = get_shape_dirs(
            shape_dirs[:, :, :BODY_SHAPES_SPACE_DIM], BS)

        expr_shape_vert_dir = get_shape_dirs(
            shape_dirs[:, :, BODY_SHAPES_SPACE_DIM:], ES)

        if body_shape_vert_dir is not None:
            utils.check_shapes(body_shape_vert_dir, (V, 3, BS))

        if expr_shape_vert_dir is not None:
            utils.check_shapes(expr_shape_vert_dir, (V, 3, ES))

        # ---

        joint_t_mean = torch.einsum(
            "...jv,...vx->...jx",
            joint_regressor,
            vert_pos,
        )

        body_shape_joint_dir = torch.einsum(
            "...jv,...vxb->...jxb",
            joint_regressor,
            body_shape_vert_dir,
        )

        if expr_shape_vert_dir is None:
            expr_shape_joint_dir = None
        else:
            expr_shape_joint_dir = torch.einsum(
                "...jv,...vxb->...jxb",
                joint_regressor,
                expr_shape_vert_dir,
            )

        # ---

        if lhand_pose_mean is not None:
            lhand_pose_mean = lhand_pose_mean \
                .reshape(-1, 3)[-model_config.hand_joints_cnt:, :]

        if rhand_pose_mean is not None:
            rhand_pose_mean = rhand_pose_mean \
                .reshape(-1, 3)[-model_config.hand_joints_cnt:, :]

        # ---

        mesh_data = mesh_utils.MeshData.from_faces(
            V, faces, device)

        tex_mesh_data = mesh_utils.MeshData.from_faces(
            TV, tex_faces, device)

        # ---

        """
        vert_nor = mesh_utils.get_area_weighted_vert_nor(
            faces=mesh_data.f_to_vvv.to(vert_pos.device),
            vert_pos=vert_pos,
        )
        """
        vert_nor = None

        # ---

        def f(obj):
            return None if obj is None else obj.to(device, dtype)

        # ---

        return ModelData(
            kin_tree=kin_tree,

            mesh_data=mesh_data,
            tex_mesh_data=tex_mesh_data,

            body_joints_cnt=model_config.body_joints_cnt,
            jaw_joints_cnt=model_config.jaw_joints_cnt,
            eye_joints_cnt=model_config.eye_joints_cnt,

            vert_pos=f(vert_pos),

            vert_nor=f(vert_nor),

            tex_vert_pos=tex_vert_pos,

            body_shape_vert_dir=f(body_shape_vert_dir),
            expr_shape_vert_dir=f(expr_shape_vert_dir),

            body_shape_joint_dir=f(body_shape_joint_dir),
            expr_shape_joint_dir=f(expr_shape_joint_dir),

            joint_t_mean=f(joint_t_mean),

            pose_vert_dir=f(pose_vert_dir),
            lbs_weight=f(lbs_weight),

            lhand_pose_mean=f(lhand_pose_mean),
            rhand_pose_mean=f(rhand_pose_mean),
        )

    @staticmethod
    def from_file(
        model_data_path: os.PathLike,
        *,
        dtype: typing.Optional[torch.dtype] = None,
        device: typing.Optional[torch.device] = None,
    ) -> ModelData:
        model_data = utils.read_pickle(model_data_path)

        assert dtype is None or dtype.is_floating_point

        def try_fetch_int(field_name: str):
            val = model_data.get(field_name, None)

            return None if val is None else \
                torch.from_numpy(model_data[field_name]) \
                .to(device, torch.long)

        def try_fetch_float(field_name: str):
            val = model_data.get(field_name, None)

            return None if val is None else \
                torch.from_numpy(model_data[field_name]) \
                .to(device, dtype)

        vert_pos = try_fetch_float("vert_pos")
        vert_nor = try_fetch_float("vert_nor")

        tex_vert_pos = try_fetch_float("tex_vert_pos")

        V, TV = utils.check_shapes(
            vert_pos, (..., -1, 3),
            vert_nor, (..., -1, 3),
            tex_vert_pos, (..., -2, 2),
        )

        return ModelData(
            kin_tree=kin_utils.KinTree.from_parents(
                model_data["joint_parents"]),

            mesh_data=mesh_utils.MeshData.from_faces(
                V, try_fetch_int("faces"), device),
            tex_mesh_data=mesh_utils.MeshData.from_faces(
                TV, try_fetch_int("tex_faces"), device),

            body_joints_cnt=model_data["body_joints_cnt"],
            jaw_joints_cnt=model_data["jaw_joints_cnt"],
            eye_joints_cnt=model_data["eye_joints_cnt"],

            vert_pos=vert_pos,
            vert_nor=vert_nor,

            tex_vert_pos=tex_vert_pos,

            body_shape_vert_dir=try_fetch_float("body_shape_vert_dir"),
            expr_shape_vert_dir=try_fetch_float("expr_shape_vert_dir"),

            body_shape_joint_dir=try_fetch_float("body_shape_joint_dir"),
            expr_shape_joint_dir=try_fetch_float("expr_shape_joint_dir"),

            joint_t_mean=try_fetch_float("joint_t_mean"),

            pose_vert_dir=try_fetch_float("pose_vert_dir"),
            lbs_weight=try_fetch_float("lbs_weight"),

            lhand_pose_mean=try_fetch_float("lhand_pose_mean"),
            rhand_pose_mean=try_fetch_float("rhand_pose_mean"),
        )

    def save(self, path: os.PathLike):
        path = utils.to_pathlib_path(path)

        def to_int_np(x: torch.Tensor):
            return None if x is None else \
                np.array(x.numpy(force=True), dtype=np.int32, copy=True)

        def to_float_np(x: torch.Tensor):
            return None if x is None else \
                np.array(x.numpy(force=True), dtype=np.float64, copy=True)

        utils.write_pickle(path, {
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

            "body_shape_vert_dir": None,
            "expr_shape_vert_dir": None,

            "body_shape_joint_dir": None,
            "expr_shape_joint_dir": None,

            "body_joints_cnt": self.body_joints_cnt,
            "jaw_joints_cnt": self.jaw_joints_cnt,
            "eye_joints_cnt": self.eye_joints_cnt,

            "joint_t_mean": None,

            "pose_vert_dir": None,

            "lbs_weight": None,

            "lhand_pose_mean": None,
            "rhand_pose_mean": None,
        }

        for key, val in d.items():
            if val is None:
                cur_val = getattr(self, key)
                d[key] = None if cur_val is None else \
                    cur_val.to(*args, **kwargs)

        return ModelData(**d)

    def midpoint_subdivide(
        self,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> ModelDataMidpointSubdivisionResult:
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

        if self.vert_nor is None:
            new_vert_nor = None
        else:
            new_vert_nor = utils.vec_normed(
                (self.vert_nor[..., vert_src_table[:, 0], :] +
                 self.vert_nor[..., vert_src_table[:, 1], :]) / 2)
            # [..., V_, 3]

        new_tex_vert_pos = (
            self.tex_vert_pos[..., tex_vert_src_table[:, 0], :] +
            self.tex_vert_pos[..., tex_vert_src_table[:, 1], :]) / 2
        # [..., TV_, 2]

        new_pose_vert_dir = (
            self.pose_vert_dir[..., vert_src_table[:, 0], :, :] +
            self.pose_vert_dir[..., vert_src_table[:, 1], :, :]) / 2
        # [..., V_, 3, (J - 1) * 3 * 3]

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

        return ModelDataMidpointSubdivisionResult(
            mesh_subdivision_result=mesh_subdivision_result,
            tex_mesh_subdivision_result=tex_mesh_subdivision_result,
            model_data=model_data,
        )

    def extract(
        self,
        target_faces: typing.Sequence[int],
    ) -> ModelDataExtractionResult:
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

        if self.vert_nor is None:
            new_vert_nor = None
        else:
            new_vert_nor = self.vert_nor[..., vert_src_table, :]
            # [..., V_, 3]

        new_tex_vert_pos = self.tex_vert_pos[..., tex_vert_src_table, :]
        # [..., TV_, 2]

        if self.body_shape_vert_dir is None:
            new_body_shape_vert_dir = None
        else:
            new_body_shape_vert_dir = \
                self.body_shape_vert_dir[..., vert_src_table, :, :]

        if self.expr_shape_vert_dir is None:
            new_expr_shape_vert_dir = None
        else:
            new_expr_shape_vert_dir = \
                self.expr_shape_vert_dir[..., vert_src_table, :, :]

        new_pose_vert_dir = self.pose_vert_dir[..., vert_src_table, :, :]
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

            body_shape_vert_dir=new_body_shape_vert_dir,
            expr_shape_vert_dir=new_expr_shape_vert_dir,

            body_shape_joint_dir=self.body_shape_joint_dir,
            expr_shape_joint_dir=self.expr_shape_joint_dir,

            joint_t_mean=self.joint_t_mean,

            pose_vert_dir=new_pose_vert_dir,
            lbs_weight=new_lbs_weight,

            lhand_pose_mean=self.lhand_pose_mean,
            rhand_pose_mean=self.rhand_pose_mean,
        )

        return ModelDataExtractionResult(
            mesh_data_extraction_result=mesh_data_extraction_result,
            tex_mesh_data_extraction_result=tex_mesh_data_extraction_result,
            model_data=model_data,
        )

    def show(self):
        self.mesh_data.show(self.vert_pos)
