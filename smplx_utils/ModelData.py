from __future__ import annotations

import collections
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
class ModelDataMidpointSubdivisionResult:
    mesh_subdivision_result: mesh_utils.MeshSubdivisionResult
    tex_mesh_subdivision_result: mesh_utils.MeshSubdivisionResult
    model_data: ModelData


@beartype
@dataclasses.dataclass
class ModelDataExtractionResult:
    mesh_graph_extraction_result: mesh_utils.MeshExtractionResult
    tex_mesh_graph_extraction_result: mesh_utils.MeshExtractionResult
    model_data: ModelData


@beartype
class ModelData:
    def __init__(
        self,
        *,
        kin_tree: kin_utils.KinTree = None,

        mesh_graph: mesh_utils.MeshGraph,
        tex_mesh_graph: mesh_utils.MeshGraph,

        body_joints_cnt: int = 0,  # BJ
        jaw_joints_cnt: int = 0,  # JJ
        eye_joints_cnt: int = 0,  # EYEJ

        vert_pos: torch.Tensor,  # [..., V, 3]

        tex_vert_pos: torch.Tensor,
        # [..., TV, 2]

        body_shape_vert_dir: torch.Tensor,
        # [..., V, 3, BS]

        expr_shape_vert_dir: torch.Tensor,
        # [..., V, 3, ES]

        body_shape_joint_dir: torch.Tensor,
        # [..., J, 3, BS]

        expr_shape_joint_dir: torch.Tensor,
        # [..., J, 3, ES]

        joint_t_mean: torch.Tensor,
        # [..., J, 3]

        pose_vert_dir: torch.Tensor,
        # [..., V, 3, (J - 1) * 3 * 3]

        lbs_weight: torch.Tensor,
        # [..., V, J]

        lhand_pose_mean: torch.Tensor,
        # [..., HANDJ, 3]

        rhand_pose_mean: torch.Tensor,
        # [..., HANDJ, 3]
    ):
        J = kin_tree.joints_cnt

        BODYJ = body_joints_cnt
        JAWJ = jaw_joints_cnt
        EYEJ = eye_joints_cnt

        assert 0 <= BODYJ
        assert 0 <= JAWJ
        assert 0 <= EYEJ

        V, F = mesh_graph.verts_cnt, mesh_graph.faces_cnt
        TV, TF = tex_mesh_graph.verts_cnt, tex_mesh_graph.faces_cnt

        assert F == 0 or TF == 0 or F == TF

        HANDJ, BS, ES = -1, -2, -3

        HANDJ, BS, ES = utils.check_shapes(
            vert_pos, (..., V, 3),

            tex_vert_pos, (..., TV, 2),

            body_shape_vert_dir, (..., V, 3, BS),
            expr_shape_vert_dir, (..., V, 3, ES),
            body_shape_joint_dir, (..., J, 3, BS),
            expr_shape_joint_dir, (..., J, 3, ES),

            pose_vert_dir, (..., V, 3, (J - 1) * 3 * 3),

            lbs_weight, (..., V, J),

            lhand_pose_mean, (..., HANDJ, 3),
            rhand_pose_mean, (..., HANDJ, 3),
        )

        assert BODYJ + JAWJ + EYEJ * 2 + HANDJ * 2 == J

        # ---

        self.kin_tree = kin_tree

        self.mesh_graph = mesh_graph
        self.tex_mesh_graph = tex_mesh_graph

        self.body_joints_cnt = body_joints_cnt
        self.jaw_joints_cnt = jaw_joints_cnt
        self.eye_joints_cnt = eye_joints_cnt

        self.vert_pos = vert_pos  # [..., V, 3]

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
        return self.lhand_pose_mean.shape[-2]

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
        return self.mesh_graph.faces_cnt

    @property
    def body_shapes_cnt(self) -> int:
        return 0 if self.body_shape_vert_dir is None else \
            self.body_shape_vert_dir.shape[-1]

    @property
    def expr_shapes_cnt(self) -> int:
        return 0 if self.expr_shape_vert_dir is None else \
            self.expr_shape_vert_dir.shape[-1]

    @staticmethod
    def empty(
        dtype: torch.dtype,
        device: torch.device,
    ) -> ModelData:
        def f(*shape):
            return torch.empty(shape, dtype=dtype, device=device),

        return ModelData(
            kin_tree=kin_utils.KinTree.empty(),

            mesh_graph=mesh_utils.MeshGraph.empty(),
            tex_mesh_graph=mesh_utils.MeshGraph.empty(),

            body_joints_cnt=0,
            jaw_joints_cnt=0,
            eye_joints_cnt=0,

            vert_pos=f(0, 3),
            tex_vert_pos=f(0, 3),

            body_shape_vert_dir=f(0, 3, 0),
            expr_shape_vert_dir=f(0, 3, 0),

            body_shape_joint_dir=f(0, 3, 0),
            expr_shape_joint_dir=f(0, 3, 0),

            joint_t_mean=f(0, 3),

            pose_vert_dir=f(0, 3, 0),
            lbs_weight=f(0, 0),

            lhand_pose_mean=f(0, 3),
            rhand_pose_mean=f(0, 3),
        )

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

        shape_vert_dir = try_fetch_float("shapedirs")

        tex_vert_pos = try_fetch_float("vt")
        TV = utils.check_shapes(tex_vert_pos, (-1, 2))

        faces = try_fetch_int("f")
        F = utils.check_shapes(faces, (-1, 3))

        tex_faces = try_fetch_int("ft")
        utils.check_shapes(tex_faces, (F, 3))

        def get_hand_pose_mean(hand_pose_mean: torch.Tensor) -> torch.Tensor:
            if HANDJ == 0 or hand_pose_mean is None:
                return torch.empty(
                    (HANDJ, 3), dtype=torch.float64, device=device)

            hand_pose_mean = hand_pose_mean.reshape(-1, 3)

            if hand_pose_mean.shape[0] < HANDJ:
                hand_pose_mean = torch.nn.functional.pad(
                    hand_pose_mean,
                    (0, 0, 0, HANDJ - hand_pose_mean.shape[0]),
                    "constant",
                    0
                )
            else:
                hand_pose_mean = hand_pose_mean[-HANDJ:, :]

        lhand_pose_mean = get_hand_pose_mean(try_fetch_float("hands_meanl"))
        rhand_pose_mean = get_hand_pose_mean(try_fetch_float("hands_meanr"))

        utils.check_shapes(
            rhand_pose_mean, (HANDJ, 3),
            lhand_pose_mean, (HANDJ, 3),
        )

        # ---

        def get_shape_dir(shape_dir: torch.Tensor, shape_dirs_cnt: int) \
                -> torch.Tensor:
            K = utils.check_shapes(shape_dir, (V, 3, -1))

            assert 0 <= shape_dirs_cnt

            if shape_dirs_cnt == 0:
                return torch.empty(
                    (V, 3, 0), dtype=torch.float64, device=device)

            if shape_dirs_cnt <= K:
                ret = shape_dir[:, :, :shape_dirs_cnt]
            else:
                ret = torch.nn.functional.pad(
                    shape_dir,
                    (0, shape_dirs_cnt - K),
                    "constant",
                    0
                )

            return ret.to(torch.float64)

        body_shape_vert_dir = get_shape_dir(
            shape_vert_dir[:, :, :BODY_SHAPES_SPACE_DIM], BS)

        expr_shape_vert_dir = get_shape_dir(
            shape_vert_dir[:, :, BODY_SHAPES_SPACE_DIM:], ES)

        utils.check_shapes(body_shape_vert_dir, (V, 3, BS))
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

        mesh_graph = mesh_utils.MeshGraph.from_faces(
            V, faces, device)

        tex_mesh_graph = mesh_utils.MeshGraph.from_faces(
            TV, tex_faces, device)

        # ---

        def f(obj):
            return None if obj is None else obj.to(device, dtype)

        # ---

        return ModelData(
            kin_tree=kin_tree,

            mesh_graph=mesh_graph,
            tex_mesh_graph=tex_mesh_graph,

            body_joints_cnt=model_config.body_joints_cnt,
            jaw_joints_cnt=model_config.jaw_joints_cnt,
            eye_joints_cnt=model_config.eye_joints_cnt,

            vert_pos=f(vert_pos),

            tex_vert_pos=f(tex_vert_pos),

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
    def from_state_dict(
        state_dict: dict[str, object],
        *,
        dtype: typing.Optional[torch.dtype] = None,
        device: typing.Optional[torch.device] = None,
    ) -> ModelData:
        assert dtype is None or dtype.is_floating_point

        def try_fetch_int(field_name: str):
            return utils.tensor_deserialize(
                state_dict.get(field_name, None), torch.long, device)

        def try_fetch_float(field_name: str):
            return utils.tensor_deserialize(
                state_dict.get(field_name, None), dtype, device)

        vert_pos = try_fetch_float("vert_pos")

        tex_vert_pos = try_fetch_float("tex_vert_pos")

        V, TV = utils.check_shapes(
            vert_pos, (..., -1, 3),
            tex_vert_pos, (..., -2, 2),
        )

        return ModelData(
            kin_tree=kin_utils.KinTree.from_parents(
                state_dict["joint_parents"]),

            mesh_graph=mesh_utils.MeshGraph.from_faces(
                V, try_fetch_int("faces"), device),
            tex_mesh_graph=mesh_utils.MeshGraph.from_faces(
                TV, try_fetch_int("tex_faces"), device),

            body_joints_cnt=state_dict["body_joints_cnt"],
            jaw_joints_cnt=state_dict["jaw_joints_cnt"],
            eye_joints_cnt=state_dict["eye_joints_cnt"],

            vert_pos=vert_pos,

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

    def state_dict(self) -> collections.OrderedDict[str, object]:
        def to_int_np(x: torch.Tensor):
            return utils.tensor_serialize(
                x, dtype=torch.int32, device=self.device)

        def to_float_np(x: torch.Tensor):
            return utils.tensor_serialize(
                x, dtype=torch.float64, device=self.device)

        return collections.OrderedDict([
            ("joint_parents", self.kin_tree.parents),

            ("body_joints_cnt", self.body_joints_cnt),
            ("jaw_joints_cnt", self.jaw_joints_cnt),
            ("eye_joints_cnt", self.eye_joints_cnt),

            ("vert_pos", to_float_np(self.vert_pos)),
            ("tex_vert_pos", to_float_np(self.tex_vert_pos)),

            ("body_shape_vert_dir", to_float_np(self.body_shape_vert_dir)),
            ("expr_shape_vert_dir", to_float_np(self.expr_shape_vert_dir)),

            ("body_shape_joint_dir", to_float_np(self.body_shape_joint_dir)),
            ("expr_shape_joint_dir", to_float_np(self.expr_shape_joint_dir)),

            ("joint_t_mean", to_float_np(self.joint_t_mean)),
            ("pose_vert_dir", to_float_np(self.pose_vert_dir)),

            ("lbs_weight", to_float_np(self.lbs_weight)),

            ("lhand_pose_mean", to_float_np(self.lhand_pose_mean)),
            ("rhand_pose_mean", to_float_np(self.rhand_pose_mean)),

            ("faces", to_int_np(self.mesh_graph.f_to_vvv)),
            ("tex_faces", to_int_np(self.tex_mesh_graph.f_to_vvv)),
        ])

    def load_state_dict(self, state_dict: dict) -> None:
        model_data = ModelData.from_state_dict(state_dict)

        self.kin_tree = model_data.kin_tree

        self.mesh_graph = model_data.mesh_graph.to(self.device)
        self.tex_mesh_graph = model_data.tex_mesh_graph.to(self.device)

        self.body_joints_cnt = model_data.body_joints_cnt
        self.jaw_joints_cnt = model_data.jaw_joints_cnt
        self.eye_joints_cnt = model_data.eye_joints_cnt

        self.vert_pos = model_data.vert_pos.to(
            self.device, self.vert_pos.dtype)
        self.tex_vert_pos = model_data.tex_vert_pos.to(
            self.device, self.tex_vert_pos.dtype)

        self.body_shape_vert_dir = model_data.body_shape_vert_dir.to(
            self.device, self.body_shape_vert_dir.dtype)
        self.expr_shape_vert_dir = model_data.expr_shape_vert_dir.to(
            self.device, self.expr_shape_vert_dir.dtype)

        self.body_shape_joint_dir = model_data.body_shape_joint_dir.to(
            self.device, self.body_shape_joint_dir.dtype)
        self.expr_shape_joint_dir = model_data.expr_shape_joint_dir.to(
            self.device, self.expr_shape_joint_dir.dtype)

        self.joint_t_mean = model_data.joint_t_mean.to(
            self.device, self.joint_t_mean.dtype)

        self.pose_vert_dir = model_data.pose_vert_dir.to(
            self.device, self.pose_vert_dir.dtype)

        self.lbs_weight = model_data.lbs_weight.to(
            self.device, self.lbs_weight.dtype)

        self.lhand_pose_mean = model_data.lhand_pose_mean.to(
            self.device, self.lhand_pose_mean.dtype)
        self.rhand_pose_mean = model_data.rhand_pose_mean.to(
            self.device, self.rhand_pose_mean.dtype)

    @property
    def device(self):
        return self.vert_pos.device

    def to(self, *args, **kwargs) -> ModelData:
        d = {
            "kin_tree": self.kin_tree,

            "mesh_graph": None,
            "tex_mesh_graph": None,

            "vert_pos": None,

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
        mesh_subdivision_result = self.mesh_graph.subdivide(
            target_faces=target_faces)

        tex_mesh_subdivision_result = self.mesh_graph.subdivide(
            target_faces=target_faces)

        new_mesh_graph = mesh_subdivision_result.mesh_graph
        new_tex_mesh_graph = tex_mesh_subdivision_result.mesh_graph

        vert_src_table = mesh_subdivision_result.vert_src_table
        # [V_, 2]

        tex_vert_src_table = tex_mesh_subdivision_result.vert_src_table
        # [TV_, 2]

        new_vert_pos = (
            self.vert_pos[..., vert_src_table[:, 0], :] +
            self.vert_pos[..., vert_src_table[:, 1], :]) / 2
        # [..., V_, 3]

        new_tex_vert_pos = (
            self.tex_vert_pos[..., tex_vert_src_table[:, 0], :] +
            self.tex_vert_pos[..., tex_vert_src_table[:, 1], :]) / 2
        # [..., TV_, 2]

        if self.pose_vert_dir is None:
            new_pose_vert_dir = None
        else:
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
            mesh_graph=new_mesh_graph,
            tex_mesh_graph=new_tex_mesh_graph,

            body_joints_cnt=self.body_joints_cnt,
            jaw_joints_cnt=self.jaw_joints_cnt,
            eye_joints_cnt=self.eye_joints_cnt,

            vert_pos=new_vert_pos,

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

    def subdivide(
        self,
        *,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> mesh_utils.MeshSubdivisionResult:
        mesh_subdivision_result = self.mesh_graph.subdivide(
            target_edges=target_edges,
            target_faces=target_faces,
        )
        # vert_src_table[V_, 2]

        mesh_edge_mark = mesh_subdivision_result.edge_mark
        # [E]

        new_vert_src_table = mesh_subdivision_result.vert_src_table
        # [V_, 2]

        new_face_src_table = mesh_subdivision_result.face_src_table

        new_mesh_graph = mesh_subdivision_result.mesh_graph

        tex_mesh_edge_mark = [False] * self.tex_mesh_graph.edges_cnt

        for fe, tex_fe in zip(
                self.mesh_graph.f_to_eee, self.tex_mesh_graph.f_to_eee):
            for e, tex_e in zip(fe, tex_fe):
                if mesh_edge_mark[e]:
                    tex_mesh_edge_mark[tex_e] = True

        tex_mesh_e_to_new_v = [-1] * self.tex_mesh_graph.edges_cnt
        tex_mesh_vert_src_table = torch.empty(
            (self.verts_cnt + tex_mesh_edge_mark.count(True), 2),
            dtype=torch.long)

        for i in range(self.verts_cnt):
            tex_mesh_vert_src_table[i] = i

        for new_v, e in enumerate(
            (e for e, mark in enumerate(tex_mesh_edge_mark) if mark),
            self.tex_mesh_graph.verts_cnt,
        ):
            tex_mesh_e_to_new_v[e] = new_v
            tex_mesh_vert_src_table[new_v] = self.tex_mesh_graph.e_to_vv[e]

        new_tex_mesh_faces = list()

        for f in range(new_mesh_graph.faces_cnt):
            src_f, src_enum = new_face_src_table[f]

            va, vb, vc = map(int, self.tex_mesh_graph.f_to_vvv[src_f])

            ua = tex_mesh_e_to_new_v[utils.min_max(vb, vc)]
            ub = tex_mesh_e_to_new_v[utils.min_max(va, vc)]
            uc = tex_mesh_e_to_new_v[utils.min_max(va, vb)]

            match src_enum:
                case mesh_utils.MeshSubdivisionFaceSrcEnum.VA_VB_VC:
                    new_tex_mesh_faces.append((va, vb, vc))

                case mesh_utils.MeshSubdivisionFaceSrcEnum.VA_EC_EB:
                    new_tex_mesh_faces.append((va, uc, ub))
                case mesh_utils.MeshSubdivisionFaceSrcEnum.VB_EA_EC:
                    new_tex_mesh_faces.append((vb, ua, uc))
                case mesh_utils.MeshSubdivisionFaceSrcEnum.VC_EB_EA:
                    new_tex_mesh_faces.append((vc, ub, ua))
                case mesh_utils.MeshSubdivisionFaceSrcEnum.EA_EB_EC:
                    new_tex_mesh_faces.append((ua, ub, uc))

                case mesh_utils.MeshSubdivisionFaceSrcEnum.VA_VB_EA:
                    new_tex_mesh_faces.append((va, vb, ua))
                case mesh_utils.MeshSubdivisionFaceSrcEnum.VC_VA_EA:
                    new_tex_mesh_faces.append((vc, va, ua))

                case mesh_utils.MeshSubdivisionFaceSrcEnum.VB_VC_EB:
                    new_tex_mesh_faces.append((vb, vc, ub))
                case mesh_utils.MeshSubdivisionFaceSrcEnum.VA_VB_EB:
                    new_tex_mesh_faces.append((va, vb, ub))

                case mesh_utils.MeshSubdivisionFaceSrcEnum.VC_VA_EC:
                    new_tex_mesh_faces.append((vc, va, uc))
                case mesh_utils.MeshSubdivisionFaceSrcEnum.VB_VC_EC:
                    new_tex_mesh_faces.append((vb, vc, uc))

        new_tex_mesh_graph = mesh_utils.MeshGraph.from_faces(
            tex_mesh_vert_src_table.shape[0], new_tex_mesh_faces)

        new_vert_pos = (
            self.vert_pos[new_vert_src_table[:, 0]] +
            self.vert_pos[new_vert_src_table[:, 1]]
        ) / 2

        new_tex_vert_pos = (
            self.tex_vert_pos[tex_mesh_vert_src_table[:, 0]] +
            self.tex_vert_pos[tex_mesh_vert_src_table[:, 1]]
        ) / 2

        new_pose_vert_dir = (
            self.pose_vert_dir[..., new_vert_src_table[:, 0], :, :] +
            self.pose_vert_dir[..., new_vert_src_table[:, 1], :, :]
        ) / 2

        new_lbs_weight = (
            self.lbs_weight[..., new_vert_src_table[:, 0], :] +
            self.lbs_weight[..., new_vert_src_table[:, 1], :]
        ) / 2

        return ModelData(
            kin_tree=self.kin_tree,

            mesh_graph=new_mesh_graph,
            tex_mesh_graph=new_tex_mesh_graph,

            body_joints_cnt=self.body_joints_cnt,
            jaw_joints_cnt=self.jaw_joints_cnt,
            eye_joints_cnt=self.eye_joints_cnt,

            vert_pos=new_vert_pos,
            tex_vert_pos=new_tex_vert_pos,

            body_shape_vert_dir=self.body_shape_vert_dir,
            expr_shape_vert_dir=self.expr_shape_vert_dir,

            joint_t_mean=self.joint_t_mean,

            pose_vert_dir=new_pose_vert_dir,

            lbs_weight=new_lbs_weight,

            lhand_pose_mean=self.lhand_pose_mean,
            rhand_pose_mean=self.rhand_pose_mean,
        )

    def extract(
        self,
        *,
        target_faces: typing.Sequence[int],
    ) -> ModelDataExtractionResult:
        mesh_graph_extraction_result = self.mesh_graph.extract(
            target_faces)
        # vert_src_table[V_]
        # face_src_table[F_]

        tex_mesh_graph_extraction_result = self.tex_mesh_graph.extract(
            target_faces)
        # vert_src_table[TV_]
        # face_src_table[F_]

        new_mesh_graph = mesh_graph_extraction_result.mesh_graph
        new_tex_mesh_graph = tex_mesh_graph_extraction_result.mesh_graph

        vert_src_table = mesh_graph_extraction_result.vert_src_table
        # [V_]

        tex_vert_src_table = tex_mesh_graph_extraction_result.vert_src_table
        # [TV_]

        new_vert_pos = self.vert_pos[..., vert_src_table, :]
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

        if self.pose_vert_dir is None:
            new_pose_vert_dir = None
        else:
            new_pose_vert_dir = self.pose_vert_dir[..., vert_src_table, :, :]
        # [..., V_, 3, (J - 1) * 3 * 3]

        new_lbs_weight = self.lbs_weight[..., vert_src_table, :]
        # [..., V_, J]

        model_data = ModelData(
            kin_tree=self.kin_tree,

            mesh_graph=new_mesh_graph,
            tex_mesh_graph=new_tex_mesh_graph,

            body_joints_cnt=self.body_joints_cnt,
            jaw_joints_cnt=self.jaw_joints_cnt,
            eye_joints_cnt=self.eye_joints_cnt,

            vert_pos=new_vert_pos,

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
            mesh_graph_extraction_result=mesh_graph_extraction_result,
            tex_mesh_graph_extraction_result=tex_mesh_graph_extraction_result,
            model_data=model_data,
        )

    def show(self):
        self.mesh_graph.show(self.vert_pos)
