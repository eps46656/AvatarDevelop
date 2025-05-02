from __future__ import annotations

import collections
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
class ModelDataSubdivisionResult:
    mesh_subdivision_result: mesh_utils.MeshSubdivisionResult
    model_data: ModelData


@beartype
@dataclasses.dataclass
class ModelDataExtractionResult:
    mesh_graph_extraction_result: typing.Optional[
        mesh_utils.MeshExtractionResult]

    tex_mesh_graph_extraction_result: typing.Optional[
        mesh_utils.MeshExtractionResult]

    model_data: ModelData


@beartype
class ModelData:
    def __init__(
        self,
        *,
        body_joints_cnt: int = 0,  # BJ
        jaw_joints_cnt: int = 0,  # JJ
        eye_joints_cnt: int = 0,  # EYEJ

        kin_tree: typing.Optional[kin_utils.KinTree] = None,

        mesh_graph: typing.Optional[mesh_utils.MeshGraph],
        tex_mesh_graph: typing.Optional[mesh_utils.MeshGraph],

        joint_t_mean: typing.Optional[torch.Tensor],  # [..., J, 3]

        vert_pos: typing.Optional[torch.Tensor],  # [..., V, 3]
        tex_vert_pos: typing.Optional[torch.Tensor],  # [..., TV, 2]

        lbs_weight: typing.Optional[torch.Tensor],  # [..., V, J]

        body_shape_joint_dir: typing.Optional[torch.Tensor],  # [..., J, 3, BS]
        expr_shape_joint_dir: typing.Optional[torch.Tensor],  # [..., J, 3, ES]

        body_shape_vert_dir: typing.Optional[torch.Tensor],  # [..., V, 3, BS]
        expr_shape_vert_dir: typing.Optional[torch.Tensor],  # [..., V, 3, ES]

        lhand_pose_mean: typing.Optional[torch.Tensor],  # [..., HANDJ, 3]
        rhand_pose_mean: typing.Optional[torch.Tensor],  # [..., HANDJ, 3]

        pose_vert_dir: typing.Optional[torch.Tensor],
        # [..., V, 3, (J - 1) * 3 * 3]
    ):
        BODYJ = body_joints_cnt
        JAWJ = jaw_joints_cnt
        EYEJ = eye_joints_cnt

        assert 0 <= BODYJ
        assert 0 <= JAWJ
        assert 0 <= EYEJ

        HANDJ, BS, ES, J, V, TV, P = -1, -2, -3, -4, -5, -6, -7

        HANDJ, BS, ES, J, V, TV, P = utils.check_shapes(
            vert_pos, (..., V, 3),
            tex_vert_pos, (..., TV, 2),

            lbs_weight, (..., V, J),

            body_shape_joint_dir, (..., J, 3, BS),
            expr_shape_joint_dir, (..., J, 3, ES),

            body_shape_vert_dir, (..., V, 3, BS),
            expr_shape_vert_dir, (..., V, 3, ES),

            lhand_pose_mean, (..., HANDJ, 3),
            rhand_pose_mean, (..., HANDJ, 3),

            pose_vert_dir, (..., V, 3, P),

            set_zero_if_undet=False,
        )

        if kin_tree is not None:
            assert J < 0 or kin_tree.joints_cnt == J
            J = kin_tree.joints_cnt

        F = -8

        if mesh_graph is not None:
            assert V < 0 or mesh_graph.verts_cnt == V
            V = mesh_graph.verts_cnt

            assert F < 0 or mesh_graph.faces_cnt == F
            F = mesh_graph.faces_cnt

        if tex_mesh_graph is not None:
            assert TV < 0 or tex_mesh_graph.verts_cnt == TV
            TV = tex_mesh_graph.verts_cnt

            assert F < 0 or tex_mesh_graph.faces_cnt == F
            F = tex_mesh_graph.faces_cnt

        BODYJ = max(0, BODYJ)
        JAWJ = max(0, JAWJ)
        EYEJ = max(0, EYEJ)
        HANDJ = max(0, HANDJ)
        J = max(0, J)

        V = max(0, V)
        TV = max(0, TV)

        F = max(0, F)

        assert max(0, P) == max(0, J - 1) * 3 * 3

        assert BODYJ + JAWJ + EYEJ * 2 + HANDJ * 2 == J

        # ---

        self.joints_cnt = J
        self.body_joints_cnt = BODYJ
        self.jaw_joints_cnt = JAWJ
        self.eye_joints_cnt = EYEJ
        self.hand_joints_cnt = HANDJ

        self.body_shapes_cnt = BS
        self.expr_shapes_cnt = ES

        self.verts_cnt = V
        self.tex_verts_cnt = TV

        self.faces_cnt = F

        self.kin_tree = kin_tree

        self.mesh_graph = mesh_graph
        self.tex_mesh_graph = tex_mesh_graph

        self.joint_t_mean = joint_t_mean  # [..., J, 3]

        self.vert_pos = vert_pos  # [..., V, 3]
        self.tex_vert_pos = tex_vert_pos  # [..., TV, 2]

        self.lbs_weight = lbs_weight  # [..., V, J]

        self.body_shape_joint_dir = body_shape_joint_dir  # [..., J, 3, BS]
        self.expr_shape_joint_dir = expr_shape_joint_dir  # [..., J, 3, ES]

        self.body_shape_vert_dir = body_shape_vert_dir  # [..., V, 3, BS]
        self.expr_shape_vert_dir = expr_shape_vert_dir  # [..., V, 3, ES]

        self.lhand_pose_mean = lhand_pose_mean  # [..., HANDJ, 3]
        self.rhand_pose_mean = rhand_pose_mean  # [..., HANDJ, 3]

        self.pose_vert_dir = pose_vert_dir  # [..., V, 3, (J - 1) * 3 * 3]

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

        kin_tree_table = model_data.get("kintree_table", None)

        if kin_tree_table is None:
            kin_tree = None
            J = 0
        else:
            kin_tree_links = [
                (int(kin_tree_table[0, j]), int(kin_tree_table[1, j]))
                for j in range(kin_tree_table.shape[1])]

            kin_tree = kin_utils.KinTree.from_links(kin_tree_links)
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

            return None if val is None \
                else torch.from_numpy(model_data[field_name]).to(torch.long)

        def try_fetch_float(field_name: str):
            val = model_data.get(field_name, None)

            return None if val is None \
                else torch.from_numpy(model_data[field_name]).to(torch.float64)

        faces = try_fetch_int("f")
        tex_faces = try_fetch_int("ft")

        vert_pos = try_fetch_float("v_template")
        tex_vert_pos = try_fetch_float("vt")

        lbs_weight = try_fetch_float("weights")

        joint_regressor = try_fetch_float("J_regressor")

        shape_vert_dir = try_fetch_float("shapedirs")

        pose_vert_dir = try_fetch_float("posedirs")

        V, TV, F = -1, -2, -3

        V, TV, F = utils.check_shapes(
            faces, (F, 3),
            tex_faces, (F, 3),

            vert_pos, (V, 3),
            tex_vert_pos, (TV, 2),

            lbs_weight, (V, J),

            joint_regressor, (J, V),
            pose_vert_dir, (V, 3, (J - 1) * 3 * 3),
        )

        # ---

        def get_hand_pose_mean(hand_pose_mean: torch.Tensor) -> torch.Tensor:
            if HANDJ == 0 or hand_pose_mean is None:
                return torch.empty((HANDJ, 3), dtype=torch.float64)

            hand_pose_mean = hand_pose_mean.reshape(-1, 3)

            if HANDJ <= hand_pose_mean.shape[0]:
                return hand_pose_mean[:HANDJ, :]

            return torch.nn.functional.pad(
                hand_pose_mean,
                (0, 0, 0, HANDJ - hand_pose_mean.shape[0]),
                "constant",
                0,
            )

        def get_shape_dir(shape_dir: torch.Tensor, shape_dirs_cnt: int) \
                -> torch.Tensor:
            assert 0 <= shape_dirs_cnt

            if shape_dirs_cnt == 0 or shape_dir is None:
                return torch.empty((V, 3, 0), dtype=torch.float64)

            K = utils.check_shapes(shape_dir, (V, 3, -1))

            if shape_dirs_cnt <= K:
                return shape_dir[:, :, :shape_dirs_cnt]

            return torch.nn.functional.pad(
                shape_dir,
                (0, shape_dirs_cnt - K),
                "constant",
                0,
            )

        body_shape_vert_dir = get_shape_dir(
            shape_vert_dir[:, :, :BODY_SHAPES_SPACE_DIM], BS)

        expr_shape_vert_dir = get_shape_dir(
            shape_vert_dir[:, :, BODY_SHAPES_SPACE_DIM:], ES)

        lhand_pose_mean = get_hand_pose_mean(try_fetch_float("hands_meanl"))
        rhand_pose_mean = get_hand_pose_mean(try_fetch_float("hands_meanr"))

        utils.check_shapes(
            body_shape_vert_dir, (V, 3, BS),
            expr_shape_vert_dir, (V, 3, ES),

            rhand_pose_mean, (HANDJ, 3),
            lhand_pose_mean, (HANDJ, 3),
        )

        # ---

        mesh_graph = mesh_utils.MeshGraph.from_faces(V, faces, device)
        tex_mesh_graph = mesh_utils.MeshGraph.from_faces(TV, tex_faces, device)

        joint_t_mean = utils.einsum(
            "...jv, ...vx -> ...jx",
            joint_regressor,
            vert_pos,
        )

        body_shape_joint_dir = utils.einsum(
            "...jv, ...vxb -> ...jxb",
            joint_regressor,
            body_shape_vert_dir,
        )

        expr_shape_joint_dir = utils.einsum(
            "...jv, ...vxb -> ...jxb",
            joint_regressor,
            expr_shape_vert_dir,
        )

        # ---

        def f(obj):
            return obj.to(device, dtype)

        # ---

        return ModelData(
            body_joints_cnt=model_config.body_joints_cnt,
            jaw_joints_cnt=model_config.jaw_joints_cnt,
            eye_joints_cnt=model_config.eye_joints_cnt,

            kin_tree=kin_tree,

            mesh_graph=mesh_graph,
            tex_mesh_graph=tex_mesh_graph,

            joint_t_mean=f(joint_t_mean),

            vert_pos=f(vert_pos),
            tex_vert_pos=f(tex_vert_pos),

            lbs_weight=f(lbs_weight),

            body_shape_joint_dir=f(body_shape_joint_dir),
            expr_shape_joint_dir=f(expr_shape_joint_dir),

            body_shape_vert_dir=f(body_shape_vert_dir),
            expr_shape_vert_dir=f(expr_shape_vert_dir),

            lhand_pose_mean=f(lhand_pose_mean),
            rhand_pose_mean=f(rhand_pose_mean),

            pose_vert_dir=f(pose_vert_dir),
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

        faces = try_fetch_int("faces")
        tex_faces = try_fetch_int("tex_faces")

        return ModelData(
            body_joints_cnt=state_dict["body_joints_cnt"],
            jaw_joints_cnt=state_dict["jaw_joints_cnt"],
            eye_joints_cnt=state_dict["eye_joints_cnt"],

            kin_tree=kin_utils.KinTree.from_parents(
                state_dict["joint_parents"]),

            mesh_graph=None if faces is None else
            mesh_utils.MeshGraph.from_faces(V, faces, device),

            tex_mesh_graph=None if tex_faces is None else
            mesh_utils.MeshGraph.from_faces(TV, tex_faces, device),

            joint_t_mean=try_fetch_float("joint_t_mean"),

            vert_pos=vert_pos,
            tex_vert_pos=tex_vert_pos,

            lbs_weight=try_fetch_float("lbs_weight"),

            body_shape_vert_dir=try_fetch_float("body_shape_vert_dir"),
            expr_shape_vert_dir=try_fetch_float("expr_shape_vert_dir"),

            body_shape_joint_dir=try_fetch_float("body_shape_joint_dir"),
            expr_shape_joint_dir=try_fetch_float("expr_shape_joint_dir"),

            lhand_pose_mean=try_fetch_float("lhand_pose_mean"),
            rhand_pose_mean=try_fetch_float("rhand_pose_mean"),

            pose_vert_dir=try_fetch_float("pose_vert_dir"),
        )

    def state_dict(self) -> collections.OrderedDict[str, object]:
        def to_int_np(x: torch.Tensor):
            return utils.tensor_serialize(x, dtype=np.int64)

        def to_float_np(x: torch.Tensor):
            return utils.tensor_serialize(x, dtype=np.float64)

        return collections.OrderedDict([
            ("joint_parents", self.kin_tree.parents),

            ("body_joints_cnt", self.body_joints_cnt),
            ("jaw_joints_cnt", self.jaw_joints_cnt),
            ("eye_joints_cnt", self.eye_joints_cnt),

            ("faces", None if self.mesh_graph is None else to_int_np(
                self.mesh_graph.f_to_vvv)),
            ("tex_faces", None if self.tex_mesh_graph is None else to_int_np(
                self.tex_mesh_graph.f_to_vvv)),

            ("joint_t_mean", to_float_np(self.joint_t_mean)),

            ("vert_pos", to_float_np(self.vert_pos)),
            ("tex_vert_pos", to_float_np(self.tex_vert_pos)),

            ("lbs_weight", to_float_np(self.lbs_weight)),

            ("body_shape_joint_dir", to_float_np(self.body_shape_joint_dir)),
            ("expr_shape_joint_dir", to_float_np(self.expr_shape_joint_dir)),

            ("body_shape_vert_dir", to_float_np(self.body_shape_vert_dir)),
            ("expr_shape_vert_dir", to_float_np(self.expr_shape_vert_dir)),

            ("lhand_pose_mean", to_float_np(self.lhand_pose_mean)),
            ("rhand_pose_mean", to_float_np(self.rhand_pose_mean)),

            ("pose_vert_dir", to_float_np(self.pose_vert_dir)),
        ])

    def load_state_dict(self, state_dict: dict) -> None:
        def f(x, *args):
            return None if x is None else x.to(*args)

        model_data = ModelData.from_state_dict(state_dict, device=self.device)

        self.kin_tree = model_data.kin_tree

        self.body_joints_cnt = model_data.body_joints_cnt
        self.jaw_joints_cnt = model_data.jaw_joints_cnt
        self.eye_joints_cnt = model_data.eye_joints_cnt

        self.mesh_graph = f(model_data.mesh_graph, self.device)
        self.tex_mesh_graph = f(model_data.tex_mesh_graph, self.device)

        self.joint_t_mean = f(model_data.joint_t_mean, self.joint_t_mean)

        self.vert_pos = f(model_data.vert_pos, self.vert_pos)
        self.tex_vert_pos = f(model_data.tex_vert_pos, self.tex_vert_pos)

        self.lbs_weight = f(model_data.lbs_weight, self.lbs_weight)

        self.body_shape_joint_dir = f(
            model_data.body_shape_joint_dir, self.body_shape_joint_dir)
        self.expr_shape_joint_dir = f(
            model_data.expr_shape_joint_dir, self.expr_shape_joint_dir)

        self.body_shape_vert_dir = f(
            model_data.body_shape_vert_dir, self.body_shape_vert_dir)
        self.expr_shape_vert_dir = f(
            model_data.expr_shape_vert_dir, self.expr_shape_vert_dir)

        self.lhand_pose_mean = f(
            model_data.lhand_pose_mean, self.lhand_pose_mean)
        self.rhand_pose_mean = f(
            model_data.rhand_pose_mean, self.rhand_pose_mean)

        self.pose_vert_dir = f(
            model_data.pose_vert_dir, self.pose_vert_dir)

    @property
    def device(self):
        return self.vert_pos.device

    def to(self, *args, **kwargs) -> ModelData:
        def f(x):
            return None if x is None else x.to(*args, **kwargs)

        return ModelData(
            body_joints_cnt=self.body_joints_cnt,
            jaw_joints_cnt=self.jaw_joints_cnt,
            eye_joints_cnt=self.eye_joints_cnt,

            kin_tree=self.kin_tree,

            mesh_graph=f(self.mesh_graph),
            tex_mesh_graph=f(self.tex_mesh_graph),

            joint_t_mean=f(self.joint_t_mean),

            vert_pos=f(self.vert_pos),
            tex_vert_pos=f(self.tex_vert_pos),

            lbs_weight=f(self.lbs_weight),

            body_shape_joint_dir=f(self.body_shape_joint_dir),
            expr_shape_joint_dir=f(self.expr_shape_joint_dir),

            body_shape_vert_dir=f(self.body_shape_vert_dir),
            expr_shape_vert_dir=f(self.expr_shape_vert_dir),

            lhand_pose_mean=f(self.lhand_pose_mean),
            rhand_pose_mean=f(self.rhand_pose_mean),

            pose_vert_dir=f(self.pose_vert_dir),
        )

    def subdivide(
        self, *,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
        mesh_subdivision_result:
            typing.Optional[mesh_utils.MeshSubdivisionResult] = None,
    ) -> ModelDataSubdivisionResult:
        if mesh_subdivision_result is None:
            assert self.mesh_graph is not None

            mesh_subdivision_result = self.mesh_graph.subdivide(
                target_edges=target_edges,
                target_faces=target_faces,
            )
            # vert_src_table[V_, 2]
        else:
            assert target_edges is None
            assert target_faces is None

        edge_mark = mesh_subdivision_result.edge_mark
        # [E]

        vert_src_table = mesh_subdivision_result.vert_src_table
        # [V_, 2]

        face_src_table = mesh_subdivision_result.face_src_table

        face_src_type_table = mesh_subdivision_result.face_src_type_table

        new_mesh_graph = mesh_subdivision_result.mesh_graph

        if self.tex_mesh_graph is None:
            new_tex_mesh_graph = None
            tex_mesh_vert_src_table = None
        else:
            tex_mesh_edge_mark = [False] * self.tex_mesh_graph.edges_cnt

            for fe, tex_fe in zip(
                    self.mesh_graph.f_to_eee, self.tex_mesh_graph.f_to_eee):
                for e, tex_e in zip(fe, tex_fe):
                    if edge_mark[e]:
                        tex_mesh_edge_mark[tex_e] = True

            tex_mesh_e_to_new_v = [-1] * self.tex_mesh_graph.edges_cnt

            tex_mesh_vert_src_table: list[tuple[int, int]] = [
                (i, i) for i in range(self.tex_verts_cnt)]

            for new_v, e in enumerate(
                (e for e, mark in enumerate(tex_mesh_edge_mark) if mark),
                self.tex_mesh_graph.verts_cnt,
            ):
                tex_mesh_e_to_new_v[e] = new_v

                tex_mesh_vert_src_table.append(
                    tuple(int(v) for v in self.tex_mesh_graph.e_to_vv[e]))

            new_tex_mesh_faces = list()

            for f in range(new_mesh_graph.faces_cnt):
                src_f = face_src_table[f]
                src_type = face_src_type_table[f]

                va, vb, vc = map(int, self.tex_mesh_graph.f_to_vvv[src_f])
                ea, eb, ec = map(int, self.tex_mesh_graph.f_to_eee[src_f])

                ua = tex_mesh_e_to_new_v[ea]
                ub = tex_mesh_e_to_new_v[eb]
                uc = tex_mesh_e_to_new_v[ec]

                match src_type:
                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.VA_VB_VC:
                        new_tex_mesh_faces.append((va, vb, vc))

                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.VA_EC_EB:
                        new_tex_mesh_faces.append((va, uc, ub))
                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.VB_EA_EC:
                        new_tex_mesh_faces.append((vb, ua, uc))
                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.VC_EB_EA:
                        new_tex_mesh_faces.append((vc, ub, ua))
                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.EA_EB_EC:
                        new_tex_mesh_faces.append((ua, ub, uc))

                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.VA_VB_EA:
                        new_tex_mesh_faces.append((va, vb, ua))
                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.VC_VA_EA:
                        new_tex_mesh_faces.append((vc, va, ua))

                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.VB_VC_EB:
                        new_tex_mesh_faces.append((vb, vc, ub))
                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.VA_VB_EB:
                        new_tex_mesh_faces.append((va, vb, ub))

                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.VC_VA_EC:
                        new_tex_mesh_faces.append((vc, va, uc))
                    case mesh_utils.MeshSubdivisionFaceSrcTypeEnum.VB_VC_EC:
                        new_tex_mesh_faces.append((vb, vc, uc))

            tex_mesh_vert_src_table = torch.tensor(
                tex_mesh_vert_src_table, dtype=torch.long, device=self.device)
            # [TV_, 2]

            new_tex_mesh_faces = torch.tensor(
                new_tex_mesh_faces, dtype=torch.long, device=self.device)
            # [F_, 3]

            new_tex_mesh_graph = mesh_utils.MeshGraph.from_faces(
                tex_mesh_vert_src_table.shape[0],
                new_tex_mesh_faces, self.device)

        new_vert_pos = None if self.vert_pos is None \
            else self.vert_pos[..., vert_src_table, :].mean(-2)

        new_tex_vert_pos = self.tex_vert_pos \
            if self.tex_vert_pos is None or tex_mesh_vert_src_table is None \
            else self.tex_vert_pos[..., tex_mesh_vert_src_table, :].mean(-2)

        new_lbs_weight = None if self.lbs_weight is None \
            else self.lbs_weight[..., vert_src_table, :].mean(-2)

        new_body_shape_vert_dir = None if self.body_shape_vert_dir is None \
            else self.body_shape_vert_dir[..., vert_src_table, :, :].mean(-3)

        new_expr_shape_vert_dir = None if self.expr_shape_vert_dir is None \
            else self.expr_shape_vert_dir[..., vert_src_table, :, :].mean(-3)

        new_pose_vert_dir = None if self.pose_vert_dir is None \
            else self.pose_vert_dir[..., vert_src_table, :, :].mean(-3)

        model_data = ModelData(
            body_joints_cnt=self.body_joints_cnt,
            jaw_joints_cnt=self.jaw_joints_cnt,
            eye_joints_cnt=self.eye_joints_cnt,

            kin_tree=self.kin_tree,

            mesh_graph=new_mesh_graph,
            tex_mesh_graph=new_tex_mesh_graph,

            joint_t_mean=self.joint_t_mean,

            vert_pos=new_vert_pos,
            tex_vert_pos=new_tex_vert_pos,

            lbs_weight=new_lbs_weight,

            body_shape_joint_dir=self.body_shape_joint_dir,
            expr_shape_joint_dir=self.expr_shape_joint_dir,

            body_shape_vert_dir=new_body_shape_vert_dir,
            expr_shape_vert_dir=new_expr_shape_vert_dir,

            lhand_pose_mean=self.lhand_pose_mean,
            rhand_pose_mean=self.rhand_pose_mean,

            pose_vert_dir=new_pose_vert_dir,
        )

        return ModelDataSubdivisionResult(
            mesh_subdivision_result=mesh_subdivision_result,
            model_data=model_data,
        )

    def extract(
        self,
        *,
        target_faces: typing.Sequence[int],
    ) -> ModelDataExtractionResult:
        if self.mesh_graph is None:
            mesh_graph_extraction_result = None
            new_mesh_graph = None
            vert_src_table = None
        else:
            mesh_graph_extraction_result = self.mesh_graph.extract(
                target_faces)
            # vert_src_table[V_]
            # face_src_table[F_]

            new_mesh_graph = mesh_graph_extraction_result.mesh_graph

            vert_src_table = mesh_graph_extraction_result.vert_src_table
            # [V_]

        if self.tex_mesh_graph is None:
            tex_mesh_graph_extraction_result = None
            new_tex_mesh_graph = None
            tex_vert_src_table = None
        else:
            tex_mesh_graph_extraction_result = self.tex_mesh_graph.extract(
                target_faces)

            new_tex_mesh_graph = tex_mesh_graph_extraction_result.mesh_graph

            tex_vert_src_table = tex_mesh_graph_extraction_result.vert_src_table
            # [TV_]

        new_vert_pos = self.vert_pos \
            if self.vert_pos is None or vert_src_table is None \
            else self.vert_pos[..., vert_src_table, :]
        # [..., V_, 3]

        new_tex_vert_pos = self.tex_vert_pos \
            if self.tex_vert_pos is None or tex_vert_src_table is None \
            else self.tex_vert_pos[..., tex_vert_src_table, :]
        # [..., TV_, 2]

        new_body_shape_vert_dir = self.body_shape_vert_dir \
            if self.body_shape_vert_dir is None or vert_src_table is None \
            else self.body_shape_vert_dir[..., vert_src_table, :, :]
        # [..., V_, 3, ES]

        new_expr_shape_vert_dir = self.expr_shape_vert_dir \
            if self.expr_shape_vert_dir is None or vert_src_table is None \
            else self.expr_shape_vert_dir[..., vert_src_table, :, :]
        # [..., V_, 3, BS]

        new_pose_vert_dir = self.pose_vert_dir \
            if self.pose_vert_dir is None or vert_src_table is None \
            else self.pose_vert_dir[..., vert_src_table, :, :]
        # [..., V_, 3, (J - 1) * 3 * 3]

        new_lbs_weight = self.lbs_weight \
            if self.lbs_weight is None or vert_src_table is None \
            else self.lbs_weight[..., vert_src_table, :]
        # [..., V_, J]

        model_data = ModelData(
            body_joints_cnt=self.body_joints_cnt,
            jaw_joints_cnt=self.jaw_joints_cnt,
            eye_joints_cnt=self.eye_joints_cnt,

            kin_tree=self.kin_tree,

            mesh_graph=new_mesh_graph,
            tex_mesh_graph=new_tex_mesh_graph,

            joint_t_mean=self.joint_t_mean,

            vert_pos=new_vert_pos,
            tex_vert_pos=new_tex_vert_pos,

            lbs_weight=new_lbs_weight,

            body_shape_joint_dir=self.body_shape_joint_dir,
            expr_shape_joint_dir=self.expr_shape_joint_dir,

            body_shape_vert_dir=new_body_shape_vert_dir,
            expr_shape_vert_dir=new_expr_shape_vert_dir,

            lhand_pose_mean=self.lhand_pose_mean,
            rhand_pose_mean=self.rhand_pose_mean,

            pose_vert_dir=new_pose_vert_dir,
        )

        return ModelDataExtractionResult(
            mesh_graph_extraction_result=mesh_graph_extraction_result,
            tex_mesh_graph_extraction_result=tex_mesh_graph_extraction_result,
            model_data=model_data,
        )

    def remesh(self, remsh_arg_pack: utils.ArgPack) -> ModelData:
        mesh_data = mesh_utils.MeshData(self.mesh_graph, self.vert_pos)

        remeshed_mesh_data = mesh_data.remesh(
            *remsh_arg_pack.args,
            **remsh_arg_pack.kwargs,
        )

        new_mesh_graph = remeshed_mesh_data.mesh_graph
        new_vert_pos = remeshed_mesh_data.vert_pos

        return ModelData(
            body_joints_cnt=self.body_joints_cnt,
            jaw_joints_cnt=self.jaw_joints_cnt,
            eye_joints_cnt=self.eye_joints_cnt,

            kin_tree=self.kin_tree,

            mesh_graph=new_mesh_graph,
            tex_mesh_graph=None,

            joint_t_mean=self.joint_t_mean,

            vert_pos=new_vert_pos,
            tex_vert_pos=None,

            lbs_weight=None,

            body_shape_joint_dir=self.body_shape_joint_dir,
            expr_shape_joint_dir=self.expr_shape_joint_dir,

            body_shape_vert_dir=None,
            expr_shape_vert_dir=None,

            lhand_pose_mean=self.lhand_pose_mean,
            rhand_pose_mean=self.rhand_pose_mean,

            pose_vert_dir=None,
        )

    def show(self) -> None:
        self.mesh_graph.show(self.vert_pos)
