from __future__ import annotations

import collections
import dataclasses
import enum
import functools
import itertools
import typing

import torch
import tqdm
import trimesh
from beartype import beartype

from . import utils


@beartype
def get_area_vec(
    vert_pos_a: torch.Tensor,  # [..., 3]
    vert_pos_b: torch.Tensor,  # [..., 3]
    vert_pos_c: torch.Tensor,  # [..., 3]
) -> torch.Tensor:  # [..., 3]
    utils.check_shapes(
        vert_pos_a, (..., 3),
        vert_pos_b, (..., 3),
        vert_pos_c, (..., 3),
    )

    batch_shape = utils.broadcast_shapes(
        vert_pos_a, vert_pos_b, vert_pos_c)

    vert_pos_a = vert_pos_a.expand(batch_shape)
    vert_pos_b = vert_pos_b.expand(batch_shape)
    vert_pos_c = vert_pos_c.expand(batch_shape)

    return utils.vec_cross(
        vert_pos_b - vert_pos_a,
        vert_pos_c - vert_pos_a)


@beartype
def get_area_weighted_vert_nor(
    *,
    faces: torch.Tensor,  # [F, 3]
    vert_pos: torch.Tensor,  # [..., V, 3]
    vert_pos_a: typing.Optional[torch.Tensor] = None,  # [..., F, 3]
    vert_pos_b: typing.Optional[torch.Tensor] = None,  # [..., F, 3]
    vert_pos_c: typing.Optional[torch.Tensor] = None,  # [..., F, 3]
):
    F, V = -1, -2

    F, V = utils.check_shapes(
        faces, (F, 3),
        vert_pos, (..., V, 3),
    )

    if vert_pos_a is None:
        vert_pos_a = vert_pos[..., faces[:, 0], :]

    if vert_pos_b is None:
        vert_pos_b = vert_pos[..., faces[:, 1], :]

    if vert_pos_c is None:
        vert_pos_c = vert_pos[..., faces[:, 2], :]

    utils.check_shapes(
        vert_pos_a, (..., F, 3),
        vert_pos_b, (..., F, 3),
        vert_pos_c, (..., F, 3),
    )

    area_vector = get_area_vec(
        vert_pos_a,
        vert_pos_b,
        vert_pos_c,
    )
    # [..., F, 3]

    vert_nor = utils.zeros_like(vert_pos)
    # [..., V, 3]

    vert_nor.index_add_(-2, faces[:, 0], area_vector)
    vert_nor.index_add_(-2, faces[:, 1], area_vector)
    vert_nor.index_add_(-2, faces[:, 2], area_vector)

    # vert_nor[..., faces[:, 0][i], :] += area_vector[..., i, :]
    # vert_nor[..., faces[:, 1][i], :] += area_vector[..., i, :]
    # vert_nor[..., faces[:, 2][i], :] += area_vector[..., i, :]

    return utils.vec_normed(vert_nor)


@beartype
def calc_adj_sums(
    adj_rel_list: torch.Tensor,  # [P, 2]
    vals: torch.Tensor,  # [..., V, D]
):
    P, V, D = -1, -2, -3

    P, V, D = utils.check_shapes(
        adj_rel_list, (P, 2),
        vals, (..., V, D)
    )

    idx_0 = adj_rel_list[:, 0]
    idx_1 = adj_rel_list[:, 1]
    # [P]

    ret = utils.zeros_like(vals)
    # [..., V, D]

    ret.index_add_(-2, idx_0, vals[..., idx_1, :])
    ret.index_add_(-2, idx_1, vals[..., idx_0, :])

    # ret[..., idx_0[i], :] += vals_1[..., i, :]
    # ret[..., idx_1[i], :] += vals_0[..., i, :]

    return ret


@beartype
def calc_adj_sums_naive(
    adj_rel_list: torch.Tensor,  # [P, 2]
    vals: torch.Tensor,  # [..., V, D]
):
    P, V, D = -1, -2, -3

    P, V, D = utils.check_shapes(
        adj_rel_list, (P, 2),
        vals, (..., V, D)
    )

    ret = utils.zeros_like(vals)
    # [..., V, D]

    for pi in range(P):
        va, vb = adj_rel_list[pi]

        ret[..., va, :] += vals[..., vb, :]
        ret[..., vb, :] += vals[..., va, :]

    return ret


class MeshSubdivisionFaceSrcTypeEnum(enum.IntEnum):
    # no subdive
    VA_VB_VC = 0

    # subdive to 2, case a, cut edge bc
    VA_VB_EA = 20
    VC_VA_EA = 21

    # subdive to 2, case b, cut edge ca
    VB_VC_EB = 22
    VA_VB_EB = 23

    # subdive to 2, case c, cut edge ab
    VC_VA_EC = 24
    VB_VC_EC = 25

    # subdive to 4
    VA_EC_EB = 40
    VB_EA_EC = 41
    VC_EB_EA = 42
    EA_EB_EC = 43


@beartype
@dataclasses.dataclass
class MeshSubdivisionResult:
    edge_mark: list[bool]

    vert_src_table: torch.Tensor
    # [V_, 2]

    face_src_table: torch.Tensor
    # [F_]

    face_src_type_table: torch.Tensor
    # [F_]

    mesh_graph: MeshGraph


@beartype
@dataclasses.dataclass
class MeshExtractionResult:
    vert_src_table: torch.Tensor
    face_src_table: torch.Tensor
    mesh_graph: MeshGraph


@beartype
class MeshGraph:
    # V: vertices cnt
    # F: faces cnt
    # E: edges cnt
    # FP: adj face pairs cnt

    def __init__(
        self,
        *,

        e_to_vv: torch.Tensor,  # [E, 2]
        vv_to_e: dict[tuple[int, int], int],  # [E]

        f_to_vvv: torch.Tensor,  # [F, 3]
        f_to_eee: torch.Tensor,  # [F, 3]

        ff: torch.Tensor,  # [FP, 2]

        vert_deg: torch.Tensor,  # [V]
        inv_vert_deg: torch.Tensor,  # [V]
    ):
        V, F, VP, FP = -1, -2, -3, -4

        V, F, VP, FP = utils.check_shapes(
            e_to_vv, (VP, 2),
            f_to_vvv, (F, 3),
            ff, (FP, 2),

            vert_deg, (V,),
            inv_vert_deg, (V,),
        )

        self.e_to_vv = e_to_vv
        self.vv_to_e = vv_to_e

        self.f_to_vvv = f_to_vvv
        self.f_to_eee = f_to_eee

        self.ff = ff

        self.vert_deg = vert_deg
        self.inv_vert_deg = inv_vert_deg

    @staticmethod
    def empty(device: torch.device) -> MeshGraph:
        return MeshGraph(
            e_to_vv=torch.empty((0, 2), dtype=torch.long, device=device),
            vv_to_e=dict(),

            f_to_vvv=torch.empty((0, 3), dtype=torch.long, device=device),
            f_to_eee=torch.empty((0, 3), dtype=torch.long, device=device),

            ff=torch.empty((0, 2), dtype=torch.long, device=device),

            vert_deg=torch.empty((0,), dtype=torch.int32, device=device),
            inv_vert_deg=torch.empty((0,), dtype=torch.float64, device=device),
        )

    @staticmethod
    def from_faces(
        verts_cnt: int,
        faces: torch.Tensor,  # [F, 3]
        device: torch.device,
    ) -> MeshGraph:
        faces_cnt = utils.check_shapes(faces, (-1, 3))

        origin_faces = faces

        faces = faces.to(utils.CPU_DEVICE, torch.long)

        ee_to_f_d: collections.defaultdict[tuple[int, int], set[int]] = \
            collections.defaultdict(set)

        for f in range(faces_cnt):
            va, vb, vc = sorted(map(int, faces[f]))

            assert 0 <= va, f"{va=}"
            assert va < vb
            assert vb < vc
            assert vc < verts_cnt, f"{vc=}, {verts_cnt=}"

            ee_to_f_d[(vb, vc)].add(f)
            ee_to_f_d[(va, vc)].add(f)
            ee_to_f_d[(va, vb)].add(f)

        e_to_vv_l = sorted(ee_to_f_d.keys())

        vv_to_e = {(va, vb): e for e, (va, vb) in enumerate(e_to_vv_l)}

        f_to_eee_l: list[tuple[int, int, int]] = list()

        for f in range(faces_cnt):
            va, vb, vc = map(int, faces[f])

            f_to_eee_l.append((
                vv_to_e[utils.min_max(vb, vc)],
                vv_to_e[utils.min_max(va, vc)],
                vv_to_e[utils.min_max(va, vb)],
            ))

        ff: set[tuple[int, int]] = set()

        vert_deg = torch.zeros((verts_cnt,), dtype=torch.int32)

        for (va, vb), fs in ee_to_f_d.items():
            assert 0 <= va
            assert va < vb
            assert vb < verts_cnt

            vert_deg[va] += 1
            vert_deg[vb] += 1

            for fa, fb in itertools.combinations(fs, 2):
                ff.add(utils.min_max(fa, fb))

        faces = origin_faces.to(device, torch.long)

        e_to_vv = torch.tensor(
            sorted(ee_to_f_d.keys()),
            dtype=torch.long,
            device=device,
        )

        if e_to_vv.numel() == 0:
            e_to_vv = e_to_vv.expand(0, 2)

        f_to_eee = torch.tensor(
            f_to_eee_l,
            dtype=torch.long,
            device=device,
        )

        if f_to_eee.numel() == 0:
            f_to_eee = f_to_eee.expand(0, 3)

        ff: torch.Tensor = torch.tensor(
            sorted(ff),
            dtype=torch.long,
            device=device,
        )

        if ff.numel() == 0:
            ff = ff.expand(0, 2)

        inv_vert_deg = torch.where(
            vert_deg == 0,
            0,
            1.0 / vert_deg,
        ).to(device, torch.float64)

        return MeshGraph(
            e_to_vv=e_to_vv,
            vv_to_e=vv_to_e,

            f_to_vvv=faces,
            f_to_eee=f_to_eee,

            ff=ff,

            vert_deg=vert_deg.to(device),
            inv_vert_deg=inv_vert_deg,
        )

    @staticmethod
    def from_state_dict(
        state_dict: typing.Mapping[str, object],
        device: torch.device,
    ) -> MeshGraph:
        return MeshGraph.from_faces(
            state_dict["verts_cnt"],
            utils.tensor_deserialize(state_dict["faces"]),
            device,
        )

    @property
    def device(self) -> torch.device:
        return self.f_to_vvv.device

    @property
    def verts_cnt(self) -> int:
        return self.vert_deg.shape[0]

    @property
    def faces_cnt(self) -> int:
        return self.f_to_vvv.shape[0]

    @property
    def edges_cnt(self) -> int:
        return self.e_to_vv.shape[0]

    @property
    def adj_face_vert_pairs_cnt(self) -> int:
        return self.faces_cnt * 3

    @property
    def adj_face_face_pairs_cnt(self) -> int:
        return self.ff.shape[0]

    def to(self, *args, **kwargs) -> MeshGraph:
        def f(x):
            return None if x is None else x.to(*args, **kwargs)

        return MeshGraph(
            e_to_vv=f(self.e_to_vv),
            vv_to_e=self.vv_to_e,

            f_to_vvv=f(self.f_to_vvv),
            f_to_eee=f(self.f_to_eee),

            ff=f(self.ff),

            vert_deg=f(self.vert_deg),
            inv_vert_deg=f(self.inv_vert_deg),
        )

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("verts_cnt", self.verts_cnt),
            ("faces", utils.tensor_serialize(self.f_to_vvv)),
        ])

    def load_state_dict(
            self, state_dict: typing.Mapping[str, object]) -> MeshGraph:
        mesh_graph = MeshGraph.from_state_dict(state_dict, self.device)

        self.e_to_vv = mesh_graph.e_to_vv
        self.vv_to_e = mesh_graph.vv_to_e

        self.f_to_vvv = mesh_graph.f_to_vvv
        self.f_to_eee = mesh_graph.f_to_eee

        self.ff = mesh_graph.ff

        self.vert_deg = mesh_graph.vert_deg
        self.inv_vert_deg = mesh_graph.inv_vert_deg

        return self

    def calc_face_cos_sim(
        self,
        face_vec: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(face_vec, (..., self.faces_cnt, -1))

        vecs_0 = face_vec[..., self.ff[:, 0], :]
        vecs_1 = face_vec[..., self.ff[:, 1], :]
        # [..., FP, D]

        return utils.vec_dot(vecs_0, vecs_1)

    def calc_face_cos_sim_naive(
        self,
        face_vec: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(face_vec, (..., self.faces_cnt, -1))

        FP = self.adj_face_face_pairs_cnt

        ret = utils.empty_like(face_vec, shape=face_vec.shape[:-2] + (FP,))

        for fp in range(FP):
            fa, fb = self.ff[fp, :]
            ret[..., fp] = utils.vec_dot(
                face_vec[..., fa, :], (face_vec[..., fb, :]))

        return ret

    def calc_face_diff(
        self,
        face_vec: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP, D]
        utils.check_shapes(face_vec, (..., self.faces_cnt, -1))

        vecs_0 = face_vec[..., self.ff[:, 0], :]
        vecs_1 = face_vec[..., self.ff[:, 1], :]
        # [..., FP, D]

        return vecs_0 - vecs_1

    def calc_face_diff_naive(
        self,
        face_vec: torch.Tensor,  # [..., F]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(face_vec, (..., self.faces_cnt, -1))

        FP = self.adj_face_face_pairs_cnt

        ret = utils.empty_like(face_vec, shape=face_vec.shape[:-2] + (FP,))

        for fp in range(FP):
            fa, fb = self.ff[fp, :]

            ret[..., fp] = face_vec[..., fa, :] - (face_vec[..., fb, :])

        return ret

    def face_lap_trans(
        self,
        face_features: torch.Tensor,  # [F, D]
        ratio: float,
    ):
        utils.check_shapes(face_features, (self.faces_cnt, -1))

        assert 0 <= ratio <= 1

        trans_ratio = (1 - ratio) / 3

        fa = self.ff[:, 0]
        fb = self.ff[:, 1]

        buffer = utils.zeros_like(face_features)

        buffer.index_add_(-2, fa, face_features[fb], alpha=trans_ratio)
        buffer.index_add_(-2, fb, face_features[fa], alpha=trans_ratio)

        return face_features * ratio + buffer

    def subdivide(
        self,
        *,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> MeshSubdivisionResult:
        f_to_vvv = self.f_to_vvv.to(utils.CPU_DEVICE)
        # [F, 3]

        f_to_eee = self.f_to_eee.to(utils.CPU_DEVICE)
        # [F, 3]

        e_to_vv = self.e_to_vv.to(utils.CPU_DEVICE)
        # [VP, 2]

        e_to_fs: list[list[int]] = [[] for _ in range(self.edges_cnt)]

        for f in range(self.faces_cnt):
            for e in map(int, f_to_eee[f]):
                e_to_fs[e].append(f)

        se_cnts = [0] * self.faces_cnt

        edge_mark = [False] * self.edges_cnt

        edge_queue = list()

        def try_add_edge_to_queue(e):
            if not edge_mark[e]:
                edge_mark[e] = True
                edge_queue.append(e)

        if target_edges is None and target_faces is None:
            for e in range(self.edges_cnt):
                try_add_edge_to_queue(e)

        if target_edges is not None:
            for e in target_edges:
                try_add_edge_to_queue(e)

        if target_faces is not None:
            for f in target_faces:
                for e in map(int, f_to_eee[f]):
                    try_add_edge_to_queue(e)

        while 0 < len(edge_queue):
            e = edge_queue.pop()

            for f in e_to_fs[e]:
                se_cnts[f] += 1

                if se_cnts[f] == 2:
                    for e in map(int, f_to_eee[f]):
                        try_add_edge_to_queue(e)

        e_to_new_v: list[int] = [-1] * self.edges_cnt

        vert_src_table = [(i, i) for i in range(self.verts_cnt)]

        for new_v, e in enumerate(
            (e for e, mark in enumerate(edge_mark) if mark),
            self.verts_cnt,
        ):
            e_to_new_v[e] = new_v
            vert_src_table.append(tuple(int(v) for v in e_to_vv[e]))

        new_faces: list[tuple[int, int, int]] = list()

        face_src_table: list[int] = list()
        face_src_type_table: list[int] = list()

        def add_face(src, src_type, new_va, new_vb, new_vc):
            new_faces.append((new_va, new_vb, new_vc))

            face_src_table.append(src)
            face_src_type_table.append(src_type.value)

        for f, cnt in enumerate(se_cnts):
            va, vb, vc = map(int, f_to_vvv[f])
            ea, eb, ec = map(int, f_to_eee[f])

            assert cnt == 0 or cnt == 1 or cnt == 3

            if cnt == 0:
                add_face(f, MeshSubdivisionFaceSrcTypeEnum.VA_VB_VC, va, vb, vc)
                continue

            if cnt == 3:
                ua = e_to_new_v[ea]
                ub = e_to_new_v[eb]
                uc = e_to_new_v[ec]

                add_face(f, MeshSubdivisionFaceSrcTypeEnum.VA_EC_EB, va, uc, ub)
                add_face(f, MeshSubdivisionFaceSrcTypeEnum.VB_EA_EC, vb, ua, uc)
                add_face(f, MeshSubdivisionFaceSrcTypeEnum.VC_EB_EA, vc, ub, ua)
                add_face(f, MeshSubdivisionFaceSrcTypeEnum.EA_EB_EC, ua, ub, uc)

                continue

            ka = edge_mark[ea]
            kb = edge_mark[eb]
            kc = edge_mark[ec]

            match ka * 0b100 + kb * 0b010 + kc * 0b001:
                case 0b100:
                    ua = e_to_new_v[ea]

                    add_face(
                        f, MeshSubdivisionFaceSrcTypeEnum.VA_VB_EA, va, vb, ua)
                    add_face(
                        f, MeshSubdivisionFaceSrcTypeEnum.VC_VA_EA, vc, va, ua)

                case 0b010:
                    ub = e_to_new_v[eb]

                    add_face(
                        f, MeshSubdivisionFaceSrcTypeEnum.VB_VC_EB, vb, vc, ub)
                    add_face(
                        f, MeshSubdivisionFaceSrcTypeEnum.VA_VB_EB, va, vb, ub)

                case 0b001:
                    uc = e_to_new_v[ec]

                    add_face(
                        f, MeshSubdivisionFaceSrcTypeEnum.VC_VA_EC, vc, va, uc)
                    add_face(
                        f, MeshSubdivisionFaceSrcTypeEnum.VB_VC_EC, vb, vc, uc)

                case _:
                    raise utils.MismatchException()

        vert_src_table = torch.tensor(
            vert_src_table, dtype=torch.long, device=self.device)
        # [V_, 2]

        face_src_table = torch.tensor(
            face_src_table, dtype=torch.long, device=self.device)
        # [F_]

        face_src_type_table = torch.tensor(
            face_src_type_table, dtype=torch.uint8, device=self.device)
        # [F_]

        new_faces = torch.tensor(new_faces, dtype=torch.long)

        if new_faces.numel() == 0:
            vert_src_table = vert_src_table.expand(0, 2)
            new_faces = new_faces.expand(0, 3)

        mesh_graph = MeshGraph.from_faces(
            len(vert_src_table),
            new_faces,
            self.device,
        )

        return MeshSubdivisionResult(
            edge_mark=edge_mark,
            vert_src_table=vert_src_table,
            face_src_table=face_src_table,
            face_src_type_table=face_src_type_table,
            mesh_graph=mesh_graph,
        )

    def extract(
        self,
        target_faces: typing.Iterable[int],
    ) -> MeshExtractionResult:
        target_faces = sorted(set(target_faces))

        f_to_vvv = self.f_to_vvv.to(utils.CPU_DEVICE)
        # [F, 3]

        v_mark = [False] * self.verts_cnt

        for f in target_faces:
            for v in map(int, f_to_vvv[f]):
                v_mark[v] = True

        v_to_new_v: dict[int, int] = dict()

        vert_src_table = torch.empty((v_mark.count(True),), dtype=torch.long)

        for new_v, v in enumerate(v for v, mark in enumerate(v_mark) if mark):
            v_to_new_v[v] = new_v
            vert_src_table[new_v] = v

        new_f_to_vvv = torch.empty((len(target_faces), 3), dtype=torch.long)

        face_src_table = torch.empty((len(target_faces),), dtype=torch.long)

        for new_f, f in enumerate(target_faces):
            for k in range(3):
                new_f_to_vvv[new_f, k] = v_to_new_v[int(f_to_vvv[f, k])]

            face_src_table[new_f] = f

        mesh_graph = MeshGraph.from_faces(
            vert_src_table.shape[0],
            new_f_to_vvv,
            device=self.device,
        )

        return MeshExtractionResult(
            vert_src_table=vert_src_table,
            face_src_table=face_src_table,
            mesh_graph=mesh_graph,
        )

    def show(
        self,
        vert_pos: torch.Tensor,  # [V, 3]
    ):
        V = self.vert_deg.shape[0]

        utils.check_shapes(vert_pos, (V, 3))

        tm = trimesh.Trimesh(
            vertices=vert_pos.detach().to(utils.CPU_DEVICE),
            faces=self.f_to_vvv.to(utils.CPU_DEVICE),
            validate=True,
        )

        tm.show()


@beartype
class MeshData:
    def __init__(
        self,
        mesh_graph: MeshGraph,
        vert_pos: torch.Tensor,  # [..., V, D]
    ):
        D = utils.check_shapes(
            vert_pos, (..., mesh_graph.verts_cnt, -1),
        )

        self.mesh_graph = mesh_graph
        self.vert_pos = vert_pos.to(self.mesh_graph.device)

    @property
    def shape(self) -> torch.Size:
        return self.vert_pos.shape[:-2]

    @property
    def device(self) -> torch.device:
        return self.mesh_graph.device

    @property
    def verts_cnt(self) -> int:
        return self.mesh_graph.verts_cnt

    @property
    def edges_cnt(self) -> int:
        return self.mesh_graph.edges_cnt

    @property
    def faces_cnt(self) -> int:
        return self.mesh_graph.faces_cnt

    def to(self, *args, **kwargs) -> MeshData:
        mesh_graph = self.mesh_graph.to(*args, **kwargs)

        vert_pos = self.vert_pos.to(*args, **kwargs)

        mesh_data = MeshData(mesh_graph, vert_pos)

        for key, val in MeshData.__dict__.items():
            if isinstance(val, functools.cached_property):
                cached_val = getattr(self, key, None)

                if hasattr(cached_val, "to"):
                    setattr(self, key, cached_val.to(*args, **kwargs))

        return mesh_data

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("mesh_graph", self.mesh_graph.state_dict()),
            ("vert_pos", utils.tensor_serialize(self.vert_pos)),
        ])

    def load_state_dict(
        self,
        state_dict: typing.Mapping[str, object],
        **kwargs,
    ) -> None:
        self.mesh_graph.load_state_dict(
            state_dict["mesh_graph"],
            **kwargs,
        )

        self.vert_pos = utils.tensor_deserialize(
            state_dict["vert_pos"],
            self.vert_pos.dtype,
            self.vert_pos.device,
        )

        self.clear_cache()

    def clear_cache(self) -> None:
        for key, val in MeshData.__dict__.items():
            if isinstance(val, functools.cached_property):
                delattr(self, key)

    @functools.cached_property
    def edge_vert_pos(self) -> torch.Tensor:  # [..., E, 2, D]
        return self.vert_pos[..., self.mesh_graph.e_to_vv, :]

    @functools.cached_property
    def edge_dir(self) -> torch.Tensor:  # [..., E, D]
        evp = self.edge_vert_pos
        # [..., E, 2, D]

        return evp[..., 0, :] - evp[..., 1, :]

    @functools.cached_property
    def edge_norm(self) -> torch.Tensor:  # [..., E]
        return utils.vec_norm(self.edge_dir)

    @functools.cached_property
    def edge_sq_norm(self) -> torch.Tensor:  # [..., E]
        return utils.vec_sq_norm(self.edge_dir)

    @functools.cached_property
    def face_vert_pos(self) -> torch.Tensor:  # [..., F, 3, D]
        return self.vert_pos[..., self.mesh_graph.f_to_vvv, :]

    @functools.cached_property
    def face_edge_dir(self) -> torch.Tensor:  # [..., F, 3, D]
        fvp = self.face_vert_pos
        # [... F, 3, D]

        buffer = utils.empty_like(fvp)
        # [..., F, 3, D]

        buffer[..., 0, :] = fvp[..., 1, :] - fvp[..., 0, :]
        buffer[..., 1, :] = fvp[..., 2, :] - fvp[..., 1, :]
        buffer[..., 2, :] = fvp[..., 0, :] - fvp[..., 2, :]

        return buffer

    @functools.cached_property
    def face_edge_dir_ab(self) -> torch.Tensor:  # [..., F, 3, D]
        return self.face_edge_dir[..., 0, :]

    @functools.cached_property
    def face_edge_dir_bc(self) -> torch.Tensor:  # [..., F, 3, D]
        return self.face_edge_dir[..., 1, :]

    @functools.cached_property
    def face_edge_dir_ca(self) -> torch.Tensor:  # [..., F, 3, D]
        return self.face_edge_dir[..., 2, :]

    @functools.cached_property
    def face_mean(self) -> torch.Tensor:  # [..., F, D]
        return self.face_vert_pos.mean(-2)

    @functools.cached_property
    def face_vert_svd(self) -> tuple[
        torch.Tensor,  # u[..., F, 3, 3]
        torch.Tensor,  # s[..., F, min(3, D)]
        torch.Tensor,  # v[..., F, D, D]
    ]:
        return torch.linalg.svd(self.face_vert_pos)

    @functools.cached_property
    def face_vert_svd_u(self) -> torch.Tensor:  # [..., F, 3, 3]
        return self.face_vert_svd[0]

    @functools.cached_property
    def face_vert_svd_s(self) -> torch.Tensor:  # [..., F, min(3, D)]
        return self.face_vert_svd[1]

    @functools.cached_property
    def face_vert_svd_vh(self) -> torch.Tensor:  # [..., F, D, D]
        return self.face_vert_svd[2]

    @functools.cached_property
    def face_edge_norm(self) -> torch.Tensor:  # [..., F, 3]
        return utils.vec_norm(self.face_edge_dir)

    @functools.cached_property
    def face_edge_sum_norm(self) -> torch.Tensor:  # [..., F]
        return self.face_edge_norm.sum(-1)

    @functools.cached_property
    def face_edge_sq_norm(self) -> torch.Tensor:  # [..., F, 3]
        return utils.vec_sq_norm(self.face_edge_dir)

    @functools.cached_property
    def face_edge_sum_sq_norm(self) -> torch.Tensor:  # [..., F]
        return self.face_edge_sq_norm.sum(-1)

    @functools.cached_property
    def face_area_vec(self) -> torch.Tensor:  # [..., F]
        assert self.vert_pos.shape[-1] == 3

        return utils.vec_cross(self.face_edge_dir_ca, self.face_edge_dir_ab) / 2

    @functools.cached_property
    def face_norm(self) -> torch.Tensor:  # [..., F, 3]
        assert self.vert_pos.shape[-1] == 3

        return self.face_area_vec / self.face_area.unsqueeze(-1)

    @functools.cached_property
    def face_area(self) -> torch.Tensor:  # [..., F]
        return self.face_sq_area.sqrt()

    @functools.cached_property
    def face_sq_area(self) -> torch.Tensor:  # [... , F]
        fen = self.face_edge_norm
        # [..., F, 3]

        sorted_fen, _ = torch.sort(fen, -1, True)
        # [..., F, 3]

        ea = sorted_fen[..., 0]
        eb = sorted_fen[..., 1]
        ec = sorted_fen[..., 2]

        eab = ea - eb

        return (
            (ea + (eb + ec)) * (ec - eab) * (ec + eab) * (ea + (eb - ec))
        ) / 16

    @functools.cached_property
    def face_edge_diff(self) -> torch.Tensor:  # [..., F, 3]
        # sq_e_a = self.face_edge_sq_norm[..., 0, :]
        # sq_e_b = self.face_edge_sq_norm[..., 1, :]
        # sq_e_c = self.face_edge_sq_norm[..., 2, :]

        # (sq_e_b + sq_e_c - sq_e_a) * denom
        # (sq_e_c + sq_e_a - sq_e_b) * denom
        # (sq_e_a + sq_e_b - sq_e_c) * denom
        # =
        # (sq_e_a + sq_e_b + sq_e_c - sq_e_a * 2) * denom
        # (sq_e_a + sq_e_b + sq_e_c - sq_e_b * 2) * denom
        # (sq_e_a + sq_e_b + sq_e_c - sq_e_c * 2) * denom

        return self.face_edge_sq_norm.sum(-1, True) - self.face_edge_sq_norm * 2

    @functools.cached_property
    def face_cot_angle(self) -> torch.Tensor:  # [..., F, 3]
        return self.face_edge_diff * (0.25 / self.face_area).unsqueeze(-1)

    @functools.cached_property
    def uni_lap_diff(self) -> torch.Tensor:  # [..., V, D]
        return calc_adj_sums(
            self.mesh_graph.e_to_vv, self.vert_pos
        ) * self.mesh_graph.inv_vert_deg.unsqueeze(-1) - self.vert_pos

    @functools.cached_property
    def cot_lap_diff(self) -> torch.Tensor:  # [..., V, D]
        e_weight = utils.zeros_like(
            self.vert_pos, shape=self.shape + (self.edges_cnt,))
        # [..., E]

        for k in range(3):
            e_weight.index_add_(
                -1, self.mesh_graph.f_to_eee[:, k],
                self.face_cot_angle[..., k].detach())

        v_sum_weight = utils.zeros_like(
            self.vert_pos, shape=self.vert_pos.shape[:-1])
        # [..., V]

        v_sum_weight.index_add_(-1, self.mesh_graph.e_to_vv[:, 0], e_weight)
        v_sum_weight.index_add_(-1, self.mesh_graph.e_to_vv[:, 1], e_weight)

        weighted_e_diff = self.edge_dir.detach() * e_weight.unsqueeze(-1)

        buffer = utils.zeros_like(self.vert_pos)
        # [..., V, D]

        buffer.index_add_(
            -2, self.mesh_graph.e_to_vv[:, 0], weighted_e_diff, alpha=+1)
        buffer.index_add_(
            -2, self.mesh_graph.e_to_vv[:, 1], weighted_e_diff, alpha=-1)

        return buffer / v_sum_weight.unsqueeze(-1)

    @functools.cached_property
    def l1_uni_lap_smoothness(self) -> torch.Tensor:  # [...]
        return utils.vec_norm(self.uni_lap_diff).mean(-1)

    @functools.cached_property
    def l2_uni_lap_smoothness(self) -> torch.Tensor:  # [...]
        return utils.vec_sq_norm(self.uni_lap_diff).mean(-1)

    @functools.cached_property
    def l1_cot_lap_smoothness(self) -> torch.Tensor:  # [...]
        return utils.vec_norm(self.cot_lap_diff).mean(-1)

    @functools.cached_property
    def l2_cot_lap_smoothness(self) -> torch.Tensor:  # [...]
        return utils.vec_sq_norm(self.cot_lap_diff).mean(-1)

    def calc_uni_lap_smoothness_pytorch3d(self) -> torch.Tensor:  # []
        import pytorch3d
        import pytorch3d.loss
        import pytorch3d.structures

        utils.check_shapes(self.vert_pos, (..., self.verts_cnt, -1))

        assert 0 <= self.mesh_graph.f_to_vvv.min()
        assert self.mesh_graph.f_to_vvv.max() < self.verts_cnt

        mesh = pytorch3d.structures.Meshes(
            verts=[self.vert_pos],
            faces=[self.f_to_vvv],
            textures=None,
        ).to(self.vert_pos.device)

        utils.torch_cuda_sync()

        return pytorch3d.loss.mesh_laplacian_smoothing(mesh, method="uniform")

    def calc_cot_lap_smoothness_pytorch3d(self) -> torch.Tensor:  # []
        import pytorch3d
        import pytorch3d.loss
        import pytorch3d.structures

        utils.check_shapes(self.vert_pos, (..., self.verts_cnt, -1))

        assert 0 <= self.mesh_graph.f_to_vvv.min()
        assert self.mesh_graph.f_to_vvv.max() < self.verts_cnt

        mesh = pytorch3d.structures.Meshes(
            verts=[self.vert_pos],
            faces=[self.f_to_vvv],
            textures=None,
        ).to(self.vert_pos.device)

        utils.torch_cuda_sync()

        return pytorch3d.loss.mesh_laplacian_smoothing(mesh, method="cot")

    @functools.cached_property
    def face_norm_cos_sim(self) -> torch.Tensor:  # [..., FP]
        return self.mesh_graph.calc_face_cos_sim(self.face_norm)

    @functools.cached_property
    def face_edge_rel_var(self) -> torch.Tensor:
        return self.face_edge_sum_sq_norm / \
            self.face_edge_sum_norm.detach().square() * 3 - 1

    @functools.cached_property
    def face_bary_coord_mat(self) -> torch.Tensor:  # [..., F, 3, 3]
        D = self.vert_pos.shape[-1]

        fvp = self.face_vert_pos
        # [..., F, 3, D]

        m = utils.empty_like(self.vert_pos, shape=(
            self.shape + (self.faces_cnt, D + 1, 3)))
        # [..., F, D + 1, 3]

        m[..., :D, 0] = fvp[..., 0, :]
        m[..., :D, 1] = fvp[..., 1, :]
        m[..., :D, 2] = fvp[..., 2, :]
        m[..., D, :] = 1

        return torch.linalg.pinv(m)

    def calc_unsigned_dist(
        self,
        point_pos: torch.Tensor,  # [..., D]
    ) -> torch.Tensor:  # [...]
        D = self.vert_pos.shape[-1]

        pp = point_pos

        utils.check_shapes(pp, (..., D))

        face_bary_coord_mat = self.face_bary_coord_mat
        # [..., F, 3, D + 1]

        ks = utils.do_rt(
            face_bary_coord_mat[..., :D],  # [..., F, 3, D]
            face_bary_coord_mat[..., D],  # [..., F, 3]
            pp,  # [..., D]
        )
        # [..., F, 3]

        ka = ks[..., 0]
        kb = ks[..., 1]
        kc = ks[..., 2]
        # [..., F]

        p_to_v_dist = utils.vec_norm(pp.unsqueeze(-2) - self.vert_pos)
        # ([..., 3] -> [..., 1, 3]) - [..., V, 3]
        # [..., V, 3]
        # [..., V]

        p_to_f_dist = torch.where(
            (0 <= ka) & (0 <= kb) & (0 <= kc),
            kz.abs(),
            torch.minimum(
                p_to_v_dist[..., self.mesh_graph.f_to_vvv[:, 0]],
                torch.minimum(
                    p_to_v_dist[..., self.mesh_graph.f_to_vvv[:, 1]],
                    p_to_v_dist[..., self.mesh_graph.f_to_vvv[:, 2]],
                ))
        )

        min_p_to_f, min_p_to_fi = p_to_f_dist.min(-1, True)
        # [..., 1]
        # [..., 1]

        ret = torch.where(
            kz.gather(-1, min_p_to_fi) < 0,  # [..., 1]
            -min_p_to_f,  # [..., 1]
            min_p_to_f,  # [..., 1]
        ).squeeze(-1)

        return ret

    def calc_signed_dist(
        self,
        point_pos: torch.Tensor,  # [..., D]
    ) -> torch.Tensor:  # [...]
        D = self.vert_pos.shape[-1]

        pp = point_pos

        utils.check_shapes(pp, (..., D))

        face_bary_coord_mat = self.face_bary_coord_mat
        # [..., F, 3, D + 1]

        ks = utils.do_rt(
            face_bary_coord_mat[..., :D],  # [..., F, 3, D]
            face_bary_coord_mat[..., D],  # [..., F, 3]
            pp,  # [..., D]
        )
        # [..., F, 3]

        kb = ks[..., 0]
        kc = ks[..., 1]
        kz = ks[..., 2]
        # [..., F]

        p_to_v_dist = utils.vec_norm(pp.unsqueeze(-2) - self.vert_pos)
        # ([..., 3] -> [..., 1, 3]) - [..., V, 3]
        # [..., V, 3]
        # [..., V]

        p_to_f_dist = torch.where(
            (0 <= kb) & (0 <= kc) & (kb + kc <= 1),
            kz.abs(),
            torch.minimum(
                p_to_v_dist[..., self.mesh_graph.f_to_vvv[:, 0]],
                torch.minimum(
                    p_to_v_dist[..., self.mesh_graph.f_to_vvv[:, 1]],
                    p_to_v_dist[..., self.mesh_graph.f_to_vvv[:, 2]],
                ))
        )

        min_p_to_f, min_p_to_fi = p_to_f_dist.min(-1, True)
        # [..., 1]
        # [..., 1]

        ret = torch.where(
            kz.gather(-1, min_p_to_fi) < 0,  # [..., 1]
            -min_p_to_f,  # [..., 1]
            min_p_to_f,  # [..., 1]
        ).squeeze(-1)

        return ret

    def calc_signed_dist_naive(
        self,
        point_pos: torch.Tensor,  # [..., 3]
    ):
        def _calc_dist(va, vb, vc, p):
            utils.check_shapes(
                va, (3,),
                vb, (3,),
                vc, (3,),
                p, (3,),
            )

            ab = vb - va
            ac = vc - va
            z = utils.vec_cross(ab, ac)

            z_norm = utils.vec_norm(z)

            t = torch.tensor([
                [ab[0], ac[0], z[0], va[0]],
                [ab[1], ac[1], z[1], va[1]],
                [ab[2], ac[2], z[2], va[2]],
                [0, 0, 0, 1],
            ], dtype=utils.FLOAT).inverse()

            ks = (t @ torch.tensor(
                [[p[0]], [p[1]], [p[2]], [1]], dtype=utils.FLOAT)) \
                .unsqueeze(-1)

            kb = ks[0]
            kc = ks[1]
            kz = ks[2]

            da = utils.vec_norm(va - p)
            db = utils.vec_norm(vb - p)
            dc = utils.vec_norm(vc - p)

            ret = min(da, db, dc)

            if 0 <= kb and 0 <= kc and kb + kc <= 1:
                assert kz.abs() * z_norm <= ret + 1e-4, f"{kz.abs()=}, {ret=}"
                ret = min(ret, kz.abs() * z_norm)

            ret = float(ret)

            assert 0 <= ret

            return -ret if kz < 0 else ret

        vp = self.vert_pos.to(utils.CPU_DEVICE)
        pp = point_pos.to(utils.CPU_DEVICE)

        V = self.verts_cnt
        F = self.faces_cnt

        utils.check_shapes(
            vp, (V, 3),
            pp, (..., 3),
        )

        vp_a = vp[self.mesh_graph.f_to_vvv[:, 0], :]
        vp_b = vp[self.mesh_graph.f_to_vvv[:, 1], :]
        vp_c = vp[self.mesh_graph.f_to_vvv[:, 2], :]
        # [F, 3]

        ret = torch.empty(pp.shape[:-1], dtype=pp.dtype)

        for pi in tqdm.tqdm(utils.get_batch_idxes(pp.shape[:-1])):
            ans = float("inf")

            p = pp[pi]

            for vi in tqdm.tqdm(range(V)):
                cur_ans = _calc_dist(vp_a[vi, :], vp_b[vi, :], vp_c[vi, :], p)

                if abs(cur_ans) < abs(ans):
                    ans = cur_ans

            ret[pi] = ans

        return ret

    def calc_signed_dist_trimesh(
        self,
        point_pos: torch.Tensor,  # [..., 3]
    ):
        tm = trimesh.Trimesh(
            vertices=self.vert_pos.to(utils.CPU_DEVICE),
            faces=self.mesh_graph.f_to_vvv.to(utils.CPU_DEVICE),
            validate=True,
        )

        with utils.Timer():
            ret = torch.from_numpy(-trimesh.proximity.signed_distance(
                tm,
                point_pos,
            ))

        return ret

    def calc_subdivided_vert_pos(
        self,
        mesh_subdivision_result: MeshSubdivisionResult
    ) -> torch.Tensor:  # [V, 3]
        pass

    def show(
        self,
        vert_pos: torch.Tensor,  # [V, 3]
    ):
        tm = trimesh.Trimesh(
            vertices=vert_pos.detach().to(utils.CPU_DEVICE),
            faces=self.mesh_graph.f_to_vvv.to(utils.CPU_DEVICE),
            validate=True,
        )

        tm.show()
