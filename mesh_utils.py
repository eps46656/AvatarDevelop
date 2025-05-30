from __future__ import annotations

import collections
import dataclasses
import enum
import functools
import heapq
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

    vert_nor = utils.zeros(like=vert_pos)
    # [..., V, 3]

    vert_nor.index_add_(-2, faces[:, 0], area_vector)
    vert_nor.index_add_(-2, faces[:, 1], area_vector)
    vert_nor.index_add_(-2, faces[:, 2], area_vector)

    # vert_nor[..., faces[:, 0][i], :] += area_vector[..., i, :]
    # vert_nor[..., faces[:, 1][i], :] += area_vector[..., i, :]
    # vert_nor[..., faces[:, 2][i], :] += area_vector[..., i, :]

    return utils.vec_normed(vert_nor)


@beartype
def calc_adj_sum(
    adj_rel_list: torch.Tensor,  # [P, 2]
    val: torch.Tensor,  # [..., V, D]
) -> torch.Tensor:  # [..., V, D]
    P, V, D = -1, -2, -3

    P, V, D = utils.check_shapes(
        adj_rel_list, (P, 2),
        val, (..., V, D)
    )

    idx_0 = adj_rel_list[:, 0]
    idx_1 = adj_rel_list[:, 1]
    # [P]

    ret = utils.zeros(like=val)
    # [..., V, D]

    ret.index_add_(-2, idx_0, val[..., idx_1, :])
    ret.index_add_(-2, idx_1, val[..., idx_0, :])

    # ret[..., idx_0[i], :] += vals_1[..., i, :]
    # ret[..., idx_1[i], :] += vals_0[..., i, :]

    return ret


@beartype
def calc_adj_sum_naive(
    adj_rel_list: torch.Tensor,  # [P, 2]
    val: torch.Tensor,  # [..., V, D]
) -> torch.Tensor:  # [..., V, D]
    P, V, D = -1, -2, -3

    P, V, D = utils.check_shapes(
        adj_rel_list, (P, 2),
        val, (..., V, D)
    )

    ret = utils.zeros(like=val)
    # [..., V, D]

    for pi in range(P):
        va, vb = adj_rel_list[pi]

        ret[..., va, :] += val[..., vb, :]
        ret[..., vb, :] += val[..., va, :]

    return ret


@beartype
def make_bary_coord_mat(
    face_vert_pos: torch.Tensor,  # [..., 3, 3]
    face_nor: torch.Tensor,  # [..., 3]
) -> torch.Tensor:  # [..., 4, 4]
    utils.check_shapes(
        face_vert_pos, (..., 3, 3),
        face_nor, (..., 3),
    )

    shape = utils.broadcast_shapes(
        face_vert_pos.shape[:-2],
        face_nor.shape[:-1],
    )

    face_vert_pos = face_vert_pos.expand(*shape, 3, 3).detach()
    face_nor = face_nor.expand(*shape, 3).detach()

    dtype = utils.promote_dtypes(face_vert_pos, face_nor)
    device = utils.all_same(face_vert_pos.device, face_nor.device)

    A = torch.zeros((*shape, 20, 16), dtype=dtype, device=device)

    for i in range(4):
        p = 4 * i
        q = p + 3

        A[..., p:q, p:q] = face_vert_pos
        A[..., p:q, q] = 1
        A[..., q, p:q] = face_nor

        A[..., 16 + i, i:-4:4] = 1

    b = torch.tensor([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        0, 0, 0, 1,
    ], dtype=dtype, device=device)[:, None].expand(*shape, 20, 1)

    return torch.linalg.lstsq(A, b).solution.view(*shape, 4, 4)


class MeshGraphSubdivideFaceSrcTypeEnum(enum.IntEnum):
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
class MeshSubdivideResult:
    edge_mark: list[bool]
    # [E]

    vert_src_table: torch.Tensor
    # [V_, 2]

    face_src_table: torch.Tensor
    # [F_]

    face_src_type_table: torch.Tensor
    # [F_]

    mesh_graph: MeshGraph


@beartype
@dataclasses.dataclass
class MeshExtractResult:
    vert_src_table: torch.Tensor
    # [V_]

    face_src_table: torch.Tensor
    # [F_]

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

        verts_cnt: int,
        f_to_vvv: torch.Tensor,  # [F, 3]
    ):
        assert 0 <= verts_cnt

        F = utils.check_shapes(f_to_vvv, (-1, 3))

        self.verts_cnt = verts_cnt
        self.f_to_vvv = f_to_vvv

    @staticmethod
    def empty(verts_cnt: int, device: torch.device) -> MeshGraph:
        return MeshGraph(
            verts_cnt=verts_cnt,
            f_to_vvv=torch.empty((0, 3), dtype=torch.long, device=device),
        )

    @staticmethod
    def from_faces(
        verts_cnt: int,
        faces: torch.Tensor,  # [F, 3]
    ) -> MeshGraph:
        assert 0 <= verts_cnt

        F = utils.check_shapes(faces, (-1, 3))

        return MeshGraph(
            verts_cnt=verts_cnt,
            f_to_vvv=faces,
        )

    @staticmethod
    def from_state_dict(
        state_dict: typing.Mapping[str, object],
        device: torch.device,
    ) -> MeshGraph:
        return MeshGraph(
            verts_cnt=state_dict["verts_cnt"],

            f_to_vvv=utils.deserialize_tensor(state_dict["faces"])
            .to(device, torch.long),
        )

    @property
    def device(self) -> torch.device:
        return self.f_to_vvv.device

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
        return MeshGraph(
            verts_cnt=self.verts_cnt,
            f_to_vvv=self.f_to_vvv.to(*args, **kwargs),
        )

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("verts_cnt", self.verts_cnt),
            ("faces", utils.serialize_tensor(self.f_to_vvv)),
        ])

    def load_state_dict(
            self, state_dict: typing.Mapping[str, object]) -> MeshGraph:
        self.verts_cnt = state_dict["verts_cnt"]

        self.f_to_vvv = utils.deserialize_tensor(
            state_dict["f_to_vvv"],
            dtype=torch.long,
            device=self.f_to_vvv.device
        )

        return self

    @functools.cached_property
    def f_to_vvv_cpu(self) -> torch.Tensor:  # [F, 3]
        return self.f_to_vvv.to(utils.CPU_DEVICE)

    @functools.cached_property
    def _v_to_fs__vv_to_fs(self):
        v_to_fs_raw: list[set[int]] = [set() for _ in range(self.verts_cnt)]

        vv_to_fs_raw: collections.defaultdict[tuple[int, int], set[int]] = \
            collections.defaultdict(set)

        for f in range(self.faces_cnt):
            va, vb, vc = sorted(map(int, self.f_to_vvv[f]))

            assert 0 <= va, f"{va=}"
            assert va < vb
            assert vb < vc
            assert vc < self.verts_cnt, f"{vc=}, {self.verts_cnt=}"

            v_to_fs_raw[va].add(f)
            v_to_fs_raw[vb].add(f)
            v_to_fs_raw[vc].add(f)

            vv_to_fs_raw[(vb, vc)].add(f)
            vv_to_fs_raw[(va, vc)].add(f)
            vv_to_fs_raw[(va, vb)].add(f)

        return v_to_fs_raw, dict(sorted(vv_to_fs_raw.items()))

    @functools.cached_property
    def v_to_fs(self) -> list[set[int]]:
        return self._v_to_fs__vv_to_fs[0]

    @functools.cached_property
    def vv_to_fs(self) -> dict[tuple[int, int], set[int]]:
        return self._v_to_fs__vv_to_fs[1]

    @functools.cached_property
    def e_to_fs(self) -> list[set[int]]:
        return list(self.vv_to_fs.values())

    @functools.cached_property
    def e_to_vv(self) -> torch.Tensor:  # [E, 2]
        return self.e_to_vv_cpu.to(self.device)

    @functools.cached_property
    def e_to_vv_cpu(self) -> torch.Tensor:  # [E, 2]
        ret = torch.tensor(list(self.vv_to_fs.keys()), dtype=torch.long)

        if ret.numel() == 0:
            ret.resize_(0, 2)

        return ret

    @functools.cached_property
    def vv_to_e(self) -> dict[tuple[int, int], int]:
        return {vv: e for e, vv in enumerate(self.vv_to_fs.keys())}

    @functools.cached_property
    def f_to_eee(self) -> torch.Tensor:  # [F, 3]
        return self.f_to_eee_cpu.to(self.device)

    @functools.cached_property
    def f_to_eee_cpu(self) -> torch.Tensor:  # [F, 3]
        f_to_vvv = self.f_to_vvv_cpu
        vv_to_e = self.vv_to_e

        ret = torch.empty((self.faces_cnt, 3), dtype=torch.long)

        for f in range(self.faces_cnt):
            va, vb, vc = map(int, f_to_vvv[f])

            ret[f, 0] = vv_to_e[utils.min_max(vb, vc)]
            ret[f, 1] = vv_to_e[utils.min_max(vc, va)]
            ret[f, 2] = vv_to_e[utils.min_max(va, vb)]

        return ret

    @functools.cached_property
    def vert_deg(self) -> torch.Tensor:  # [V]
        e_to_vv = self.e_to_vv
        # [E, 2]

        return \
            e_to_vv[:, 0].bincount(minlength=self.verts_cnt) + \
            e_to_vv[:, 1].bincount(minlength=self.verts_cnt)

    @functools.cached_property
    def vert_deg_cpu(self) -> torch.Tensor:  # [V]
        return self.vert_deg.to(utils.CPU_DEVICE)

    @functools.cached_property
    def inv_vert_deg(self) -> torch.Tensor:  # [V]
        return torch.where(
            self.vert_deg == 0, 0, 1.0 / self.vert_deg,
        ).to(torch.float64)

    @functools.cached_property
    def inv_vert_deg_cpu(self) -> torch.Tensor:  # [V]
        return self.inv_vert_deg.to(utils.CPU_DEVICE)

    @functools.cached_property
    def ff(self) -> torch.Tensor:  # [FP, 2]
        return self.ff_cpu.to(self.device)

    @functools.cached_property
    def ff_cpu(self) -> torch.Tensor:  # [FP, 2]
        s: set[tuple[int, int]] = set()

        for f in self.vv_to_fs.values():
            s.update(utils.min_max(fa, fb)
                     for fa, fb in itertools.combinations(f, 2))

        ret = torch.tensor(sorted(s), dtype=torch.long)

        if ret.numel() == 0:
            ret.resize_(0, 2)

        return ret

    def calc_face_cos_sim(
        self,
        face_vec: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(face_vec, (..., self.faces_cnt, -1))

        return utils.vec_dot(
            face_vec[..., self.ff[:, 0], :], face_vec[..., self.ff[:, 1], :])

    def calc_face_diff(
        self,
        face_vec: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP, D]
        utils.check_shapes(face_vec, (..., self.faces_cnt, -1))

        return face_vec[..., self.ff[:, 0], :] - face_vec[..., self.ff[:, 1], :]

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

        buffer = utils.zeros(like=face_features)

        buffer.index_add_(-2, fa, face_features[fb], alpha=trans_ratio)
        buffer.index_add_(-2, fb, face_features[fa], alpha=trans_ratio)

        return face_features * ratio + buffer

    def subdivide(
        self,
        *,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> MeshSubdivideResult:
        f_to_vvv = self.f_to_vvv_cpu
        # [F, 3]

        f_to_eee = self.f_to_eee_cpu
        # [F, 3]

        e_to_vv = self.e_to_vv_cpu
        # [VP, 2]

        e_to_f = self.e_to_fs

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

            for f in e_to_f[e]:
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

        TypeEnum = MeshGraphSubdivideFaceSrcTypeEnum

        for f, cnt in enumerate(se_cnts):
            va, vb, vc = map(int, f_to_vvv[f])
            ea, eb, ec = map(int, f_to_eee[f])

            assert cnt == 0 or cnt == 1 or cnt == 3

            if cnt == 0:
                add_face(f, TypeEnum.VA_VB_VC, va, vb, vc)
                continue

            if cnt == 3:
                ua = e_to_new_v[ea]
                ub = e_to_new_v[eb]
                uc = e_to_new_v[ec]

                add_face(f, TypeEnum.VA_EC_EB, va, uc, ub)
                add_face(f, TypeEnum.VB_EA_EC, vb, ua, uc)
                add_face(f, TypeEnum.VC_EB_EA, vc, ub, ua)
                add_face(f, TypeEnum.EA_EB_EC, ua, ub, uc)

                continue

            ka = edge_mark[ea]
            kb = edge_mark[eb]
            kc = edge_mark[ec]

            match ka * 0b100 + kb * 0b010 + kc * 0b001:
                case 0b100:
                    ua = e_to_new_v[ea]

                    add_face(f, TypeEnum.VA_VB_EA, va, vb, ua)
                    add_face(f, TypeEnum.VC_VA_EA, vc, va, ua)

                case 0b010:
                    ub = e_to_new_v[eb]

                    add_face(f, TypeEnum.VB_VC_EB, vb, vc, ub)
                    add_face(f, TypeEnum.VA_VB_EB, va, vb, ub)

                case 0b001:
                    uc = e_to_new_v[ec]

                    add_face(f, TypeEnum.VC_VA_EC, vc, va, uc)
                    add_face(f, TypeEnum.VB_VC_EC, vb, vc, uc)

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

        new_faces = torch.tensor(
            new_faces, dtype=torch.long, device=self.device)

        if new_faces.numel() == 0:
            vert_src_table.resize_(0, 2)
            new_faces.resize_(0, 3)

        mesh_graph = MeshGraph.from_faces(
            len(vert_src_table),
            new_faces,
        )

        return MeshSubdivideResult(
            edge_mark=edge_mark,
            vert_src_table=vert_src_table,
            face_src_table=face_src_table,
            face_src_type_table=face_src_type_table,
            mesh_graph=mesh_graph,
        )

    def remove_orphan_vert(self) -> MeshGraph:
        return self.extract(range(self.faces_cnt), True)

    def extract(
        self,
        target_faces: typing.Iterable[int],
        remove_orphan_vert: bool,
        device: typing.Optional[torch.device] = None,
    ) -> MeshExtractResult:

        if device is None:
            device = self.device

        f_to_vvv = self.f_to_vvv_cpu
        # [F, 3]

        face_src_table = torch.tensor(
            sorted(set(target_faces)), dtype=torch.long)
        # [F_]

        new_f_to_old_vvv = f_to_vvv[face_src_table]
        # [F_, 3]

        if remove_orphan_vert:
            v_mark = torch.zeros(self.verts_cnt, dtype=torch.bool)
            v_mark.index_put_(new_f_to_old_vvv.view(-1), True)
        else:
            v_mark = torch.ones(self.verts_cnt, dtype=torch.bool)

        v_to_new_v = v_mark.cumsum(dtype=torch.long) - 1
        # [V]

        vert_src_table = v_mark.nonzero(as_tuple=True)[0]
        # [V_]

        new_f_to_vvv = v_to_new_v[new_f_to_old_vvv]
        # [F_, 3]

        mesh_graph = MeshGraph.from_faces(
            vert_src_table.shape[0],
            new_f_to_vvv.to(device),
        )

        return MeshExtractResult(
            vert_src_table=vert_src_table,
            face_src_table=face_src_table,
            mesh_graph=mesh_graph,
        )

    def get_cluster(
        self,
        vert_conn: bool,
        edge_conn: bool,
    ) -> list[list[int]]:
        ret: list[list[int]] = list()

        passed = [False] * self.faces_cnt

        q = list()

        for f in range(self.faces_cnt):
            if passed[f]:
                continue

            q.append(f)

            cur_cluster = list()

            while 0 < len(q):
                f = heapq.heappop(q)
                cur_cluster.append(f)

                if passed[f]:
                    continue

                if vert_conn:
                    v_to_fs = self.v_to_fs

                    for v in map(int, self.f_to_vvv[f]):
                        for adj_f in v_to_fs[v]:
                            if not passed[adj_f]:
                                heapq.heappush(q, adj_f)

                if edge_conn:
                    e_to_fs = self.e_to_fs

                    for e in map(int, self.f_to_eee[f]):
                        for adj_f in e_to_fs[e]:
                            if not passed[adj_f]:
                                heapq.heappush(q, adj_f)

            ret.append(cur_cluster)

        return ret

    def show(
        self,
        vert_pos: torch.Tensor,  # [V, 3]
    ) -> None:
        V = self.vert_deg.shape[0]

        utils.check_shapes(vert_pos, (V, 3))

        vert_pos = vert_pos.detach().to(utils.CPU_DEVICE)

        tm = trimesh.Trimesh(
            vertices=vert_pos,
            faces=self.f_to_vvv_cpu,
            validate=True,
        )

        pc = trimesh.points.PointCloud(vert_pos)

        scene = trimesh.Scene()

        scene.add_geometry(tm)
        scene.add_geometry(pc)

        scene.show()


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
            ("vert_pos", utils.serialize_tensor(self.vert_pos)),
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

        self.vert_pos = utils.deserialize_tensor(
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

        m = torch.tensor([
            [-1, 1, 0],
            [0, -1, 1],
            [1, 0, -1],
        ], dtype=fvp.dtype, device=fvp.device)

        return m @ fvp

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

        return self.face_area_vec / self.face_area[..., None]

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
        return self.face_edge_diff * (0.25 / self.face_area)[..., None]

    @functools.cached_property
    def uni_lap_diff(self) -> torch.Tensor:  # [..., V, D]
        return calc_adj_sum(
            self.mesh_graph.e_to_vv, self.vert_pos
        ) * self.mesh_graph.inv_vert_deg[..., None] - self.vert_pos

    @functools.cached_property
    def cot_lap_diff(self) -> torch.Tensor:  # [..., V, D]
        e_weight = utils.zeros(
            like=self.vert_pos, shape=(*self.shape, self.edges_cnt))
        # [..., E]

        for k in range(3):
            e_weight.index_add_(
                -1, self.mesh_graph.f_to_eee[:, k],
                self.face_cot_angle[..., k].detach())

        e_to_vv = self.mesh_graph.e_to_vv

        v_sum_weight = utils.zeros(
            like=self.vert_pos, shape=self.vert_pos.shape[:-1])
        # [..., V]

        v_sum_weight.index_add_(-1, e_to_vv[:, 0], e_weight)
        v_sum_weight.index_add_(-1, e_to_vv[:, 1], e_weight)

        weighted_e_diff = self.edge_dir * e_weight[..., None]

        buffer = utils.zeros(like=self.vert_pos)
        # [..., V, D]

        buffer.index_add_(-2, e_to_vv[:, 0], weighted_e_diff, alpha=+1)
        buffer.index_add_(-2, e_to_vv[:, 1], weighted_e_diff, alpha=-1)

        return buffer / v_sum_weight[..., None]

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
            faces=[self.mesh_graph.f_to_vvv],
            textures=None,
        ).to(self.vert_pos.device)

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
            faces=[self.mesh_graph.f_to_vvv],
            textures=None,
        ).to(self.vert_pos.device)

        return pytorch3d.loss.mesh_laplacian_smoothing(mesh, method="cot")

    @functools.cached_property
    def face_norm_cos_sim(self) -> torch.Tensor:  # [..., FP]
        return self.mesh_graph.calc_face_cos_sim(self.face_norm)

    @functools.cached_property
    def face_edge_var(self) -> torch.Tensor:
        mean_sq_x = self.face_edge_sum_sq_norm / 3
        # [..., F]

        sq_mean_x = (self.face_edge_sum_norm / 3).square()
        # [..., F]

        return mean_sq_x - sq_mean_x

    @functools.cached_property
    def face_edge_rel_var(self) -> torch.Tensor:
        mean_sq_x = self.face_edge_sum_sq_norm / 3
        # [..., F]

        sq_mean_x = (self.face_edge_sum_norm / 3).square()
        # [..., F]

        return (mean_sq_x - sq_mean_x) / (
            utils.EPS[sq_mean_x.dtype] + sq_mean_x.detach())

    @functools.cached_property
    def face_bary_coord_mat(self) -> torch.Tensor:  # [..., F, 4, 4]
        return make_bary_coord_mat(self.face_vert_pos, self.face_norm)

    @functools.cached_property
    def area_weighted_vert_norm(self) -> torch.Tensor:  # [..., V, 3]
        faces = self.mesh_graph.f_to_vvv

        face_norm = self.face_norm
        # [..., F, 3]

        buffer = utils.zeros(like=self.vert_pos)
        # [..., V, 3]

        buffer.index_add_(-2, faces[:, 0], face_norm)
        buffer.index_add_(-2, faces[:, 1], face_norm)
        buffer.index_add_(-2, faces[:, 2], face_norm)

        # buffer[..., faces[:, 0][i], :] += face_norm[..., i, :]
        # buffer[..., faces[:, 1][i], :] += face_norm[..., i, :]
        # buffer[..., faces[:, 2][i], :] += face_norm[..., i, :]

        return utils.vec_normed(buffer)

    def remesh(
        self, epochs_cnt: int, lr: float, betas: tuple[float, float],
    ) -> MeshData:
        vert_pos = self.vert_pos.clone().requires_grad_()

        optimizer = torch.optim.Adam([vert_pos], lr=lr, betas=betas)

        alpha_edge_var = 0.1
        alpha_lap_smoothness = 0.1

        for _ in range(epochs_cnt):
            optimizer.zero_grad()

            mesh_data = MeshData(self.mesh_graph, vert_pos)

            loss = \
                alpha_edge_var * mesh_data.face_edge_var.mean() + \
                alpha_lap_smoothness * mesh_data.l2_uni_lap_smoothness.mean()

            loss.backward()

            optimizer.step()

        return MeshData(
            self.mesh_graph,
            vert_pos.detach(),
        )

        """
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        mesh = bpy.data.meshes.new("mesh")
        obj = bpy.data.objects.new("object", mesh)
        bpy.context.collection.objects.link(obj)

        mesh.from_pydata(
            self.vert_pos.tolist(), [], self.mesh_graph.f_to_vvv.tolist())
        mesh.update()

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.modifier_add(type='REMESH')
        modifier = obj.modifiers[-1]
        modifier.mode = 'VOXEL'
        modifier.voxel_size = target_length
        modifier.use_smooth_shade = False

        bpy.ops.object.modifier_apply(modifier=modifier.name)

        remeshed_vert_pos = torch.tensor(
            [v.co[:] for v in obj.data.vertices],
            dtype=self.vert_pos.dtype, device=self.device)
        # [V_, 3]

        remeshed_faces = torch.tensor(
            [tuple(p.vertices) for p in obj.data.polygons],
            dtype=torch.long, device=self.device)
        # [F, 3]

        return MeshData(
            MeshGraph.from_faces(
                remeshed_vert_pos.shape[0],
                remeshed_faces,
                self.device,
            ),
            remeshed_vert_pos,
        )
        """

    def calc_unsigned_dist(
        self,
        point_pos: torch.Tensor,  # [..., 3]
    ) -> torch.Tensor:  # [...]
        assert self.vert_pos.shape[-1] == 3

        pp = point_pos

        utils.check_shapes(pp, (..., 3))

        face_bary_coord_mat = self.face_bary_coord_mat
        # [..., F, 4, 4]

        ls = utils.do_rt(
            face_bary_coord_mat[..., :3],  # [..., F, 4, 3]
            face_bary_coord_mat[..., 3],  # [..., F, 3]
            pp[..., None, :],  # [..., 1, 3]
        )
        # [..., F, 4]

        la = ls[..., 0]
        lb = ls[..., 1]
        lc = ls[..., 2]
        ln = ls[..., 3]
        # [..., F]

        p_to_v_dist = utils.vec_norm(pp[..., None, :] - self.vert_pos)
        # ([..., 3] -> [..., 1, 3]) - [..., V, 3]
        # [..., V, 3]
        # [..., V]

        p_to_fv_dict = p_to_v_dist[..., self.mesh_graph.f_to_vvv]
        # [..., F, 3]

        p_to_f_dist = torch.where(
            (0 <= la) & (0 <= lb) & (0 <= lc),
            ln.abs(),  # [..., F]
            p_to_fv_dict.min(-1)[0],  # [..., F]
        )
        # [..., F]

        return p_to_f_dist.min(-1)[0]

    def calc_signed_dist(
        self,
        point_pos: torch.Tensor,  # [..., 3]
    ) -> torch.Tensor:  # [...]
        assert self.vert_pos.shape[-1] == 3

        pp = point_pos

        utils.check_shapes(pp, (..., 3))

        face_bary_coord_mat = self.face_bary_coord_mat
        # [..., F, 4, 4]

        ls = utils.do_rt(
            face_bary_coord_mat[..., :3],  # [..., F, 4, 3]
            face_bary_coord_mat[..., 3],  # [..., F, 3]
            pp[..., None, :],  # [..., 1, 3]
        )
        # [..., F, 4]

        la = ls[..., 0]
        lb = ls[..., 1]
        lc = ls[..., 2]
        ln = ls[..., 3]
        # [..., F]

        p_to_v_dist = utils.vec_norm(pp[..., None, :] - self.vert_pos)
        # ([..., 3] -> [..., 1, 3]) - [..., V, 3]
        # [..., V, 3]
        # [..., V]

        p_to_fv_dict = p_to_v_dist[..., self.mesh_graph.f_to_vvv]
        # [..., F, 3]

        p_to_f_dist = torch.where(
            (0 <= la) & (0 <= lb) & (0 <= lc),
            ln.abs(),  # [..., F]
            p_to_fv_dict.min(-1)[0],  # [..., F]
        )
        # [..., F]

        min_p_to_f, min_p_to_fi = p_to_f_dist.min(-1, True)
        # [..., 1]
        # [..., 1]

        ret = torch.where(
            ln.gather(-1, min_p_to_fi) < 0,  # [..., 1]
            -min_p_to_f,  # [..., 1]
            min_p_to_f,  # [..., 1]
        )[..., 0]

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
                [[p[0]], [p[1]], [p[2]], [1]], dtype=utils.FLOAT))[..., None]

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
                point_pos.reshape(-1, 3).cpu(),
            )).to(point_pos)

        return ret.view(point_pos.shape[:-1])

    def show(self) -> None:
        self.mesh_graph.show(self.vert_pos)


@beartype
def read_obj(
    f: typing.TextIO,
) -> tuple[
    list[tuple[int, int, int]],  # vert_pos_faces[F, 3]
    list[tuple[int, int, int]],  # tex_vert_pos_faces[F, 3]
    list[tuple[int, int, int]],  # vert_nor_faces[F, 3]
    list[tuple[float, float, float]],  # vert_pos[VP, 3]
    list[tuple[float, float, float]],  # tex_vert_pos[TVP, 3]
    list[tuple[float, float, float]],  # vert_nor[VN, 3]
]:
    vert_pos_faces: list[tuple[int, int, int]] = list()
    tex_vert_pos_faces: list[tuple[int, int, int]] = list()
    vert_nor_faces: list[tuple[int, int, int]] = list()

    vert_pos: list[tuple[float, float, float]] = list()
    tex_vert_pos: list[tuple[float, float, float]] = list()
    vert_nor: list[tuple[float, float, float]] = list()

    def _read_float3(ks) -> tuple[float, float, float]:
        ka = float(ks[1]) if 1 < len(ks) else 0.0
        kb = float(ks[2]) if 2 < len(ks) else 0.0
        kc = float(ks[3]) if 3 < len(ks) else 0.0

        return ka, kb, kc

    def _read_int3(ks) -> tuple[int, int, int]:
        ka = int(ks[0]) if 0 < len(ks) else 0
        kb = int(ks[1]) if 1 < len(ks) else 0
        kc = int(ks[2]) if 2 < len(ks) else 0

        return ka, kb, kc

    def _handle_f(parts):
        vpa, vta, vna = _read_int3(parts[1].split("/"))
        vpb, vtb, vnb = _read_int3(parts[2].split("/"))
        vpc, vtc, vnc = _read_int3(parts[3].split("/"))

        vert_pos_faces.append((vpa - 1, vpb - 1, vpc - 1))
        tex_vert_pos_faces.append((vta - 1, vtb - 1, vtc - 1))
        vert_nor_faces.append((vna - 1, vnb - 1, vnc - 1))

    for line in f:
        parts = line.split()

        if len(parts) == 0:
            continue

        match parts[0]:
            case "v":
                vert_pos.append(_read_float3(parts))

            case "vt":
                tex_vert_pos.append(_read_float3(parts))

            case "vn":
                vert_nor.append(_read_float3(parts))

            case "f":
                _handle_f(parts)

    return \
        vert_pos_faces, tex_vert_pos_faces, vert_nor_faces, \
        vert_pos, tex_vert_pos, vert_nor
