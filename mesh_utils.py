import collections
import dataclasses
import typing

import torch
from beartype import beartype

import graph_utils
import utils


@beartype
def GetAreaVector(
    vertex_positions_a: torch.Tensor,  # [..., 3]
    vertex_positions_b: torch.Tensor,  # [..., 3]
    vertex_positions_c: torch.Tensor,  # [..., 3]
) -> torch.Tensor:  # [..., 3]
    utils.CheckShapes(
        vertex_positions_a, (..., 3),
        vertex_positions_b, (..., 3),
        vertex_positions_c, (..., 3),
    )

    batch_shapes = list(utils.GetCommonShape([
        vertex_positions_a.shape[:-1],
        vertex_positions_b.shape[:-1],
        vertex_positions_c.shape[:-1],
    ]))

    s = batch_shapes + [3]

    vertex_positions_a = vertex_positions_a.expand(s)
    vertex_positions_b = vertex_positions_b.expand(s)
    vertex_positions_c = vertex_positions_c.expand(s)

    return torch.linalg.cross(
        vertex_positions_b - vertex_positions_a,
        vertex_positions_c - vertex_positions_a)


@beartype
def GetAreaWeightedVertexNormals(
    *,
    faces: torch.Tensor,  # [F, 3]
    vertex_positions: torch.Tensor,  # [..., V, 3]
    vertex_positions_a: typing.Optional[torch.Tensor] = None,  # [..., F, 3]
    vertex_positions_b: typing.Optional[torch.Tensor] = None,  # [..., F, 3]
    vertex_positions_c: typing.Optional[torch.Tensor] = None,  # [..., F, 3]
):
    F, V = -1, -2

    F, V = utils.CheckShapes(
        faces, (F, 3),
        vertex_positions, (..., V, 3),
    )

    if vertex_positions_a is None:
        vertex_positions_a = vertex_positions[..., faces[:, 0], :]

    if vertex_positions_b is None:
        vertex_positions_b = vertex_positions[..., faces[:, 1], :]

    if vertex_positions_c is None:
        vertex_positions_c = vertex_positions[..., faces[:, 2], :]

    utils.CheckShapes(
        vertex_positions_a, (..., F, 3),
        vertex_positions_b, (..., F, 3),
        vertex_positions_c, (..., F, 3),
    )

    area_vector = GetAreaVector(
        vertex_positions_a,
        vertex_positions_b,
        vertex_positions_c,
    )
    # [..., F, 3]

    vertex_normals = torch.zeros_like(vertex_positions)
    # [..., V, 3]

    vertex_normals.index_add_(dim=-2, index=faces[:, 0], source=area_vector)
    vertex_normals.index_add_(dim=-2, index=faces[:, 1], source=area_vector)
    vertex_normals.index_add_(dim=-2, index=faces[:, 2], source=area_vector)

    # vertex_normals[..., faces[:, 0][i], :] += area_vector[..., i, :]
    # vertex_normals[..., faces[:, 1][i], :] += area_vector[..., i, :]
    # vertex_normals[..., faces[:, 2][i], :] += area_vector[..., i, :]

    return utils.Normalized(vertex_normals)


@dataclasses.dataclass
class MeshData:
    faces_cnt: int
    vertices_cnt: int

    face_vertex_adj_list: torch.Tensor  # [F, 3]

    vertex_vertex_adj_rel_list: torch.Tensor  # [2, ?]

    face_face_adj_rel_list: torch.Tensor  # [2, ?]

    vertex_degrees: torch.Tensor  # [V]

    inv_vertex_degrees: torch.Tensor  # [V]

    @beartype
    def FromFaceVertexAdjList(
        vertices_cnt: int,  # vertices_cnt
        face_vertex_adj_list: torch.Tensor,  # [F, 3]
        device: torch.device,
    ):
        faces_cnt, = utils.CheckShapes(face_vertex_adj_list, (-1, 3))

        face_vertex_adj_list = face_vertex_adj_list.to(device=utils.CPU)

        edge_to_face_d: collections.defaultdict[tuple[int, int],
                                                list[int]] = \
            collections.defaultdict(list)

        for f in range(faces_cnt):
            va, vb, vc = sorted(int(v) for v in face_vertex_adj_list[f, :])

            assert 0 <= va
            assert va < vb
            assert vb < vc
            assert vc < vertices_cnt

            edge_to_face_d[(vb, vc)].append(f)
            edge_to_face_d[(va, vc)].append(f)
            edge_to_face_d[(va, vb)].append(f)

        face_face_adj_rel_list: set[tuple[int, int]] = set()

        vertex_degrees = torch.zeros((vertices_cnt,), dtype=utils.INT)

        for (va, vb), fs in edge_to_face_d.items():
            vertex_degrees[va] += 1
            vertex_degrees[vb] += 1

            l = len(fs)

            for i in range(l):
                for j in range(i+1, l):
                    fa, fb = fs[i], fs[j]
                    face_face_adj_rel_list.add((min(fa, fb), max(fa, fb)))

        face_vertex_adj_list = face_vertex_adj_list.to(device=device)

        vertex_vertex_adj_rel_list = torch.tensor(
            sorted(edge_to_face_d.keys()),
            dtype=torch.long,
            device=device,
        ).transpose(0, 1)

        face_face_adj_rel_list = torch.tensor(
            sorted(face_face_adj_rel_list),
            dtype=torch.long,
            device=device,
        )

        inv_vertex_degrees = torch.where(
            vertex_degrees == 0,
            0,
            1.0 / vertex_degrees,
        ).to(dtype=utils.FLOAT, device=device)

        vertex_degrees = vertex_degrees.to(device=device)

        return MeshData(
            faces_cnt=faces_cnt,
            vertices_cnt=vertices_cnt,

            face_vertex_adj_list=face_vertex_adj_list,
            vertex_vertex_adj_rel_list=vertex_vertex_adj_rel_list,
            face_face_adj_rel_list=face_face_adj_rel_list,

            vertex_degrees=vertex_degrees,
            inv_vertex_degrees=inv_vertex_degrees,
        )

    @beartype
    def GetLapDiff(
        self,
        vertex_positions: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]
        utils.CheckShapes(vertex_positions, (..., self.vertices_cnt, -1))

        index_0 = self.vertex_vertex_adj_rel_list[0, :]
        index_1 = self.vertex_vertex_adj_rel_list[1, :]
        # [?]

        positions_0 = vertex_positions[..., index_0, :]
        positions_1 = vertex_positions[..., index_1, :]
        # [..., ?, D]

        buffer = torch.zeros_like(vertex_positions)
        # [..., V, D]

        buffer.index_add_(dim=-2, index=index_0, source=positions_1)
        buffer.index_add_(dim=-2, index=index_1, source=positions_0)

        # buffer[..., index_0[i], :] += positions_1[..., i, :]
        # buffer[..., index_1[i], :] += positions_0[..., i, :]

        buffer = buffer * self.inv_vertex_degrees.unsqueeze(-1)
        # [..., V, D]

        return buffer - vertex_positions

    @beartype
    def GetNormalSim(
        self,
        face_normals: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., ?]
        utils.CheckShapes(face_normals, (..., self.faces_cnt, -1))

        normals_0 = face_normals[
            ..., self.face_face_adj_rel_list[0, :], :]

        normals_1 = face_normals[
            ..., self.face_face_adj_rel_list[1, :], :]

        return utils.Dot(normals_0, normals_1)


@dataclasses.dataclass
class MeshData:
    vertices_cnt: int
    faces_cnt: int

    face_vertex_adj_rel_list: torch.Tensor  # [2, ?]

    vertex_vertex_adj_rel_list: torch.Tensor  # [2, ?]

    face_face_adj_rel_list: torch.Tensor  # [2, ?]

    vertex_degrees: torch.Tensor  # [V]


"""
class MeshManager:
    IDX_MIN = 0
    IDX_MAX = 2**24 - 1

    VERTEX_IDX_MIN = IDX_MIN
    VERTEX_IDX_MAX = (IDX_MIN + IDX_MAX) // 2

    FACE_IDX_MIN = (IDX_MIN + IDX_MAX) // 2 + 1
    FACE_IDX_MAX = IDX_MAX

    def _AllocateVertexIdx(self):
        return utils.AllocateID(
            MeshManager.VERTEX_IDX_MIN, MeshManager.VERTEX_IDX_MAX)

    def _AllocateFaceIdx(self):
        return utils.AllocateID(
            MeshManager.FACE_IDX_MIN, MeshManager.FACE_IDX_MAX)

    @staticmethod
    def _GetVS(va: int, vb: int, vc: int):
        va, vb = min(va, vb), max(va, vb)
        vb, vc = min(vb, vc), max(vb, vc)
        va, vb = min(va, vb), max(va, vb)

        return (va, vb, vc)

    def __init__(self):
        self.v_e_graph: graph_utils.Graph = graph_utils.Graph()
        self.e_f_graph: graph_utils.Graph = graph_utils.Graph()

        self.f_vs: dict[int, tuple[int, int, int]] = dict()

    def IsVertex(self, idx) -> bool:
        return idx in self.v_to_vs

    def IsFace(self, idx) -> bool:
        return idx in self.f_to_vs

    def GetFace(self, va: int, vb: int, vc: int) -> int:
        vs = tuple(sorted((va, vb, vc)))
        return self.f_to_vs.get(MeshManager._GetVS(va, vb, vc), (-1, (-1, -1, -1)))[0]

    def AddVertex(self, v: int = -1) -> int:
        if v == -1:
            v = self._AllocateVertexIdx()
        elif v in self.v_to_vs:
            return -1

        self.v_to_vs[v] = set()

        return v

    def AddFace(self, va: int, vb: int, vc: int, f: int = -1) -> int:
        vs = MeshManager._GetVS(va, vb, vc)

        if any(self.IsVertex(v) for v in vs) or vs[0] == vs[1] or vs[1] == vs[2]:
            return -1

        if f == -1:
            f = self._AllocateFaceIdx()

        old_f = self.vs_to_f.setdefault(vs, (f, (va, vb, vc)))[0]

        if old_f != f:
            self.vs_to_f[vs] = (old_f, (va, vb, vc))
            return old_f

        self.v_to_vs[va][vb] += 1
        self.v_to_vs[va][vc] += 1

        self.v_to_vs[vb][va] += 1
        self.v_to_vs[vb][vc] += 1

        self.v_to_vs[vc][va] += 1
        self.v_to_vs[va][vc] += 1

    def RemoveVertex(self, v: int) -> bool:
        if not self.IsVertex(v):
            return False

        # TODO

    def RemoveFace(self, f: int, keep_sole_vertices: bool = False) -> bool:
        if not self.IsFace(f):
            return False

        vs = MeshManager._GetVS(*self.f_to_vs.pop(f)[1])

        # self.

    def GetAdjVertices(self, idx: int):
        pass

    def GetAdjFaces(self, idx: int):
        pass

    def ToCompressedIdx(self, idx: int) -> int:
        pass

    def ToIdx(self, compress_idx: int) -> int:
        pass

    def GetFaceVertexAdjRelList(self) -> torch.Tensor:
        pass

    def GetMeshData(self) -> MeshData:
        pass

"""
