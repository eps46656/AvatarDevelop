import collections
import dataclasses
import typing

import torch
from beartype import beartype

from . import utils


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

    batch_shapes = utils.BroadcastShapes(
        vertex_positions_a,
        vertex_positions_b,
        vertex_positions_c,
    )

    vertex_positions_a = vertex_positions_a.expand(batch_shapes)
    vertex_positions_b = vertex_positions_b.expand(batch_shapes)
    vertex_positions_c = vertex_positions_c.expand(batch_shapes)

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


@beartype
@dataclasses.dataclass
class MeshData:
    # V: vertices cnt
    # F: faces cnt
    # VP: adj vertex pairs cnt
    # FP: adj face pairs cnt

    vertices_cnt: int
    faces_cnt: int

    vertex_vertex_adj_rel_list: torch.Tensor  # [2, VP]
    face_vertex_adj_list: torch.Tensor  # [F, 3]
    face_face_adj_rel_list: torch.Tensor  # [2, FP]

    vertex_degrees: torch.Tensor  # [V]
    inv_vertex_degrees: torch.Tensor  # [V]

    def __init__(
        self,
        *,
        vertices_cnt: int,
        faces_cnt: int,

        vertex_vertex_adj_rel_list: torch.Tensor,  # [2, VP]
        face_vertex_adj_list: torch.Tensor,  # [F, 3]
        face_face_adj_rel_list: torch.Tensor,  # [2, FP]

        vertex_degrees: torch.Tensor,  # [V]
        inv_vertex_degrees: torch.Tensor,  # [V]
    ):
        assert 0 <= vertices_cnt
        assert 0 <= faces_cnt

        self.vertices_cnt = vertices_cnt
        self.faces_cnt = faces_cnt

        V = self.vertices_cnt
        F = self.faces_cnt

        VP, FP = -1, -2

        utils.CheckShapes(
            vertex_vertex_adj_rel_list, (2, VP),
            face_vertex_adj_list, (F, 3),
            face_face_adj_rel_list, (2, FP),

            vertex_degrees, (V,),
            inv_vertex_degrees, (V,),
        )

        self.vertex_vertex_adj_rel_list = vertex_vertex_adj_rel_list
        self.face_vertex_adj_list = face_vertex_adj_list
        self.face_face_adj_rel_list = face_face_adj_rel_list

        self.vertex_degrees = vertex_degrees
        self.inv_vertex_degrees = inv_vertex_degrees

    @staticmethod
    def FromFaceVertexAdjList(
        vertices_cnt: int,  # vertices_cnt
        face_vertex_adj_list: torch.Tensor,  # [F, 3]
        device: torch.device,
    ) -> typing.Self:
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
                    face_face_adj_rel_list.add(utils.MinMax(fs[i], fs[j]))

        face_vertex_adj_list = face_vertex_adj_list.to(
            dtype=torch.long, device=device)

        vertex_vertex_adj_rel_list = torch.tensor(
            sorted(edge_to_face_d.keys()),
            dtype=torch.long,
            device=device,
        ).transpose(0, 1)

        face_face_adj_rel_list = torch.tensor(
            sorted(face_face_adj_rel_list),
            dtype=torch.long,
            device=device,
        ).transpose(0, 1)

        inv_vertex_degrees = torch.where(
            vertex_degrees == 0,
            0,
            1.0 / vertex_degrees,
        ).to(dtype=utils.FLOAT, device=device)

        vertex_degrees = vertex_degrees.to(device=device)

        return MeshData(
            vertices_cnt=vertices_cnt,
            faces_cnt=faces_cnt,

            vertex_vertex_adj_rel_list=vertex_vertex_adj_rel_list,
            face_vertex_adj_list=face_vertex_adj_list,
            face_face_adj_rel_list=face_face_adj_rel_list,

            vertex_degrees=vertex_degrees,
            inv_vertex_degrees=inv_vertex_degrees,
        )

    def GetLapDiff(
        self,
        vertex_positions: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]
        utils.CheckShapes(vertex_positions, (..., self.vertices_cnt, -1))

        index_0 = self.vertex_vertex_adj_rel_list[0, :]
        index_1 = self.vertex_vertex_adj_rel_list[1, :]
        # [VP]

        positions_0 = vertex_positions[..., index_0, :]
        positions_1 = vertex_positions[..., index_1, :]
        # [..., VP, D]

        buffer = torch.zeros_like(vertex_positions)
        # [..., V, D]

        buffer.index_add_(dim=-2, index=index_0, source=positions_1)
        buffer.index_add_(dim=-2, index=index_1, source=positions_0)

        # buffer[..., index_0[i], :] += positions_1[..., i, :]
        # buffer[..., index_1[i], :] += positions_0[..., i, :]

        buffer = buffer * self.inv_vertex_degrees.unsqueeze(-1)
        # [..., V, D]

        return buffer - vertex_positions

    def GetLapDiffNaive(
        self,
        vertex_positions: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]
        utils.CheckShapes(vertex_positions, (..., self.vertices_cnt, -1))

        VP = self.vertex_vertex_adj_rel_list.shape[1]

        buffer = torch.zeros_like(vertex_positions)
        # [..., V, D]

        for vp in range(VP):
            va, vb = self.vertex_vertex_adj_rel_list[:, vp]

            buffer[..., va, :] += vertex_positions[..., vb, :]
            buffer[..., vb, :] += vertex_positions[..., va, :]

        buffer = buffer * self.inv_vertex_degrees.unsqueeze(-1)
        # [..., V, D]

        return buffer - vertex_positions

    def GetNormalSim(
        self,
        face_normals: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.CheckShapes(face_normals, (..., self.faces_cnt, -1))

        normals_0 = face_normals[
            ..., self.face_face_adj_rel_list[0, :], :]

        normals_1 = face_normals[
            ..., self.face_face_adj_rel_list[1, :], :]
        # [..., ?, D]

        return utils.Dot(normals_0, normals_1)

    def GetNormalSimNaive(
        self,
        face_normals: torch.Tensor,  # [..., F]
    ) -> torch.Tensor:  # [..., FP]
        utils.CheckShapes(face_normals, (..., self.faces_cnt, -1))

        FP = self.face_face_adj_rel_list.shape[1]

        ret = torch.empty(face_normals.shape[:-2] + (FP,), dtype=utils.FLOAT)

        for fp in range(FP):

            fa, fb = self.face_face_adj_rel_list[:, fp]

            ret[..., fp] = utils.Dot(
                face_normals[..., fa, :], (face_normals[..., fb, :]))

        return ret
