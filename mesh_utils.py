import collections
import itertools
import typing

import torch
import tqdm
import trimesh
from beartype import beartype

import pytorch3d
import pytorch3d.loss
import pytorch3d.structures

from . import utils


@beartype
def get_area_vec(
    vertex_positions_a: torch.Tensor,  # [..., 3]
    vertex_positions_b: torch.Tensor,  # [..., 3]
    vertex_positions_c: torch.Tensor,  # [..., 3]
) -> torch.Tensor:  # [..., 3]
    utils.check_shapes(
        vertex_positions_a, (..., 3),
        vertex_positions_b, (..., 3),
        vertex_positions_c, (..., 3),
    )

    batch_shape = utils.broadcast_shapes(
        vertex_positions_a,
        vertex_positions_b,
        vertex_positions_c,
    )

    vertex_positions_a = vertex_positions_a.expand(batch_shape)
    vertex_positions_b = vertex_positions_b.expand(batch_shape)
    vertex_positions_c = vertex_positions_c.expand(batch_shape)

    return torch.linalg.cross(
        vertex_positions_b - vertex_positions_a,
        vertex_positions_c - vertex_positions_a)


@beartype
def get_area_weighted_vertex_normals(
    *,
    faces: torch.Tensor,  # [F, 3]
    vertex_positions: torch.Tensor,  # [..., V, 3]
    vertex_positions_a: typing.Optional[torch.Tensor] = None,  # [..., F, 3]
    vertex_positions_b: typing.Optional[torch.Tensor] = None,  # [..., F, 3]
    vertex_positions_c: typing.Optional[torch.Tensor] = None,  # [..., F, 3]
):
    F, V = -1, -2

    F, V = utils.check_shapes(
        faces, (F, 3),
        vertex_positions, (..., V, 3),
    )

    if vertex_positions_a is None:
        vertex_positions_a = vertex_positions[..., faces[:, 0], :]

    if vertex_positions_b is None:
        vertex_positions_b = vertex_positions[..., faces[:, 1], :]

    if vertex_positions_c is None:
        vertex_positions_c = vertex_positions[..., faces[:, 2], :]

    utils.check_shapes(
        vertex_positions_a, (..., F, 3),
        vertex_positions_b, (..., F, 3),
        vertex_positions_c, (..., F, 3),
    )

    area_vector = get_area_vec(
        vertex_positions_a,
        vertex_positions_b,
        vertex_positions_c,
    )
    # [..., F, 3]

    vertex_normals = torch.zeros_like(vertex_positions)
    # [..., V, 3]

    vertex_normals.index_add_(-2, faces[:, 0], area_vector)
    vertex_normals.index_add_(-2, faces[:, 1], area_vector)
    vertex_normals.index_add_(-2, faces[:, 2], area_vector)

    # vertex_normals[..., faces[:, 0][i], :] += area_vector[..., i, :]
    # vertex_normals[..., faces[:, 1][i], :] += area_vector[..., i, :]
    # vertex_normals[..., faces[:, 2][i], :] += area_vector[..., i, :]

    return utils.normalized(vertex_normals)


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

    index_0 = adj_rel_list[:, 0]
    index_1 = adj_rel_list[:, 1]
    # [P]

    ret = torch.zeros_like(vals)
    # [..., V, D]

    ret.index_add_(-2, index_0, vals[..., index_1, :])
    ret.index_add_(-2, index_1, vals[..., index_0, :])

    # ret[..., index_0[i], :] += vals_1[..., i, :]
    # ret[..., index_1[i], :] += vals_0[..., i, :]

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

    ret = torch.zeros_like(vals)
    # [..., V, D]

    for pi in range(P):
        va, vb = adj_rel_list[pi]

        ret[..., va, :] += vals[..., vb, :]
        ret[..., vb, :] += vals[..., va, :]

    return ret


@beartype
class MeshData:
    # V: vertices cnt
    # F: faces cnt
    # VP: adj vertex pairs cnt
    # FP: adj face pairs cnt

    def __init__(
        self,
        *,
        vertex_vertex_adj_rel_list: torch.Tensor,  # [VP, 2]
        face_vertex_adj_list: torch.Tensor,  # [F, 3]
        face_face_adj_rel_list: torch.Tensor,  # [FP, 2]

        vertex_degrees: torch.Tensor,  # [V]
        inv_vertex_degrees: torch.Tensor,  # [V]
    ):
        V, F, VP, FP = -1, -2, -3, -4

        V, F, VP, FP = utils.check_shapes(
            vertex_vertex_adj_rel_list, (VP, 2),
            face_vertex_adj_list, (F, 3),
            face_face_adj_rel_list, (FP, 2),

            vertex_degrees, (V,),
            inv_vertex_degrees, (V,),
        )

        self.vertex_vertex_adj_rel_list = vertex_vertex_adj_rel_list
        self.face_vertex_adj_list = face_vertex_adj_list
        self.face_face_adj_rel_list = face_face_adj_rel_list

        self.vertex_degrees = vertex_degrees
        self.inv_vertex_degrees = inv_vertex_degrees

    @staticmethod
    def from_face_vertex_adj_list(
        vertices_cnt: int,  # vertices_cnt
        face_vertex_adj_list: torch.Tensor,  # [F, 3]
        device: torch.device,
    ) -> typing.Self:
        faces_cnt = utils.check_shapes(face_vertex_adj_list, (-1, 3))

        face_vertex_adj_list = face_vertex_adj_list.to(utils.CPU_DEVICE)

        edge_to_face_d: collections.defaultdict[tuple[int, int],
                                                set[int]] = \
            collections.defaultdict(set)

        for f in range(faces_cnt):
            va, vb, vc = sorted(int(v) for v in face_vertex_adj_list[f, :])

            assert 0 <= va
            assert va < vb
            assert vb < vc
            assert vc < vertices_cnt

            edge_to_face_d[(vb, vc)].add(f)
            edge_to_face_d[(va, vc)].add(f)
            edge_to_face_d[(va, vb)].add(f)

        face_face_adj_rel_list: set[tuple[int, int]] = set()

        vertex_degrees = torch.zeros((vertices_cnt,), dtype=utils.INT)

        for (va, vb), fs in edge_to_face_d.items():
            assert 0 <= va
            assert va < vb
            assert vb < vertices_cnt

            vertex_degrees[va] += 1
            vertex_degrees[vb] += 1

            for fa, fb in itertools.combinations(fs, 2):
                face_face_adj_rel_list.add(utils.min_max(fa, fb))

        face_vertex_adj_list = face_vertex_adj_list.to(device, torch.long)

        vertex_vertex_adj_rel_list = torch.tensor(
            sorted(edge_to_face_d.keys()),
            dtype=torch.long,
            device=device,
        )

        face_face_adj_rel_list = torch.tensor(
            sorted(face_face_adj_rel_list),
            dtype=torch.long,
            device=device,
        )

        inv_vertex_degrees = torch.where(
            vertex_degrees == 0,
            0,
            1.0 / vertex_degrees,
        ).to(device, utils.FLOAT)

        vertex_degrees = vertex_degrees.to(device)

        return MeshData(
            vertex_vertex_adj_rel_list=vertex_vertex_adj_rel_list,
            face_vertex_adj_list=face_vertex_adj_list,
            face_face_adj_rel_list=face_face_adj_rel_list,

            vertex_degrees=vertex_degrees,
            inv_vertex_degrees=inv_vertex_degrees,
        )

    @property
    def vertices_cnt(self) -> int:
        return self.vertex_degrees.shape[0]

    @property
    def faces_cnt(self) -> int:
        return self.face_vertex_adj_list.shape[0]

    @property
    def adj_vertex_vertex_pairs_cnt(self) -> int:
        return self.vertex_vertex_adj_rel_list.shape[0]

    @property
    def adj_face_vertex_pairs_cnt(self) -> int:
        return self.faces_cnt * 3

    @property
    def adj_face_face_pairs_cnt(self) -> int:
        return self.face_face_adj_rel_list.shape[0]

    def to(self, *args, **kwargs) -> typing.Self:
        d = {
            "vertex_vertex_adj_rel_list": None,
            "face_vertex_adj_list": None,
            "face_face_adj_rel_list": None,

            "vertex_degrees": None,
            "inv_vertex_degrees": None,
        }

        for key, val in d.items():
            if val is None:
                cur_val = getattr(self, key)
                d[key] = None if cur_val is None else \
                    cur_val.to(*args, **kwargs)

        return MeshData(**d)

    def midpoint_subdivision(self) -> typing.Self:
        es: dict[tuple[int, int], int] = dict()

        V = self.vertices_cnt
        VP = self.adj_vertex_vertex_pairs_cnt

        for vp in range(VP):
            va, vb = self.vertex_vertex_adj_rel_list[vp]

            es[(va, vb)] = V
            V += 1

        F = self.faces_cnt

        new_face_vertex_adj_list = torch.empty((F * 4, 3), dtype=torch.long)

        for f in range(F):
            va, vb, vc = self.face_vertex_adj_list[f]

            ea = es[utils.min_max(vb, vc)]
            eb = es[utils.min_max(vc, va)]
            ec = es[utils.min_max(va, vb)]

            new_face_vertex_adj_list[F * 0 + f] = (ea, eb, ec)
            new_face_vertex_adj_list[F * 1 + f] = (va, ec, eb)
            new_face_vertex_adj_list[F * 2 + f] = (vb, ea, ec)
            new_face_vertex_adj_list[F * 3 + f] = (vc, eb, ea)

        return MeshData.from_face_vertex_adj_list(
            V, new_face_vertex_adj_list)

    def calc_vertex_adj_sums(
        self,
        vertex_positions: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]
        return calc_adj_sums(self.vertex_vertex_adj_rel_list, vertex_positions)

    def calc_vertex_adj_sums_naive(
        self,
        vertex_positions: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]
        return calc_adj_sums_naive(
            self.vertex_vertex_adj_rel_list, vertex_positions)

    def calc_lap_smoothing_loss(
        self,
        vertex_positions: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # []
        vps = vertex_positions

        D = utils.check_shapes(vps, (..., self.vertices_cnt, -1))

        return utils.vector_norm(
            calc_adj_sums(self.vertex_vertex_adj_rel_list, vertex_positions) * self.inv_vertex_degrees.unsqueeze(-1) - vps).mean()

    def calc_lap_smoothing_loss_pytorch3d(
        self,
        vertex_positions: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # []
        utils.check_shapes(vertex_positions, (..., self.vertices_cnt, -1))

        assert 0 <= self.face_vertex_adj_list.min()
        assert self.face_vertex_adj_list.max() < self.vertices_cnt

        mesh = pytorch3d.structures.Meshes(
            verts=[vertex_positions],
            faces=[self.face_vertex_adj_list],
            textures=None,
        ).to(vertex_positions.device)

        utils.torch_cuda_sync()

        return pytorch3d.loss.mesh_laplacian_smoothing(mesh, method="uniform")

    def calc_lap_smoothing_loss_naive(
        self,
        vertex_positions: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]

        buffer = torch.zeros_like(vertex_positions)
        # [..., V, D]

        for vpi in range(self.adj_vertex_vertex_pairs_cnt):
            va, vb = self.vertex_vertex_adj_rel_list[vpi]

            vpa = vertex_positions[va]
            vpb = vertex_positions[vb]

            buffer[..., va, :] += self.inv_vertex_degrees[va] * (vpb - vpa)
            buffer[..., vb, :] += self.inv_vertex_degrees[vb] * (vpa - vpb)

        return utils.vector_norm(buffer).mean()

    def calc_face_cos_sims(
        self,
        vecs: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(vecs, (..., self.faces_cnt, -1))

        vecs_0 = vecs[..., self.face_face_adj_rel_list[:, 0], :]
        vecs_1 = vecs[..., self.face_face_adj_rel_list[:, 1], :]
        # [..., FP, D]

        return utils.dot(vecs_0, vecs_1)

    def calc_face_cos_sims_naive(
        self,
        vecs: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(vecs, (..., self.faces_cnt, -1))

        FP = self.adj_face_face_pairs_cnt

        ret = torch.empty(vecs.shape[:-2] + (FP,), dtype=utils.FLOAT)

        for fp in range(FP):

            fa, fb = self.face_face_adj_rel_list[fp]

            ret[..., fp] = utils.dot(
                vecs[..., fa, :], (vecs[..., fb, :]))

        return ret

    def get_face_diffs(
        self,
        face_vecs: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(face_vecs, (..., self.faces_cnt, -1))

        vecs_0 = face_vecs[..., self.face_face_adj_rel_list[:, 0], :]
        vecs_1 = face_vecs[..., self.face_face_adj_rel_list[:, 1], :]
        # [..., FP, D]

        return vecs_0 - vecs_1

    def get_face_diffs_naive(
        self,
        face_vecs: torch.Tensor,  # [..., F]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(face_vecs, (..., self.faces_cnt, -1))

        FP = self.adj_face_face_pairs_cnt

        ret = torch.empty(face_vecs.shape[:-2] + (FP,), dtype=utils.FLOAT)

        for fp in range(FP):
            fa, fb = self.face_face_adj_rel_list[fp]

            ret[..., fp] = face_vecs[..., fa, :] - (face_vecs[..., fb, :])

        return ret

    def lap_smoothing(
        self,
        vertex_positions: torch.Tensor,  # [..., V, 3]
        t: float,
    ):
        vertex_adj_centers = self.calc_vertex_adj_sums(vertex_positions) * \
            self.inv_vertex_degrees

        return vertex_positions * (1 - t) + vertex_adj_centers * t

    def pretty_subdivision(
        self,
        vertex_positions: torch.Tensor,  # [..., V, 3]
    ) -> tuple[
        typing.Self,
        torch.Tensor,  # [..., V + VP, 3]
    ]:
        vps_a = vertex_positions

        V = self.vertices_cnt
        VP = self.adj_vertex_vertex_pairs_cnt

        utils.check_shapes(vps_a, (..., V, 3))

        batch_shape = vps_a.shape[:-2]

        vps_b = torch.empty(
            batch_shape + (V + VP, 3),
            dtype=vps_a.dtype,
            device=vps_a.device,
        )
        # [..., V * 4, 3]

        subdivied_mesh_data = self.midpoint_subdivision()

        vps_b[..., :V, :] = self.lap_smoothing(vps_a, 1 / 8)

        vps_b[..., V:, :] = (
            vps_b[..., self.vertex_vertex_adj_rel_list[:, 0]] +
            vps_b[..., self.vertex_vertex_adj_rel_list[:, 1]]
        ) / 2

        vps_c = subdivied_mesh_data.lap_smoothing(vps_b, 1 / 4)

        return subdivied_mesh_data, vps_c

    def calc_signed_dists(
        self,
        vertex_positions: torch.Tensor,  # [V, 3]
        point_positions: torch.Tensor,  # [..., 3]
    ) -> torch.Tensor:  # [...]
        vps = vertex_positions
        pps = point_positions

        V = self.vertices_cnt
        F = self.faces_cnt

        utils.check_shapes(
            vps, (V, 3),
            pps, (..., 3),
        )

        vps_a = vps[self.face_vertex_adj_list[:, 0], :]
        vps_b = vps[self.face_vertex_adj_list[:, 1], :]
        vps_c = vps[self.face_vertex_adj_list[:, 2], :]
        # [F, 3]

        rs = torch.empty((F, 3, 3), dtype=vps.dtype, device=vps.device)

        axis_x = rs[:, :, 0] = vps_b - vps_a
        axis_y = rs[:, :, 1] = vps_c - vps_a
        axis_z = rs[:, :, 2] = torch.linalg.cross(axis_x, axis_y)

        axis_z_norm = utils.vector_norm(axis_z)

        ks = (rs.inverse() @ (pps.unsqueeze(-2) - vps_a).unsqueeze(-1)) \
            .squeeze(-1)
        # [..., F, 3]

        kb = ks[..., 0]
        kc = ks[..., 1]
        kz = ks[..., 2]
        # [..., F]

        p_to_v_dist = utils.vector_norm(pps.unsqueeze(-2) - vps)
        # ([..., 3] -> [..., 1, 3]) - [V, 3]
        # [..., V, 3]
        # [..., V]

        p_to_f_dist = torch.where(
            (0 <= kb) & (0 <= kc) & (kb + kc <= 1),
            kz.abs() * axis_z_norm,
            torch.minimum(
                p_to_v_dist[..., self.face_vertex_adj_list[:, 0]],
                torch.minimum(
                    p_to_v_dist[..., self.face_vertex_adj_list[:, 1]],
                    p_to_v_dist[..., self.face_vertex_adj_list[:, 2]],
                ))
        )
        # [..., F]

        min_p_to_f, min_p_to_fi = p_to_f_dist.min(-1, True)
        # [..., 1]
        # [..., 1]

        """

        kz[i, j, ...., p_to_fi[i, j, ..., f]]
        # [..., 1]

        """

        ret = torch.where(
            kz.gather(-1, min_p_to_fi) < 0,  # [..., 1]
            -min_p_to_f,  # [..., 1]
            min_p_to_f,  # [..., 1]
        ).squeeze(-1)

        return ret

    def calc_signed_dist_naive(
        self,
        vertex_positions: torch.Tensor,  # [V, 3]
        point_positions: torch.Tensor,  # [..., 3]
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
            z = torch.linalg.cross(ab, ac)

            z_norm = utils.vector_norm(z)

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

            da = utils.vector_norm(va - p)
            db = utils.vector_norm(vb - p)
            dc = utils.vector_norm(vc - p)

            ret = min(da, db, dc)

            if 0 <= kb and 0 <= kc and kb + kc <= 1:
                assert kz.abs() * z_norm <= ret + 1e-4, f"{kz.abs()=}, {ret=}"
                ret = min(ret, kz.abs() * z_norm)

            ret = float(ret)

            assert 0 <= ret

            return -ret if kz < 0 else ret

        vps = vertex_positions.to(utils.CPU_DEVICE)
        pps = point_positions.to(utils.CPU_DEVICE)

        V = self.vertices_cnt
        F = self.faces_cnt

        utils.check_shapes(
            vps, (V, 3),
            pps, (..., 3),
        )

        vas = vps[self.face_vertex_adj_list[:, 0], :]
        vbs = vps[self.face_vertex_adj_list[:, 1], :]
        vcs = vps[self.face_vertex_adj_list[:, 2], :]

        ret = torch.empty(pps.shape[:-1], dtype=pps.dtype)

        for pi in tqdm.tqdm(utils.get_idxes(pps.shape[:-1])):
            ans = float("inf")

            p = pps[pi]

            for vi in tqdm.tqdm(range(V)):
                cur_ans = _calc_dist(vas[vi], vbs[vi], vcs[vi], p)

                if abs(cur_ans) < abs(ans):
                    ans = cur_ans

            ret[pi] = ans

        return ret

    def calc_signed_dist_trimesh(
        self,
        vertex_positions: torch.Tensor,  # [V, 3]
        point_positions: torch.Tensor,  # [..., 3]
    ):
        V = self.vertex_degrees.shape[0]

        utils.check_shapes(vertex_positions, (V, 3))

        tm = trimesh.Trimesh(
            vertices=vertex_positions.to(utils.CPU_DEVICE),
            faces=self.face_vertex_adj_list.to(utils.CPU_DEVICE),
            validate=True,
        )

        with utils.Timer():
            ret = torch.from_numpy(-trimesh.proximity.signed_distance(
                tm,
                point_positions,
            ))

        return ret

    def show(
        self,
        vertex_positions: torch.Tensor,  # [V, 3]
    ):
        V = self.vertex_degrees.shape[0]

        utils.check_shapes(vertex_positions, (V, 3))

        tm = trimesh.Trimesh(
            vertices=vertex_positions.to(utils.CPU_DEVICE),
            faces=self.face_vertex_adj_list.to(utils.CPU_DEVICE),
            validate=True,
        )

        tm.show()
