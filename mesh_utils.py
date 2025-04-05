from __future__ import annotations

import collections
import dataclasses
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

    vert_nor = torch.zeros_like(vert_pos)
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

    ret = torch.zeros_like(vals)
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

    ret = torch.zeros_like(vals)
    # [..., V, D]

    for pi in range(P):
        va, vb = adj_rel_list[pi]

        ret[..., va, :] += vals[..., vb, :]
        ret[..., vb, :] += vals[..., va, :]

    return ret


@dataclasses.dataclass
class PartialSubdivisionResult:
    f4: set[int]
    f2: set[int]


@beartype
class MeshData:
    # V: vertices cnt
    # F: faces cnt
    # E: edges cnt
    # FP: adj face pairs cnt

    def __init__(
        self,
        *,
        e_to_vv: torch.Tensor,  # [E, 2]
        f_to_vvv: torch.Tensor,  # [F, 3]
        ff: torch.Tensor,  # [FP, 2]

        vert_degrees: torch.Tensor,  # [V]
        inv_vert_degrees: torch.Tensor,  # [V]
    ):
        V, F, VP, FP = -1, -2, -3, -4

        V, F, VP, FP = utils.check_shapes(
            e_to_vv, (VP, 2),
            f_to_vvv, (F, 3),
            ff, (FP, 2),

            vert_degrees, (V,),
            inv_vert_degrees, (V,),
        )

        self.e_to_vv = e_to_vv
        self.f_to_vvv = f_to_vvv
        self.ff = ff

        self.vert_degrees = vert_degrees
        self.inv_vert_degrees = inv_vert_degrees

    @staticmethod
    def from_faces(
        verts_cnt: int,
        faces: torch.Tensor,  # [F, 3]
        device: torch.device,
    ) -> MeshData:
        faces_cnt = utils.check_shapes(faces, (-1, 3))

        faces = faces.to(utils.CPU_DEVICE, torch.int64)

        edge_to_face_d: collections.defaultdict[tuple[int, int], set[int]] = \
            collections.defaultdict(set)

        for f in range(faces_cnt):
            va, vb, vc = sorted(map(int, faces[f]))

            assert 0 <= va
            assert va < vb
            assert vb < vc
            assert vc < verts_cnt

            edge_to_face_d[(vb, vc)].add(f)
            edge_to_face_d[(va, vc)].add(f)
            edge_to_face_d[(va, vb)].add(f)

        ff: set[tuple[int, int]] = set()

        vert_degrees = torch.zeros((verts_cnt,), dtype=torch.int32)

        for (va, vb), fs in edge_to_face_d.items():
            assert 0 <= va
            assert va < vb
            assert vb < verts_cnt

            vert_degrees[va] += 1
            vert_degrees[vb] += 1

            for fa, fb in itertools.combinations(fs, 2):
                ff.add(utils.min_max(fa, fb))

        faces = faces.to(device, torch.int64)

        e_to_vv = torch.tensor(
            sorted(edge_to_face_d.keys()),
            dtype=torch.int64,
            device=device,
        )

        ff = torch.tensor(
            sorted(ff),
            dtype=torch.int64,
            device=device,
        )

        inv_vert_degrees = torch.where(
            vert_degrees == 0,
            0,
            1.0 / vert_degrees,
        ).to(device, torch.float64)

        return MeshData(
            e_to_vv=e_to_vv,
            f_to_vvv=faces,
            ff=ff,

            vert_degrees=vert_degrees.to(device),
            inv_vert_degrees=inv_vert_degrees,
        )

    @property
    def verts_cnt(self) -> int:
        return self.vert_degrees.shape[0]

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

    def to(self, *args, **kwargs) -> MeshData:
        d = {
            "e_to_vv": None,
            "f_to_vvv": None,
            "ff": None,

            "vert_degrees": None,
            "inv_vert_degrees": None,
        }

        for key, val in d.items():
            if val is None:
                cur_val = getattr(self, key)
                d[key] = None if cur_val is None else \
                    cur_val.to(*args, **kwargs)

        return MeshData(**d)

    def calc_vert_adj_sums(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]
        return calc_adj_sums(self.e_to_vv, vert_pos)

    def calc_vert_adj_sums_naive(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]
        return calc_adj_sums_naive(
            self.e_to_vv, vert_pos)

    def calc_lap_smoothing_loss(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # []
        vp = vert_pos

        D = utils.check_shapes(vp, (..., self.verts_cnt, -1))

        return utils.vec_norm(
            calc_adj_sums(self.e_to_vv, vert_pos) * self.inv_vert_degrees.unsqueeze(-1) - vp).mean()

    def calc_lap_smoothing_loss_pytorch3d(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # []
        utils.check_shapes(vert_pos, (..., self.verts_cnt, -1))

        assert 0 <= self.f_to_vvv.min()
        assert self.f_to_vvv.max() < self.verts_cnt

        mesh = pytorch3d.structures.Meshes(
            verts=[vert_pos],
            faces=[self.f_to_vvv],
            textures=None,
        ).to(vert_pos.device)

        utils.torch_cuda_sync()

        return pytorch3d.loss.mesh_laplacian_smoothing(mesh, method="uniform")

    def calc_lap_smoothing_loss_naive(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]

        buffer = torch.zeros_like(vert_pos)
        # [..., V, D]

        for vpi in range(self.edges_cnt):
            va, vb = self.e_to_vv[vpi]

            vp_a = vert_pos[va]
            vp_b = vert_pos[vb]

            buffer[..., va, :] += self.inv_vert_degrees[va] * (vp_b - vp_a)
            buffer[..., vb, :] += self.inv_vert_degrees[vb] * (vp_a - vp_b)

        return utils.vec_norm(buffer).mean()

    def calc_face_cos_sims(
        self,
        vecs: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(vecs, (..., self.faces_cnt, -1))

        vecs_0 = vecs[..., self.ff[:, 0], :]
        vecs_1 = vecs[..., self.ff[:, 1], :]
        # [..., FP, D]

        return utils.vec_dot(vecs_0, vecs_1)

    def calc_face_cos_sims_naive(
        self,
        vecs: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(vecs, (..., self.faces_cnt, -1))

        FP = self.adj_face_face_pairs_cnt

        ret = torch.empty(vecs.shape[:-2] + (FP,), dtype=utils.FLOAT)

        for fp in range(FP):
            fa, fb = self.ff[fp]
            ret[..., fp] = utils.vec_dot(vecs[..., fa, :], (vecs[..., fb, :]))

        return ret

    def get_face_diffs(
        self,
        face_vecs: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(face_vecs, (..., self.faces_cnt, -1))

        vecs_0 = face_vecs[..., self.ff[:, 0], :]
        vecs_1 = face_vecs[..., self.ff[:, 1], :]
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
            fa, fb = self.ff[fp]

            ret[..., fp] = face_vecs[..., fa, :] - (face_vecs[..., fb, :])

        return ret

    def lap_smoothing(
        self,
        vert_pos: torch.Tensor,  # [..., V, 3]
        t: float,
    ):
        vert_adj_centers = self.calc_vert_adj_sums(vert_pos) * \
            self.inv_vert_degrees

        return vert_pos * (1 - t) + vert_adj_centers * t

    def calc_signed_dists(
        self,
        vert_pos: torch.Tensor,  # [V, 3]
        point_pos: torch.Tensor,  # [..., 3]
    ) -> torch.Tensor:  # [...]
        vp = vert_pos
        pp = point_pos

        V = self.verts_cnt
        F = self.faces_cnt

        utils.check_shapes(
            vp, (V, 3),
            pp, (..., 3),
        )

        vp_a = vp[self.f_to_vvv[:, 0], :]
        vp_b = vp[self.f_to_vvv[:, 1], :]
        vp_c = vp[self.f_to_vvv[:, 2], :]
        # [F, 3]

        rs = torch.empty((F, 3, 3), dtype=vp.dtype, device=vp.device)

        axis_x = rs[:, :, 0] = vp_b - vp_a
        axis_y = rs[:, :, 1] = vp_c - vp_a
        axis_z = rs[:, :, 2] = utils.vec_cross(axis_x, axis_y)

        axis_z_norm = utils.vec_norm(axis_z)

        ks = (rs.inverse() @ (pp.unsqueeze(-2) - vp_a).unsqueeze(-1)) \
            .squeeze(-1)
        # [..., F, 3]

        kb = ks[..., 0]
        kc = ks[..., 1]
        kz = ks[..., 2]
        # [..., F]

        p_to_v_dist = utils.vec_norm(pp.unsqueeze(-2) - vp)
        # ([..., 3] -> [..., 1, 3]) - [V, 3]
        # [..., V, 3]
        # [..., V]

        p_to_f_dist = torch.where(
            (0 <= kb) & (0 <= kc) & (kb + kc <= 1),
            kz.abs() * axis_z_norm,
            torch.minimum(
                p_to_v_dist[..., self.f_to_vvv[:, 0]],
                torch.minimum(
                    p_to_v_dist[..., self.f_to_vvv[:, 1]],
                    p_to_v_dist[..., self.f_to_vvv[:, 2]],
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
        vert_pos: torch.Tensor,  # [V, 3]
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

        vp = vert_pos.to(utils.CPU_DEVICE)
        pp = point_pos.to(utils.CPU_DEVICE)

        V = self.verts_cnt
        F = self.faces_cnt

        utils.check_shapes(
            vp, (V, 3),
            pp, (..., 3),
        )

        vp_a = vp[self.f_to_vvv[:, 0], :]
        vp_b = vp[self.f_to_vvv[:, 1], :]
        vp_c = vp[self.f_to_vvv[:, 2], :]

        ret = torch.empty(pp.shape[:-1], dtype=pp.dtype)

        for pi in tqdm.tqdm(utils.get_idxes(pp.shape[:-1])):
            ans = float("inf")

            p = pp[pi]

            for vi in tqdm.tqdm(range(V)):
                cur_ans = _calc_dist(vp_a[vi], vp_b[vi], vp_c[vi], p)

                if abs(cur_ans) < abs(ans):
                    ans = cur_ans

            ret[pi] = ans

        return ret

    def calc_signed_dist_trimesh(
        self,
        vert_pos: torch.Tensor,  # [V, 3]
        point_pos: torch.Tensor,  # [..., 3]
    ):
        V = self.vert_degrees.shape[0]

        utils.check_shapes(vert_pos, (V, 3))

        tm = trimesh.Trimesh(
            vertices=vert_pos.to(utils.CPU_DEVICE),
            faces=self.f_to_vvv.to(utils.CPU_DEVICE),
            validate=True,
        )

        with utils.Timer():
            ret = torch.from_numpy(-trimesh.proximity.signed_distance(
                tm,
                point_pos,
            ))

        return ret

    @beartype
    @dataclasses.dataclass
    class SubdivisionResult:
        vert_src_table: torch.Tensor
        mesh_data: MeshData

    def subdivide(
        self,
        *,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> SubdivisionResult:
        f_to_vvv = self.f_to_vvv.to(utils.CPU_DEVICE)
        # [F, 3]

        e_to_vv = self.e_to_vv.to(utils.CPU_DEVICE)
        # [VP, 2]

        vv_to_e = {(va, vb): e for e, (va, vb) in enumerate(e_to_vv)}

        e_to_fs = [[] for _ in range(self.edges_cnt)]

        for f in range(self.faces_cnt):
            va, vb, vc = sorted(map(int, f_to_vvv[f]))

            e_to_fs[vv_to_e[(vb, vc)]].append(f)
            e_to_fs[vv_to_e[(va, vc)]].append(f)
            e_to_fs[vv_to_e[(va, vb)]].append(f)

        if target_faces is None:
            if target_edges is None:
                target_edges = range(self.edges_cnt)
        else:
            target_edges = set() if target_edges is None else set(target_edges)

            for f in target_faces:
                va, vb, vc = sorted(map(int, f_to_vvv[f]))

                target_edges.add(vv_to_e[(vb, vc)])
                target_edges.add(vv_to_e[(va, vc)])
                target_edges.add(vv_to_e[(va, vb)])

        se_cnts = [0] * self.faces_cnt

        edge_queue = set(target_edges)

        edge_mark = [False] * self.edges_cnt

        for e in edge_queue:
            edge_mark[e] = True

        while 0 < len(edge_queue):
            e = edge_queue.pop()

            for f in e_to_fs[e]:
                se_cnts[f] += 1

                if se_cnts[f] != 2:
                    continue

                va, vb, vc = sorted(map(int, f_to_vvv[f]))

                ea = vv_to_e[(vb, vc)]
                eb = vv_to_e[(va, vc)]
                ec = vv_to_e[(va, vb)]

                for ek in (ea, eb, ec):
                    if not edge_mark[ek]:
                        edge_queue.add(ek)
                        edge_mark[ek] = True

        e_to_new_v: dict[int, int] = dict()

        vert_src_table = torch.empty(
            (self.verts_cnt + edge_mark.count(True), 2), dtype=torch.int64)

        for i in range(self.verts_cnt):
            vert_src_table[i] = i

        for new_v, e in \
                enumerate((e for e, mark in enumerate(edge_mark) if mark),
                          self.verts_cnt):
            e_to_new_v[e] = new_v
            vert_src_table[new_v] = e_to_vv[e]

        new_faces: list[tuple[int, int, int]] = list()

        for f, cnt in enumerate(se_cnts):
            if cnt == 0:
                new_faces.append(tuple(self.f_to_vvv[f]))
                continue

            va, vb, vc = map(int, f_to_vvv[f])

            ea = vv_to_e[utils.min_max(vb, vc)]
            eb = vv_to_e[utils.min_max(va, vc)]
            ec = vv_to_e[utils.min_max(va, vb)]

            if 1 < cnt:
                ua = e_to_new_v[ea]
                ub = e_to_new_v[eb]
                uc = e_to_new_v[ec]

                new_faces.append((va, uc, ub))
                new_faces.append((vb, ua, uc))
                new_faces.append((vc, ub, ua))
                new_faces.append((ua, ub, uc))

                continue

            ka = edge_mark[ea]
            kb = edge_mark[eb]
            kc = edge_mark[ec]

            match ka * 4 + kb * 2 + kc:
                case 0b100:
                    ua = e_to_new_v[ea]
                    new_faces.append((va, vb, ua))
                    new_faces.append((vc, va, ua))

                case 0b010:
                    ua = e_to_new_v[eb]
                    new_faces.append((vb, vc, ua))
                    new_faces.append((va, vb, ua))

                case 0b001:
                    ua = e_to_new_v[eb]
                    new_faces.append((vc, va, ua))
                    new_faces.append((vb, vc, ua))

                case _:
                    raise utils.MismatchException()

        mesh_data = MeshData.from_faces(vert_src_table.shape[0], new_faces)

        return MeshData.SubdivisionResult(
            vert_src_table=vert_src_table,
            mesh_data=mesh_data,
        )

    @beartype
    @dataclasses.dataclass
    class ExtractionResult:
        vert_src_table: torch.Tensor
        face_src_table: torch.Tensor
        mesh_data: MeshData

    def extract(
        self,
        target_faces: typing.Iterable[int],
    ) -> ExtractionResult:
        target_faces = sorted(set(target_faces))

        f_to_vvv = self.f_to_vvv.to(utils.CPU_DEVICE)
        # [F, 3]

        v_mark = [False] * self.verts_cnt

        for f in target_faces:
            va, vb, vc = map(int, f_to_vvv[f])

            v_mark[va] = True
            v_mark[vb] = True
            v_mark[vc] = True

        v_to_new_v: dict[int, int] = dict()

        vert_src_table = torch.empty((v_mark.count(True),), dtype=torch.int64)

        for new_v, v in enumerate(v for v, mark in enumerate(v_mark) if mark):
            v_to_new_v[v] = new_v
            vert_src_table[new_v] = v

        new_f_to_vvv: list[tuple[int, int, int]] = list()

        face_src_table = torch.empty((len(target_faces),), dtype=torch.int64)

        for new_f, f in enumerate(target_faces):
            new_f_to_vvv.append(tuple(v_to_new_v[v] for v in f_to_vvv[f]))
            face_src_table[new_f] = f

        mesh_data = MeshData.from_faces(vert_src_table.shape[0], new_f_to_vvv)

        return MeshData.ExtractionResult(
            vert_src_table=vert_src_table,
            face_src_table=face_src_table,
            mesh_data=mesh_data,
        )

    def show(
        self,
        vert_pos: torch.Tensor,  # [V, 3]
    ):
        V = self.vert_degrees.shape[0]

        utils.check_shapes(vert_pos, (V, 3))

        tm = trimesh.Trimesh(
            vertices=vert_pos.to(utils.CPU_DEVICE),
            faces=self.f_to_vvv.to(utils.CPU_DEVICE),
            validate=True,
        )

        tm.show()
