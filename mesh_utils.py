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


@dataclasses.dataclass
class PartialSubdivisionResult:
    f4: set[int]
    f2: set[int]


@beartype
class MeshData:
    # V: vertices cnt
    # F: faces cnt
    # VP: adj vert pairs cnt
    # FP: adj face pairs cnt

    def __init__(
        self,
        *,
        vert_vert_adj_rel_list: torch.Tensor,  # [VP, 2]
        face_vert_adj_list: torch.Tensor,  # [F, 3]
        face_face_adj_rel_list: torch.Tensor,  # [FP, 2]

        vert_degrees: torch.Tensor,  # [V]
        inv_vert_degrees: torch.Tensor,  # [V]
    ):
        V, F, VP, FP = -1, -2, -3, -4

        V, F, VP, FP = utils.check_shapes(
            vert_vert_adj_rel_list, (VP, 2),
            face_vert_adj_list, (F, 3),
            face_face_adj_rel_list, (FP, 2),

            vert_degrees, (V,),
            inv_vert_degrees, (V,),
        )

        self.vert_vert_adj_rel_list = vert_vert_adj_rel_list
        self.face_vert_adj_list = face_vert_adj_list
        self.face_face_adj_rel_list = face_face_adj_rel_list

        self.vert_degrees = vert_degrees
        self.inv_vert_degrees = inv_vert_degrees

    @staticmethod
    def from_face_vert_adj_list(
        vertices_cnt: int,  # vertices_cnt
        face_vert_adj_list: torch.Tensor,  # [F, 3]
        device: torch.device,
    ) -> typing.Self:
        faces_cnt = utils.check_shapes(face_vert_adj_list, (-1, 3))

        face_vert_adj_list = face_vert_adj_list.to(utils.CPU_DEVICE)

        edge_to_face_d: collections.defaultdict[tuple[int, int],
                                                set[int]] = \
            collections.defaultdict(set)

        for f in range(faces_cnt):
            va, vb, vc = sorted(int(v) for v in face_vert_adj_list[f])

            assert 0 <= va
            assert va < vb
            assert vb < vc
            assert vc < vertices_cnt

            edge_to_face_d[(vb, vc)].add(f)
            edge_to_face_d[(va, vc)].add(f)
            edge_to_face_d[(va, vb)].add(f)

        face_face_adj_rel_list: set[tuple[int, int]] = set()

        vert_degrees = torch.zeros((vertices_cnt,), dtype=utils.INT)

        for (va, vb), fs in edge_to_face_d.items():
            assert 0 <= va
            assert va < vb
            assert vb < vertices_cnt

            vert_degrees[va] += 1
            vert_degrees[vb] += 1

            for fa, fb in itertools.combinations(fs, 2):
                face_face_adj_rel_list.add(utils.min_max(fa, fb))

        face_vert_adj_list = face_vert_adj_list.to(device, torch.long)

        vert_vert_adj_rel_list = torch.tensor(
            sorted(edge_to_face_d.keys()),
            dtype=torch.long,
            device=device,
        )

        face_face_adj_rel_list = torch.tensor(
            sorted(face_face_adj_rel_list),
            dtype=torch.long,
            device=device,
        )

        inv_vert_degrees = torch.where(
            vert_degrees == 0,
            0,
            1.0 / vert_degrees,
        ).to(device, utils.FLOAT)

        vert_degrees = vert_degrees.to(device)

        return MeshData(
            vert_vert_adj_rel_list=vert_vert_adj_rel_list,
            face_vert_adj_list=face_vert_adj_list,
            face_face_adj_rel_list=face_face_adj_rel_list,

            vert_degrees=vert_degrees,
            inv_vert_degrees=inv_vert_degrees,
        )

    @property
    def vertices_cnt(self) -> int:
        return self.vert_degrees.shape[0]

    @property
    def faces_cnt(self) -> int:
        return self.face_vert_adj_list.shape[0]

    @property
    def adj_vert_vert_pairs_cnt(self) -> int:
        return self.vert_vert_adj_rel_list.shape[0]

    @property
    def adj_face_vert_pairs_cnt(self) -> int:
        return self.faces_cnt * 3

    @property
    def adj_face_face_pairs_cnt(self) -> int:
        return self.face_face_adj_rel_list.shape[0]

    def to(self, *args, **kwargs) -> typing.Self:
        d = {
            "vert_vert_adj_rel_list": None,
            "face_vert_adj_list": None,
            "face_face_adj_rel_list": None,

            "vert_degrees": None,
            "inv_vert_degrees": None,
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
        VP = self.adj_vert_vert_pairs_cnt

        for vp in range(VP):
            va, vb = self.vert_vert_adj_rel_list[vp]

            es[(va, vb)] = V
            V += 1

        F = self.faces_cnt

        new_face_vert_adj_list = torch.empty((F * 4, 3), dtype=torch.long)

        for f in range(F):
            va, vb, vc = self.face_vert_adj_list[f]

            ea = es[utils.min_max(vb, vc)]
            eb = es[utils.min_max(vc, va)]
            ec = es[utils.min_max(va, vb)]

            new_face_vert_adj_list[F * 0 + f] = (ea, eb, ec)
            new_face_vert_adj_list[F * 1 + f] = (va, ec, eb)
            new_face_vert_adj_list[F * 2 + f] = (vb, ea, ec)
            new_face_vert_adj_list[F * 3 + f] = (vc, eb, ea)

        return MeshData.from_face_vert_adj_list(
            V, new_face_vert_adj_list)

    def calc_vert_adj_sums(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]
        return calc_adj_sums(self.vert_vert_adj_rel_list, vert_pos)

    def calc_vert_adj_sums_naive(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]
        return calc_adj_sums_naive(
            self.vert_vert_adj_rel_list, vert_pos)

    def calc_lap_smoothing_loss(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # []
        vp = vert_pos

        D = utils.check_shapes(vp, (..., self.vertices_cnt, -1))

        return utils.vec_norm(
            calc_adj_sums(self.vert_vert_adj_rel_list, vert_pos) * self.inv_vert_degrees.unsqueeze(-1) - vp).mean()

    def calc_lap_smoothing_loss_pytorch3d(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # []
        utils.check_shapes(vert_pos, (..., self.vertices_cnt, -1))

        assert 0 <= self.face_vert_adj_list.min()
        assert self.face_vert_adj_list.max() < self.vertices_cnt

        mesh = pytorch3d.structures.Meshes(
            verts=[vert_pos],
            faces=[self.face_vert_adj_list],
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

        for vpi in range(self.adj_vert_vert_pairs_cnt):
            va, vb = self.vert_vert_adj_rel_list[vpi]

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

        vecs_0 = vecs[..., self.face_face_adj_rel_list[:, 0], :]
        vecs_1 = vecs[..., self.face_face_adj_rel_list[:, 1], :]
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
            fa, fb = self.face_face_adj_rel_list[fp]
            ret[..., fp] = utils.vec_dot(vecs[..., fa, :], (vecs[..., fb, :]))

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
        vert_pos: torch.Tensor,  # [..., V, 3]
        t: float,
    ):
        vert_adj_centers = self.calc_vert_adj_sums(vert_pos) * \
            self.inv_vert_degrees

        return vert_pos * (1 - t) + vert_adj_centers * t

    def pretty_subdivision(
        self,
        vert_pos: torch.Tensor,  # [..., V, 3]
    ) -> tuple[
        typing.Self,
        torch.Tensor,  # [..., V + VP, 3]
    ]:
        vp_a = vert_pos

        V = self.vertices_cnt
        VP = self.adj_vert_vert_pairs_cnt

        utils.check_shapes(vp_a, (..., V, 3))

        batch_shape = vp_a.shape[:-2]

        vp_b = torch.empty(
            batch_shape + (V + VP, 3),
            dtype=vp_a.dtype,
            device=vp_a.device,
        )
        # [..., V + VP, 3]

        subdivided_mesh_data = self.midpoint_subdivision()

        vp_b[..., :V, :] = self.lap_smoothing(vp_a, 1 / 8)

        vp_b[..., V:, :] = (
            vp_b[..., self.vert_vert_adj_rel_list[:, 0]] +
            vp_b[..., self.vert_vert_adj_rel_list[:, 1]]
        ) / 2

        vp_c = subdivided_mesh_data.lap_smoothing(vp_b, 1 / 4)

        return subdivided_mesh_data, vp_c

    def calc_signed_dists(
        self,
        vert_pos: torch.Tensor,  # [V, 3]
        point_pos: torch.Tensor,  # [..., 3]
    ) -> torch.Tensor:  # [...]
        vp = vert_pos
        pp = point_pos

        V = self.vertices_cnt
        F = self.faces_cnt

        utils.check_shapes(
            vp, (V, 3),
            pp, (..., 3),
        )

        vp_a = vp[self.face_vert_adj_list[:, 0], :]
        vp_b = vp[self.face_vert_adj_list[:, 1], :]
        vp_c = vp[self.face_vert_adj_list[:, 2], :]
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
                p_to_v_dist[..., self.face_vert_adj_list[:, 0]],
                torch.minimum(
                    p_to_v_dist[..., self.face_vert_adj_list[:, 1]],
                    p_to_v_dist[..., self.face_vert_adj_list[:, 2]],
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

        V = self.vertices_cnt
        F = self.faces_cnt

        utils.check_shapes(
            vp, (V, 3),
            pp, (..., 3),
        )

        vp_a = vp[self.face_vert_adj_list[:, 0], :]
        vp_b = vp[self.face_vert_adj_list[:, 1], :]
        vp_c = vp[self.face_vert_adj_list[:, 2], :]

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
            faces=self.face_vert_adj_list.to(utils.CPU_DEVICE),
            validate=True,
        )

        with utils.Timer():
            ret = torch.from_numpy(-trimesh.proximity.signed_distance(
                tm,
                point_pos,
            ))

        return ret

    def partially_subdivide(
        self,
        target_edges: typing.Iterable[int],
    ) -> typing.Self:
        fv = self.face_vert_adj_list.to(utils.CPU_DEVICE)
        # [F, 3]

        vv = self.vert_vert_adj_rel_list.to(utils.CPU_DEVICE)
        # [VP, 2]

        vv_to_e = {(va, vb): e for e, (va, vb) in enumerate(vv)}

        ef = [[] for _ in range(self.adj_vert_vert_pairs_cnt)]

        for f in range(self.faces_cnt):
            va, vb, vc = sorted(int(v) for v in fv[f, :])

            ef[vv_to_e[(vb, vc)]].append(f)
            ef[vv_to_e[(va, vc)]].append(f)
            ef[vv_to_e[(va, vb)]].append(f)

        se_cnts = [0] * self.faces_cnt

        edge_queue = set(target_edges)

        edge_mark = [False] * self.adj_vert_vert_pairs_cnt

        for e in edge_queue:
            edge_mark[e] = True

        while 0 < len(edge_queue):
            e = edge_queue.pop()

            for f in ef[e]:
                se_cnts[f] += 1

                if se_cnts[f] != 2:
                    continue

                va, vb, vc = sorted(int(v) for v in fv[f])

                ea = vv_to_e[(vb, vc)]
                eb = vv_to_e[(va, vc)]
                ec = vv_to_e[(va, vb)]

                for ek in (ea, eb, ec):
                    if not edge_mark[ek]:
                        edge_queue.add(ek)
                        edge_mark[ek] = True

        new_vert_idx = [-1] * self.adj_vert_vert_pairs_cnt

        v = self.vertices_cnt

        for e, mark in enumerate(edge_mark):
            if mark:
                new_vert_idx[e] = v
                v += 1

        new_faces = list()

        for f, cnt in enumerate(se_cnts):
            if cnt == 0:
                new_faces.append(tuple(self.face_vert_adj_list[f]))
                continue

            va, vb, vc = fv[f]

            ea = vv_to_e[utils.min_max(vb, vc)]
            eb = vv_to_e[utils.min_max(va, vc)]
            ec = vv_to_e[utils.min_max(va, vb)]

            if 1 < cnt:
                ua = new_vert_idx[ea]
                ub = new_vert_idx[eb]
                uc = new_vert_idx[ec]

                new_faces.append((va, uc, ub))
                new_faces.append((vb, ua, uc))
                new_faces.append((vc, ub, ua))
                new_faces.append((ua, ub, uc))

                continue

            ka = edge_mark[ea]
            kb = edge_mark[eb]
            kc = edge_mark[ec]

            match kc * 4 + kb * 2 + ka:
                case 0b001:
                    ua = new_vert_idx[ea]
                    new_faces.append((va, vb, ua))
                    new_faces.append((vc, va, ua))

                case 0b010:
                    ua = new_vert_idx[eb]
                    new_faces.append((vb, vc, ua))
                    new_faces.append((va, vb, ua))

                case 0b100:
                    ua = new_vert_idx[eb]
                    new_faces.append((vc, va, ua))
                    new_faces.append((vb, vc, ua))

                case _:
                    raise utils.MismatchException()

        return MeshData.from_face_vert_adj_list(new_faces)

    def show(
        self,
        vert_pos: torch.Tensor,  # [V, 3]
    ):
        V = self.vert_degrees.shape[0]

        utils.check_shapes(vert_pos, (V, 3))

        tm = trimesh.Trimesh(
            vertices=vert_pos.to(utils.CPU_DEVICE),
            faces=self.face_vert_adj_list.to(utils.CPU_DEVICE),
            validate=True,
        )

        tm.show()


@beartype
def partially_subdivide(
    mesh_data: MeshData,  # [F, 3]
    f4: set[int],  # [F]
):
    adj_f4_cnts = [0] * mesh_data.faces_cnt

    for f in f4:
        adj_f4_cnts[f] = 2

    f4_q = collections.deque(f4)

    while 0 < len(f4_q):
        f = f4_q.popleft()

        for fa in mesh_data.face_face_adj_rel_list[f]:
            adj_f4_cnts[fa] += 1

            if adj_f4_cnts[fa] == 2:
                f4_q.append(fa)

    f2_final = list()
    f4_final = list()

    for f, cnt in enumerate(adj_f4_cnts):
        match cnt:
            case 1:
                f2_final.append(f)
            case 2:
                f4_final.append(f)
            case 3:
                f4_final.append(f)
