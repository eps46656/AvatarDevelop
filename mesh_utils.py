from __future__ import annotations

import collections
import dataclasses
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


@beartype
@dataclasses.dataclass
class MeshSubdivisionResult:
    vert_src_table: torch.Tensor
    mesh_data: MeshData


@beartype
@dataclasses.dataclass
class MeshExtractionResult:
    vert_src_table: torch.Tensor
    face_src_table: torch.Tensor
    mesh_data: MeshData


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
    def from_faces(
        verts_cnt: int,
        faces: torch.Tensor,  # [F, 3]
        device: torch.device,
    ) -> MeshData:
        faces_cnt = utils.check_shapes(faces, (-1, 3))

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

        f_to_eee_l: list[tuple[int, int, int]] = \
            [None for _ in range(faces_cnt)]

        for f in range(faces_cnt):
            va, vb, vc = sorted(map(int, faces[f]))

            f_to_eee_l[f] = (
                vv_to_e[(vb, vc)],
                vv_to_e[(va, vc)],
                vv_to_e[(va, vb)],
            )

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

        faces = faces.to(device, torch.long)

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

        ff = torch.tensor(
            sorted(ff),
            dtype=torch.long,
            device=device,
        )

        if ff.numel() == 0:
            ff = f.expand(0, 2)

        inv_vert_deg = torch.where(
            vert_deg == 0,
            0,
            1.0 / vert_deg,
        ).to(device, torch.float64)

        return MeshData(
            e_to_vv=e_to_vv,
            vv_to_e=vv_to_e,

            f_to_vvv=faces,
            f_to_eee=f_to_eee,

            ff=ff,

            vert_deg=vert_deg.to(device),
            inv_vert_deg=inv_vert_deg,
        )

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

    def to(self, *args, **kwargs) -> MeshData:
        d = {
            "e_to_vv": None,
            "vv_to_e": self.vv_to_e,

            "f_to_vvv": None,
            "f_to_eee": None,

            "ff": None,

            "vert_deg": None,
            "inv_vert_deg": None,
        }

        for key, val in d.items():
            if val is None:
                cur_val = getattr(self, key)

                if d[key] is None:
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

    def calc_uni_lap_diff(
        self,
        vert_pos: torch.Tensor,  # [... , V, D]
    ) -> torch.Tensor:  # [..., V, D]
        utils.check_shapes(vert_pos, (..., self.verts_cnt, -1))

        return calc_adj_sums(self.e_to_vv, vert_pos) * \
            self.inv_vert_deg.unsqueeze(-1) - vert_pos

    def calc_l1_uni_lap_smoothness(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [...]
        return utils.vec_norm(self.calc_uni_lap_diff(vert_pos)).mean(-1)

    def calc_l2_uni_lap_smoothness(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [...]
        V = vert_pos.shape[-2]

        return self.calc_uni_lap_diff(vert_pos).square().sum((-1, -2)) / V

    def calc_uni_lap_smoothness_pytorch3d(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # []
        import pytorch3d
        import pytorch3d.loss
        import pytorch3d.structures

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

    def calc_lap_smoothness_naive(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # [..., V, D]

        buffer = utils.zeros_like(vert_pos)
        # [..., V, D]

        for vpi in range(self.edges_cnt):
            va, vb = self.e_to_vv[vpi]

            vp_a = vert_pos[va]
            vp_b = vert_pos[vb]

            buffer[..., va, :] += self.inv_vert_deg[va] * (vp_b - vp_a)
            buffer[..., vb, :] += self.inv_vert_deg[vb] * (vp_a - vp_b)

        return utils.vec_norm(buffer).mean()

    def uni_lap_smoothing(
        self,
        vert_pos: torch.Tensor,  # [..., V, 3]
        t: float,
    ):
        vert_adj_centers = self.calc_vert_adj_sums(vert_pos) * \
            self.inv_vert_deg

        return vert_pos * (1 - t) + vert_adj_centers * t

    def calc_cot_lap_diff(
        self,
        vert_pos: torch.Tensor,  # [..., V, 3]
    ) -> torch.Tensor:  # [..., V, 3]
        utils.check_shapes(vert_pos, (..., self.verts_cnt, 3))

        e_diff = \
            vert_pos[..., self.e_to_vv[:, 1], :] - \
            vert_pos[..., self.e_to_vv[:, 0], :]

        e_len = utils.vec_norm(e_diff.detach())
        # [..., E]

        e_len_a = e_len[..., self.f_to_eee[:, 0]]
        e_len_b = e_len[..., self.f_to_eee[:, 1]]
        e_len_c = e_len[..., self.f_to_eee[:, 2]]
        # [..., F]

        s = (e_len_a + e_len_b + e_len_c) / 2
        # [..., F]

        rareas = (s * (s - e_len_a) * (s - e_len_b) * (s - e_len_c)).rsqrt()
        # [..., F]

        e_sq_len_a = e_len_a.square()
        e_sq_len_b = e_len_b.square()
        e_sq_len_c = e_len_c.square()
        # [..., F]

        cot_a_4 = (e_sq_len_b + e_sq_len_c - e_sq_len_a) * rareas
        cot_b_4 = (e_sq_len_c + e_sq_len_a - e_sq_len_b) * rareas
        cot_c_4 = (e_sq_len_a + e_sq_len_b - e_sq_len_c) * rareas
        # [..., F]

        e_weight = utils.zeros_like(vert_pos, shape=e_len.shape)
        # [..., E]

        e_weight.index_add_(-1, self.f_to_eee[:, 0], cot_a_4)
        e_weight.index_add_(-1, self.f_to_eee[:, 1], cot_b_4)
        e_weight.index_add_(-1, self.f_to_eee[:, 2], cot_c_4)

        v_sum_weight = utils.zeros_like(vert_pos, shape=vert_pos.shape[:-1])
        # [..., V]

        v_sum_weight.index_add_(-1, self.e_to_vv[:, 0], e_weight)
        v_sum_weight.index_add_(-1, self.e_to_vv[:, 1], e_weight)

        weighted_e_diff = e_diff * e_weight.unsqueeze(-1)
        # [..., E, 3]

        buffer = utils.zeros_like(vert_pos)
        # [..., V, 3]

        buffer.index_add_(-2, self.e_to_vv[:, 0], weighted_e_diff, alpha=+1)
        buffer.index_add_(-2, self.e_to_vv[:, 1], weighted_e_diff, alpha=-1)

        return buffer / v_sum_weight.unsqueeze(-1)

    def calc_l1_cot_lap_smoothness(
        self,
        vert_pos: torch.Tensor,  # [..., V, 3]
    ) -> torch.Tensor:  # [..., V]
        return utils.vec_norm(self.calc_cot_lap_diff(vert_pos)).mean(-1)

    def calc_l2_cot_lap_smoothness(
        self,
        vert_pos: torch.Tensor,  # [..., V, 3]
    ) -> torch.Tensor:  # [..., V]
        V = vert_pos.shape[-2]

        return self.calc_cot_lap_diff(vert_pos).square().sum((-1, -2)) / V

    def calc_cot_lap_smoothness_pytorch3d(
        self,
        vert_pos: torch.Tensor,  # [..., V, D]
    ) -> torch.Tensor:  # []
        import pytorch3d
        import pytorch3d.loss
        import pytorch3d.structures

        utils.check_shapes(vert_pos, (..., self.verts_cnt, -1))

        assert 0 <= self.f_to_vvv.min()
        assert self.f_to_vvv.max() < self.verts_cnt

        mesh = pytorch3d.structures.Meshes(
            verts=[vert_pos],
            faces=[self.f_to_vvv],
            textures=None,
        ).to(vert_pos.device)

        utils.torch_cuda_sync()

        return pytorch3d.loss.mesh_laplacian_smoothing(mesh, method="cot")

    def calc_face_cos_sim(
        self,
        vecs: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(vecs, (..., self.faces_cnt, -1))

        vecs_0 = vecs[..., self.ff[:, 0], :]
        vecs_1 = vecs[..., self.ff[:, 1], :]
        # [..., FP, D]

        return utils.vec_dot(vecs_0, vecs_1)

    def calc_face_cos_sim_naive(
        self,
        vecs: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(vecs, (..., self.faces_cnt, -1))

        FP = self.adj_face_face_pairs_cnt

        ret = utils.empty_like(vecs, shape=vecs.shape[:-2] + (FP,))

        for fp in range(FP):
            fa, fb = self.ff[fp, :]
            ret[..., fp] = utils.vec_dot(vecs[..., fa, :], (vecs[..., fb, :]))

        return ret

    def calc_face_diff(
        self,
        face_vecs: torch.Tensor,  # [..., F, D]
    ) -> torch.Tensor:  # [..., FP, D]
        utils.check_shapes(face_vecs, (..., self.faces_cnt, -1))

        vecs_0 = face_vecs[..., self.ff[:, 0], :]
        vecs_1 = face_vecs[..., self.ff[:, 1], :]
        # [..., FP, D]

        return vecs_0 - vecs_1

    def calc_face_diff_naive(
        self,
        vecs: torch.Tensor,  # [..., F]
    ) -> torch.Tensor:  # [..., FP]
        utils.check_shapes(vecs, (..., self.faces_cnt, -1))

        FP = self.adj_face_face_pairs_cnt

        ret = utils.empty_like(vecs, shape=vecs.shape[:-2] + (FP,))

        for fp in range(FP):
            fa, fb = self.ff[fp, :]

            ret[..., fp] = vecs[..., fa, :] - (vecs[..., fb, :])

        return ret

    def calc_signed_dist(
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

        rs = utils.empty_like(vp, shape=(F, 3, 3))

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
        vert_pos: torch.Tensor,  # [V, 3]
        point_pos: torch.Tensor,  # [..., 3]
    ):
        V = self.vert_deg.shape[0]

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

    def vert_lap_prob(
        self,
        vert_features: torch.Tensor,  # [V, D]
    ):
        pass

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

        e_to_vv = self.e_to_vv.to(utils.CPU_DEVICE)
        # [VP, 2]

        e_to_fs = [[] for _ in range(self.edges_cnt)]

        for f in range(self.faces_cnt):
            va, vb, vc = sorted(map(int, f_to_vvv[f]))

            e_to_fs[self.vv_to_e[(vb, vc)]].append(f)
            e_to_fs[self.vv_to_e[(va, vc)]].append(f)
            e_to_fs[self.vv_to_e[(va, vb)]].append(f)

        if target_faces is None:
            if target_edges is None:
                target_edges = range(self.edges_cnt)
        else:
            target_edges = set() if target_edges is None else \
                set(map(int, target_edges))

            for f in target_faces:
                va, vb, vc = sorted(map(int, f_to_vvv[f]))

                target_edges.add(self.vv_to_e[(vb, vc)])
                target_edges.add(self.vv_to_e[(va, vc)])
                target_edges.add(self.vv_to_e[(va, vb)])

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

                ea = self.vv_to_e[(vb, vc)]
                eb = self.vv_to_e[(va, vc)]
                ec = self.vv_to_e[(va, vb)]

                for ek in (ea, eb, ec):
                    if not edge_mark[ek]:
                        edge_queue.add(ek)
                        edge_mark[ek] = True

        e_to_new_v: dict[int, int] = dict()

        vert_src_table = torch.empty(
            (self.verts_cnt + edge_mark.count(True), 2), dtype=torch.long)

        for i in range(self.verts_cnt):
            vert_src_table[i] = i

        for new_v, e in \
                enumerate((e for e, mark in enumerate(edge_mark) if mark),
                          self.verts_cnt):
            e_to_new_v[e] = new_v
            vert_src_table[new_v] = e_to_vv[e]

        new_faces: list[tuple[int, int, int]] = list()

        for f, cnt in enumerate(se_cnts):
            va, vb, vc = map(int, f_to_vvv[f])

            if cnt == 0:
                new_faces.append((va, vb, vc))
                continue

            ea = self.vv_to_e[utils.min_max(vb, vc)]
            eb = self.vv_to_e[utils.min_max(va, vc)]
            ec = self.vv_to_e[utils.min_max(va, vb)]

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

        new_faces = torch.tensor(new_faces, dtype=torch.long)
        # [F, 3]

        if new_faces.numel() == 0:
            new_faces = new_faces.expand(0, 3)

        mesh_data = MeshData.from_faces(
            vert_src_table.shape[0],
            new_faces,
            self.f_to_vvv.device,
        )

        return MeshSubdivisionResult(
            vert_src_table=vert_src_table,
            mesh_data=mesh_data,
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
            va, vb, vc = map(int, f_to_vvv[f])

            v_mark[va] = True
            v_mark[vb] = True
            v_mark[vc] = True

        v_to_new_v: dict[int, int] = dict()

        vert_src_table = torch.empty((v_mark.count(True),), dtype=torch.long)

        for new_v, v in enumerate(v for v, mark in enumerate(v_mark) if mark):
            v_to_new_v[v] = new_v
            vert_src_table[new_v] = v

        new_f_to_vvv = torch.empty((len(target_faces), 3), dtype=torch.long)

        face_src_table = torch.empty((len(target_faces),), dtype=torch.long)

        for new_f, f in enumerate(target_faces):
            new_f_to_vvv[new_f, 0] = v_to_new_v[int(f_to_vvv[f, 0])]
            new_f_to_vvv[new_f, 1] = v_to_new_v[int(f_to_vvv[f, 1])]
            new_f_to_vvv[new_f, 2] = v_to_new_v[int(f_to_vvv[f, 2])]

            face_src_table[new_f] = f

        mesh_data = MeshData.from_faces(
            vert_src_table.shape[0],
            new_f_to_vvv,
            device=self.f_to_vvv.device,
        )

        return MeshExtractionResult(
            vert_src_table=vert_src_table,
            face_src_table=face_src_table,
            mesh_data=mesh_data,
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
