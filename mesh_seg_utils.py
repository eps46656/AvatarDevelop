from __future__ import annotations

import dataclasses
import typing

import torch
from beartype import beartype

from . import mesh_utils, utils


@beartype
def vote(
    pix_to_face:  torch.Tensor,  # [..., H, W]
    ballot: torch.Tensor,  # [..., H, W]
    face_ballot_box: torch.Tensor,  # [F]
) -> None:
    # H image height
    # W image width
    # F number of faces

    H, W, F = -1, -2, -3

    H, W, F = utils.check_shapes(
        pix_to_face, (..., H, W),
        ballot, (..., H, W),
        face_ballot_box, (F,),
    )

    batch_shape = utils.broadcast_shapes(
        pix_to_face.shape[:-2],
        ballot.shape[:-2],
    )

    pix_to_face = pix_to_face.expand(*batch_shape, H, W)
    ballot = ballot.expand(*batch_shape, H, W)
    face_ballot_box = face_ballot_box.expand(*batch_shape, F)

    pix_to_face = pix_to_face.view(*batch_shape, H * W)
    # [..., H * W]

    ballot = ballot.view(*batch_shape, H * W)
    # [..., H * W]

    face_ballot_box.scatter_add_(-1, pix_to_face, ballot)

    """

    face_ballot_box[..., pixel_to_face[..., i]] += ballot[..., i]

    """


@beartype
def batch_vote(
    pixel_to_face:  torch.Tensor,  # [H, W]
    ballot: torch.Tensor,  # [B, H, W]
    face_ballot_box: torch.Tensor,  # [F, B]
):
    # B number of parts
    # H image height
    # W image width
    # F number of faces

    N, B, H, W, F = -1, -2, -3, -4

    N, B, H, W = utils.check_shapes(
        pixel_to_face, (N, H, W),
        ballot, (N, B, H, W),
        face_ballot_box, (F, B),
    )

    for n in range(N):
        for b in range(B):
            vote(
                pixel_to_face,  # [H, W]
                ballot[b],  # [H, W]
                b,
                face_ballot_box[:, b],  # [F]
            )


@beartype
def elect(
    pixel_to_face: list[torch.Tensor],  # [H, W]
    ballot: torch.Tensor,  # [..., K, H, W]
    faces_cnt: int,
):
    assert 0 <= faces_cnt

    K, H, W = -1, -2, -3

    K, H, W = utils.check_shapes(
        pixel_to_face, (..., H, W),
        ballot, (..., K, H, W),
    )

    batch_shape = utils.broadcast_shapes(
        pixel_to_face.shape[:-2],
        ballot.shape[:-2],
    )

    pixel_to_face = pixel_to_face.expand(*batch_shape, H, W)
    ballot = ballot.expand(*batch_shape, K, H, W)

    face_ballot_box = utils.zeros_like(
        ballot,
        shape=(K, faces_cnt + 1),  # F + 1 for -1 pixel to face index
    )

    for batch_idxes in utils.get_batch_idxes(batch_shape):
        for k in range(K):
            vote(
                pixel_to_face[batch_idxes],  # [H, W]
                ballot[*batch_idxes, k],  # [H, W]
                face_ballot_box[k],
            )

    face_obj_idx = face_ballot_box.argmax(0)
    # [F]

    pass


@beartype
def assign(
    face_ballot_box: torch.Tensor,  # [..., F, K]
    threshold: float = 0.5,
    bg_idx: int = -1,
):
    utils.check_shapes(face_ballot_box, (..., -1, -2))

    max_ballot, max_obj = face_ballot_box.max(-1)
    # max_ballot[..., F]
    # max_obj[..., F]

    return torch.where(
        threshold <= max_ballot,
        max_obj,
        bg_idx,
    )


@beartype
@dataclasses.dataclass
class FaceSegmentationResult:
    mesh_subdivision_result: mesh_utils.MeshSubdivisionResult
    sub_mesh_data: mesh_utils.MeshData
    sub_face_obj_idx: torch.Tensor  # [F_]
    target_faces: list[list[int]]


@beartype
class MeshSegmentor:
    def __init__(
        self,
        mesh_graph: mesh_utils.MeshGraph,

        vert_ballot_cnt: torch.Tensor,  # [V + 1, O]
        vert_ballot_box: torch.Tensor,  # [V + 1, O]

        device: typing.Optional[torch.device] = None,
    ):
        O = utils.check_shapes(
            vert_ballot_cnt, (mesh_graph.verts_cnt + 1, -1),
            vert_ballot_box, (mesh_graph.verts_cnt + 1, -1),
        )

        self.mesh_graph = mesh_graph

        self.vert_ballot_cnt = vert_ballot_cnt
        self.vert_ballot_box = vert_ballot_box

        if device is not None:
            self.mesh_graph = self.mesh_graph.to(device)
            self.vert_ballot_cnt = self.vert_ballot_cnt.to(device)
            self.vert_ballot_box = self.vert_ballot_box.to(device)

    @staticmethod
    def empty(dtype: torch.dtype, device: torch.device) -> MeshSegmentor:
        return MeshSegmentor(
            mesh_utils.MeshGraph.empty(0),
            torch.zeros((1, 0), dtype=dtype, device=device),
            torch.zeros((1, 0), dtype=dtype, device=device),
            device,
        )

    @staticmethod
    def from_mesh_graph(
        mesh_graph: mesh_utils.MeshGraph,
        objs_cnt: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> MeshSegmentor:
        return MeshSegmentor(
            mesh_graph,

            torch.zeros((mesh_graph.verts_cnt + 1, objs_cnt),
                        dtype=dtype, device=device),

            torch.zeros((mesh_graph.verts_cnt + 1, objs_cnt),
                        dtype=dtype, device=device),
        )

    @staticmethod
    def from_state_dict(
        state_dict: typing.Mapping[str, object],
        dtype: typing.Optional[torch.dtype],
        device: torch.device,
    ) -> MeshSegmentor:
        return MeshSegmentor(
            mesh_utils.MeshGraph.from_state_dict(
                state_dict["mesh_graph"], device),

            utils.tensor_deserialize(
                state_dict["vert_ballot_cnt"], dtype, device),

            utils.tensor_deserialize(
                state_dict["vert_ballot_box"], dtype, device),

            device,
        )

    def state_dict(self):
        return {
            "mesh_graph": self.mesh_graph.state_dict(),
            "vert_ballot_cnt": utils.tensor_serialize(self.vert_ballot_cnt),
            "vert_ballot_box": utils.tensor_serialize(self.vert_ballot_box),
        }

    def load_state_dict(
            self, state_dict: typing.Mapping[str, object]) -> MeshSegmentor:
        face_segmentor: MeshSegmentor = MeshSegmentor.from_state_dict(
            state_dict, self.mesh_graph.device)

        self.mesh_graph = face_segmentor.mesh_graph
        self.vert_ballot_cnt = face_segmentor.vert_ballot_cnt
        self.vert_ballot_box = face_segmentor.vert_ballot_box

        return self

    def vote(
        self,
        obj_idx: int,
        face_idx: torch.Tensor,  # [...]
        bary_coord: torch.Tensor,  # [..., 3]
        ballot: torch.Tensor,  # [...]
    ):
        utils.check_shapes(bary_coord, (..., 3))

        V = self.mesh_graph.verts_cnt

        shape = utils.broadcast_shapes(
            face_idx,
            bary_coord.shape[:-1],
            ballot,
        )

        N = shape.numel()

        face_idx = face_idx.expand(shape).reshape(N)  # [N]
        bary_coord = bary_coord.expand(*shape, 3).reshape(N, 3)  # [N, 3]
        ballot = ballot.expand(shape).reshape(N)  # [N]

        vert_idx = torch.where(
            (face_idx == -1)[:, None].expand(N, 3),
            V,
            self.mesh_graph.f_to_vvv[face_idx.clamp(0, None)]  # [N, 3]
        )

        vert_ballot = bary_coord * ballot.unsqueeze(-1)  # [N, 3]

        for i in range(3):
            self.vert_ballot_cnt[:, obj_idx].index_add_(
                0,
                vert_idx[:, i],
                bary_coord[:, i].to(self.vert_ballot_cnt),
            )

            self.vert_ballot_box[:, obj_idx].index_add_(
                0,
                vert_idx[:, i],
                vert_ballot[:, i].to(self.vert_ballot_box),
            )

    def segment(
        self,
        mesh_data: mesh_utils.MeshData,
    ) -> FaceSegmentationResult:
        max_vert_ballot, max_vert_ballot_idx = \
            self.vert_ballot_box[:-1].max(-1)
        # max_vert_ballot[V]
        # max_vert_ballot_idx[V]

        vert_bg_mask = 0 < max_vert_ballot

        vert_obj_kdx = torch.where(
            vert_bg_mask,
            max_vert_ballot_idx + 1,
            0,
        )
        # [V]

        vert_weight = 1 / torch.where(
            vert_bg_mask,

            1e-3 + max_vert_ballot / max_vert_ballot.max().clamp(1e-3, None),
            # [0, 1]

            1e3,
        )
        # [V]

        mesh_subdivision_result = mesh_data.mesh_graph.subdivide()

        sub_mesh_graph = mesh_subdivision_result.mesh_graph

        vert_src_table = mesh_subdivision_result.vert_src_table

        sub_vert_obj_kdx_ab = vert_obj_kdx[vert_src_table]
        # [V_, 2]

        sub_vert_obj_kdx = torch.where(
            sub_vert_obj_kdx_ab[:, 0] == sub_vert_obj_kdx_ab[:, 1],
            sub_vert_obj_kdx_ab[:, 0],
            0,
        )
        # [V_]

        sub_face_obj_idx = (
            sub_vert_obj_kdx[sub_mesh_graph.f_to_vvv].max(-1).values - 1
        ).to(utils.CPU_DEVICE)
        # [F_]

        edge_weight = vert_weight[mesh_subdivision_result.vert_src_table]
        # [V_, 2]

        sub_vert_pos = (
            mesh_data.vert_pos[mesh_subdivision_result.vert_src_table] *
            edge_weight.unsqueeze(-1)
            # [V_, 2, D]
        ).sum(-2) / (1e-2 + edge_weight.sum(-1, True))
        # [V_, D]

        sub_mesh_data = mesh_utils.MeshData(
            mesh_subdivision_result.mesh_graph,
            sub_vert_pos,
        )

        O = self.vert_ballot_box.shape[1]

        target_faces: list[list[int]] = [list() for _ in range(O)]

        for f, obj_idx in enumerate(sub_face_obj_idx):
            if 0 <= obj_idx:
                target_faces[int(obj_idx)].append(f)

        return FaceSegmentationResult(
            mesh_subdivision_result,
            sub_mesh_data,
            sub_face_obj_idx,
            target_faces,
        )
