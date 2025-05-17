from __future__ import annotations

import collections
import dataclasses
import typing

import torch
import tqdm
from beartype import beartype

from . import (avatar_utils, camera_utils, dataset_utils, mesh_utils,
               rendering_utils, transform_utils, utils)


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
) -> None:
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
class MeshSegmentationResult:
    mesh_subdivision_result: mesh_utils.MeshSubdivisionResult
    sub_vert_obj_kdx: torch.Tensor  # [V_]
    sub_face_obj_idx: torch.Tensor  # [F_]
    target_faces: list[list[int]]


@beartype
class MeshSegmentor:
    def __init__(
        self,
        *,
        mesh_graph: mesh_utils.MeshGraph,

        vert_ballot_cnt: torch.Tensor,  # [V + 1, O]
        vert_ballot_box: torch.Tensor,  # [V + 1, O]
    ):
        V = mesh_graph.verts_cnt

        O = utils.check_shapes(
            vert_ballot_cnt, (V + 1, -1),
            vert_ballot_box, (V + 1, -1),
        )

        self.mesh_graph = mesh_graph

        self.vert_ballot_cnt = vert_ballot_cnt
        self.vert_ballot_box = vert_ballot_box

    @staticmethod
    def from_empty(
        mesh_graph: mesh_utils.MeshGraph,

        objs_cnt: int,

        ballot_cnt_dtype: torch.dtype = torch.float64,
        ballot_box_dtype: torch.dtype = torch.float64,

        device: typing.Optional[torch.device] = None,
    ) -> MeshSegmentor:
        V = mesh_graph.verts_cnt
        O = objs_cnt

        vert_ballot_cnt = torch.zeros(
            (V + 1, O), dtype=ballot_cnt_dtype, device=device)

        vert_ballot_box = torch.zeros(
            (V + 1, O), dtype=ballot_box_dtype, device=device)

        return MeshSegmentor(
            mesh_graph=mesh_graph,

            vert_ballot_cnt=vert_ballot_cnt,
            vert_ballot_box=vert_ballot_box,
        )

    @staticmethod
    def from_state_dict(
        state_dict: typing.Mapping[str, object],
        dtype: typing.Optional[torch.dtype],
        device: torch.device,
    ) -> MeshSegmentor:
        return MeshSegmentor(
            mesh_graph=mesh_utils.MeshGraph.from_state_dict(
                state_dict["mesh_graph"], device),

            vert_ballot_cnt=utils.tensor_deserialize(
                state_dict["vert_ballot_cnt"], dtype, device),

            vert_ballot_box=utils.tensor_deserialize(
                state_dict["vert_ballot_box"], dtype, device),
        )

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("mesh_graph", self.mesh_graph.state_dict()),
            ("vert_ballot_cnt", utils.tensor_serialize(self.vert_ballot_cnt)),
            ("vert_ballot_box", utils.tensor_serialize(self.vert_ballot_box)),
        ])

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

        vert_idx = torch.where(
            (face_idx == -1)[..., None].expand(*face_idx.shape, 3),
            V,
            self.mesh_graph.f_to_vvv[face_idx.clamp(0, None)]
        ).expand(*shape, 3).reshape(N, 3)

        vert_ballot = bary_coord * ballot[..., None]

        for i in range(3):
            self.vert_ballot_cnt[:, obj_idx].index_add_(
                0,

                vert_idx[:, i],

                bary_coord[..., i].expand(shape)
                .to(self.vert_ballot_cnt).reshape(N),
            )

            self.vert_ballot_box[:, obj_idx].index_add_(
                0,

                vert_idx[:, i],

                vert_ballot[..., i].expand(shape)
                .to(self.vert_ballot_box).reshape(N),
            )

    def segment(self, threshold: float) -> MeshSegmentationResult:
        max_vert_ballot, max_vert_ballot_idx = \
            self.vert_ballot_box[:-1].max(-1)
        # max_vert_ballot[V]
        # max_vert_ballot_idx[V]

        vert_bg_mask = threshold <= max_vert_ballot

        vert_obj_kdx = torch.where(
            vert_bg_mask,
            max_vert_ballot_idx + 1,
            0,
        )
        # [V]

        mesh_subdivision_result: mesh_utils.MeshSubdivisionResult = \
            self.mesh_graph.subdivide()

        vert_src_table = mesh_subdivision_result.vert_src_table
        # [V_, 2]

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

        O = self.vert_ballot_box.shape[1]

        target_faces: list[list[int]] = [list() for _ in range(O)]

        for f, obj_idx in enumerate(sub_face_obj_idx):
            if 0 <= obj_idx:
                target_faces[int(obj_idx)].append(f)

        return MeshSegmentationResult(
            mesh_subdivision_result,
            sub_vert_obj_kdx,
            sub_face_obj_idx,
            target_faces,
        )


@beartype
def segment_mesh(
    *,
    avatar_blender: avatar_utils.AvatarBlender,

    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,

    blending_param: typing.Any,

    obj_mask: list[typing.Iterable[torch.Tensor]],  # [H, W]],  # [K, H, W]

    batch_size: int,

    device: typing.Optional[torch.device],
) -> MeshSegmentor:
    H, W = camera_config.img_h, camera_config.img_w

    O = len(obj_mask)

    assert 0 < batch_size

    shape = utils.broadcast_shapes(
        camera_transform,
        blending_param,
    )

    camera_transform = camera_transform.expand(shape)
    blending_param = blending_param.expand(shape)

    avatar_model: avatar_utils.AvatarModel = avatar_blender.get_avatar_model()

    mesh_data = avatar_model.mesh_data

    mesh_segmentor: MeshSegmentor = MeshSegmentor.from_empty(
        mesh_graph=mesh_data.mesh_graph,
        objs_cnt=O,
        ballot_cnt_dtype=torch.float64,
        ballot_box_dtype=torch.float64,
        device=device,
    )

    with torch.no_grad():
        for B, batch_idx in tqdm.tqdm(dataset_utils.BatchIdxIterator(
                shape, batch_size=batch_size, shuffle=False)):
            cur_camera_transform = camera_transform[batch_idx].to(device)

            cur_blending_param = blending_param[batch_idx].to(device)

            cur_avatar_model: avatar_utils.AvatarModel = \
                avatar_blender(cur_blending_param)

            mesh_ras_result = rendering_utils.rasterize_mesh(
                vert_pos=cur_avatar_model.vert_pos.to(device),
                faces=cur_avatar_model.mesh_graph.f_to_vvv.to(device),
                camera_config=camera_config,
                camera_transform=cur_camera_transform.to(device),
                faces_per_pixel=1,
            )

            cur_pix_to_face = mesh_ras_result.pix_to_face.reshape(B, H, W)
            # [B, H, W]

            cur_bary_coord = mesh_ras_result.bary_coord.reshape(B, H, W, 3)
            # [B, H, W, 3]

            for b in range(B):
                for o in range(O):
                    cur_mask = next(obj_mask[o])
                    # [H, W]

                    mesh_segmentor.vote(
                        o,
                        cur_pix_to_face[b],  # [H, W]
                        cur_bary_coord[b],  # [H, W, 3]
                        (cur_mask * 2 - 1).to(device),  # [H, W]
                    )

    return mesh_segmentor
