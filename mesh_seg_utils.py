from __future__ import annotations

import dataclasses
import typing

import torch
import tqdm
from beartype import beartype

from . import (avatar_utils, camera_utils, dataset_utils,
               kernel_splatting_utils, mesh_utils, rendering_utils,
               smplx_utils, transform_utils, utils, vision_utils)


@beartype
def vote(
    *,
    box_weight: torch.Tensor,  # [P],
    box_conf: torch.Tensor,  # [P],

    idx: torch.Tensor,  # [...]
    ballot_weight: torch.Tensor,  # [...]
    ballot_conf: torch.Tensor,  # [...]
) -> None:
    P = utils.check_shapes(
        box_weight, (-1,),
        box_conf, (-1,),
        idx, (...,),
        ballot_weight, (...,),
        ballot_conf, (...,),
    )

    shape = utils.broadcast_shapes(
        idx,
        ballot_weight,
        ballot_conf,
    )

    idx = idx.expand(shape)

    box_weight.index_add_(
        0, idx.flatten(),
        ballot_weight.expand(shape).to(box_weight).flatten(),
    )

    box_conf.index_add_(
        0, idx.flatten(),
        ballot_conf.expand(shape).to(box_conf).flatten(),
    )


@beartype
def face_vert_vote_ras(
    *,
    vert_weight: torch.Tensor,  # [V + 1],
    vert_conf: torch.Tensor,  # [V + 1],

    f_to_vvv: torch.Tensor,  # [F, 3]
    face_idx: torch.Tensor,  # [...]
    bary_coord: torch.Tensor,  # [..., 3]

    ballot_conf: torch.Tensor,  # [...]
) -> None:
    V_, F = -1, -2

    V_, F = utils.check_shapes(
        vert_weight, (V_,),
        vert_conf, (V_,),
        f_to_vvv, (F, 3),
        face_idx, (...,),
        bary_coord, (..., 3),
        ballot_conf, (...,),
    )

    V = V_ - 1

    vote(
        box_weight=vert_weight,
        box_conf=vert_conf,

        idx=torch.where(
            (face_idx == -1)[..., None].expand(*face_idx.shape, 3),
            V,
            f_to_vvv[face_idx.clamp(0, None)]
        ),  # [..., 3]

        ballot_weight=bary_coord,  # [..., 3]
        ballot_conf=bary_coord * ballot_conf[..., None],  # [..., 3]
    )


@beartype
def face_vert_vote(
    *,
    vert_weight: typing.Optional[torch.Tensor],  # [V + 1, O]
    vert_conf: typing.Optional[torch.Tensor],  # [V + 1, O]

    avatar_blender: avatar_utils.AvatarBlender,

    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,

    blending_param: typing.Any,

    obj_conf: list[typing.Iterable[torch.Tensor]],
    # [O][H, W]] or [O][K, H, W]

    batch_size: int,

    device: typing.Optional[torch.device],
) -> tuple[
    torch.Tensor,  # vert_weight[V + 1, O]
    torch.Tensor,  # vert_conf[V + 1, O]
]:
    H, W = camera_config.img_h, camera_config.img_w

    O = len(obj_conf)

    assert 0 < batch_size

    shape = utils.broadcast_shapes(
        camera_transform,
        blending_param,
    )

    camera_transform = camera_transform.expand(shape)
    blending_param = blending_param.expand(shape)

    avatar_model: avatar_utils.AvatarModel = avatar_blender.get_avatar_model()

    mesh_data = avatar_model.mesh_data
    mesh_graph = mesh_data.mesh_graph

    V = mesh_graph.verts_cnt

    if vert_weight is None:
        vert_weight = torch.zeros(
            (V + 1, O), dtype=torch.float64, device=device)

    if vert_conf is None:
        vert_conf = torch.zeros(
            (V + 1, O), dtype=torch.float64, device=device)

    utils.check_shapes(
        vert_weight, (V + 1, O),
        vert_conf, (V + 1, O),
    )

    with torch.no_grad():
        for B, batch_idx in tqdm.tqdm(dataset_utils.BatchIdxIterator(
            shape,
            batch_size=batch_size,
            shuffle=False,
        )):
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
                    cur_obj_conf = next(obj_conf[o])
                    # [H, W]

                    face_vert_vote_ras(
                        vert_weight=vert_weight[:, o],  # [V + 1]
                        vert_conf=vert_conf[:, o],  # [V + 1]

                        f_to_vvv=cur_avatar_model.mesh_graph.f_to_vvv,
                        # [F, 3]

                        face_idx=cur_pix_to_face[b],  # [H, W]
                        bary_coord=cur_bary_coord[b],  # [H, W, 3]

                        ballot_conf=cur_obj_conf[0, ...].to(device),
                        # [H, W]
                    )

    return vert_weight, vert_conf


@beartype
def refine_vert_conf(
    *,
    vert_conf: torch.Tensor,  # [V, O]

    vert_pos: torch.Tensor,  # [V, 3]

    kernel: typing.Callable[[torch.Tensor], torch.Tensor],

    iters_cnt: int,
) -> torch.Tensor:  # [V, O]
    V, O = -1, -2

    V, O = utils.check_shapes(
        vert_conf, (V, O),
        vert_pos, (V, 3),
    )

    vert_weight = 1 / (1e-2 + kernel_splatting_utils.calc_density(
        vert_pos,  # [V, 3]
        kernel,
    ))
    # [V]

    for _ in range(iters_cnt):
        vert_conf = kernel_splatting_utils.query(
            vert_pos,  # [V, 3]
            vert_conf,  # [V, K]
            vert_weight,  # [V]
            vert_pos,  # [V, 3]
            kernel,
        )

    return vert_conf


@beartype
@dataclasses.dataclass
class AssignSubMeshFacesResult:
    mesh_subdivide_result: mesh_utils.MeshSubdivideResult
    sub_vert_obj_idx: torch.Tensor  # [V_]
    sub_face_obj_idx: torch.Tensor  # [F_]
    target_faces: list[list[int]]  # [O]


@beartype
def assign_sub_mesh_faces(
    *,
    mesh_graph: mesh_utils.MeshGraph,
    vert_conf: torch.Tensor,  # [V + 1, O]
    threshold: typing.Optional[float],
) -> AssignSubMeshFacesResult:
    V, F = mesh_graph.verts_cnt, mesh_graph.faces_cnt

    O = utils.check_shapes(vert_conf, (V + 1, -1))

    vert_obj_conf, vert_obj_idx = vert_conf[:-1].max(-1)
    # vert_obj_conf[V]
    # vert_obj_idx[V]

    if threshold is not None:
        vert_obj_idx = torch.where(
            threshold <= vert_obj_conf,
            vert_obj_idx,
            -1,
        )

    e_to_vv = mesh_graph.e_to_vv

    edge_obj_idx = vert_obj_idx[e_to_vv]
    # [E, 2]

    border_edges = (edge_obj_idx[:, 0] != edge_obj_idx[:, 1]) \
        .nonzero().view(-1).tolist()
    # [E_]

    mesh_subdivide_result = mesh_graph.subdivide(target_edges=border_edges)

    sub_mesh_graph = mesh_subdivide_result.mesh_graph

    vert_src_table = mesh_subdivide_result.vert_src_table
    # [V_, 2]

    sub_vert_obj_idx_ab = vert_obj_idx[vert_src_table]
    # [V_, 2]

    sub_vert_obj_idx = torch.where(
        sub_vert_obj_idx_ab[:, 0] == sub_vert_obj_idx_ab[:, 1],
        sub_vert_obj_idx_ab[:, 0],
        -1,
    )
    # [V_]

    sub_face_obj_idx = (
        sub_vert_obj_idx[sub_mesh_graph.f_to_vvv].max(-1).values
    ).to(utils.CPU_DEVICE)
    # [F_]

    target_faces: list[list[int]] = [list() for _ in range(O)]

    for f, obj_idx in enumerate(sub_face_obj_idx):
        if 0 <= obj_idx:
            target_faces[int(obj_idx)].append(f)

    return AssignSubMeshFacesResult(
        mesh_subdivide_result=mesh_subdivide_result,
        sub_vert_obj_idx=sub_vert_obj_idx,
        sub_face_obj_idx=sub_face_obj_idx,
        target_faces=target_faces,
    )


@beartype
@dataclasses.dataclass
class ExtractSubMeshDataResult:
    mesh_subdivide_result: mesh_utils.MeshSubdivideResult

    union_sub_vert_pos_a: torch.Tensor  # [V_, 3]
    union_sub_vert_pos_b: torch.Tensor  # [V_, 3]
    union_sub_vert_t: torch.Tensor  # [V_]
    union_sub_vert_pos: torch.Tensor  # [V_, 3]

    union_sub_mesh_data: mesh_utils.MeshData

    sub_mesh_extract_result: list[mesh_utils.MeshExtractResult]
    sub_mesh_data: list[mesh_utils.MeshData]

    remaining_sub_mesh_extract_result: mesh_utils.MeshExtractResult
    remaining_sub_mesh_data: mesh_utils.MeshData


@beartype
def extract_sub_mesh_data(
    *,
    mesh_data: mesh_utils.MeshData,
    mesh_subdivide_result: mesh_utils.MeshSubdivideResult,
    target_faces: list[list[int]]  # [O]
) -> ExtractSubMeshDataResult:
    O = len(target_faces)

    vert_src_table = mesh_subdivide_result.vert_src_table
    # [V_, 2]

    V_ = vert_src_table.shape[0]

    union_vert_pos = mesh_data.vert_pos[vert_src_table]
    # [V_, 2, 3]

    union_sub_vert_pos_a = union_vert_pos[:, 0]
    union_sub_vert_pos_b = union_vert_pos[:, 1]
    # [V_, 3]

    union_sub_mesh_graph = mesh_subdivide_result.mesh_graph

    T_LB = 0.1
    T_RB = 0.9

    raw_vert_t = utils.zeros(
        like=mesh_data.vert_pos, shape=(V_,)
    ).requires_grad_()

    optimizer = torch.optim.Adam(
        [raw_vert_t],

        lr=max(
            1e-5,

            (
                mesh_data.vert_pos.max(0).values -
                mesh_data.vert_pos.min(0).values
            ).mean().item() * 1e-4
        ),

        betas=(0.5, 0.5),
    )

    vert_optimizable_table = vert_src_table[:, 0] != vert_src_table[:, 1]

    sub_mesh_graphs = [
        union_sub_mesh_graph.extract(
            target_faces=target_faces[o],
            remove_orphan_vert=False,
        ).mesh_graph
        for o in range(O)
    ]

    for epoch_i in tqdm.tqdm(range(800)):
        optimizer.zero_grad()

        union_sub_vert_t = torch.where(
            vert_optimizable_table,
            utils.smooth_clamp(raw_vert_t, T_LB, T_RB),
            0.5,
        )
        # [V_] [0.1, 0.9]

        cur_union_sub_vert_pos = \
            union_sub_vert_pos_a * (1 - union_sub_vert_t)[..., None] + \
            union_sub_vert_pos_b * union_sub_vert_t[..., None]

        loss = 0.0

        for o in range(O):
            loss = loss + mesh_utils.MeshData(
                sub_mesh_graphs[o],
                cur_union_sub_vert_pos,
            ).l2_uni_lap_smoothness

        loss.backward()

        optimizer.step()

    union_sub_vert_t = torch.where(
        vert_src_table[:, 0] == vert_src_table[:, 1],
        0.5,
        utils.smooth_clamp(raw_vert_t, T_LB, T_RB),
    )
    # [V_] [0.1, 0.9]

    union_sub_vert_pos = \
        union_sub_vert_pos_a * (1 - union_sub_vert_t)[..., None] + \
        union_sub_vert_pos_b * union_sub_vert_t[..., None]
    # [V_, 3]

    union_sub_mesh_data = mesh_utils.MeshData(
        mesh_graph=union_sub_mesh_graph,
        vert_pos=union_sub_vert_pos,
    )

    remaining_faces = set(range(union_sub_mesh_graph.faces_cnt))

    sub_mesh_extract_result: list[mesh_utils.MeshExtractResult] = list()
    sub_mesh_data: list[mesh_utils.MeshData] = list()

    for o in range(O):
        remaining_faces.difference_update(target_faces[o])

        cur_sub_mesh_extract_result = union_sub_mesh_graph.extract(
            target_faces=target_faces[o],
            remove_orphan_vert=True,
        )

        sub_mesh_extract_result.append(cur_sub_mesh_extract_result)

        sub_mesh_data.append(mesh_utils.MeshData(
            mesh_graph=cur_sub_mesh_extract_result.mesh_graph,
            vert_pos=union_sub_vert_pos[
                cur_sub_mesh_extract_result.vert_src_table],
        ))

    remaining_sub_mesh_extract_result = union_sub_mesh_graph.extract(
        target_faces=remaining_faces,
        remove_orphan_vert=True,
    )

    remaining_sub_mesh_data = mesh_utils.MeshData(
        mesh_graph=remaining_sub_mesh_extract_result.mesh_graph,

        vert_pos=union_sub_vert_pos[
            remaining_sub_mesh_extract_result.vert_src_table],
    )

    return ExtractSubMeshDataResult(
        mesh_subdivide_result=mesh_subdivide_result,

        union_sub_vert_pos_a=union_sub_vert_pos_a,
        union_sub_vert_pos_b=union_sub_vert_pos_b,
        union_sub_vert_t=union_sub_vert_t,
        union_sub_vert_pos=union_sub_vert_pos,

        union_sub_mesh_data=union_sub_mesh_data,

        sub_mesh_extract_result=sub_mesh_extract_result,
        sub_mesh_data=sub_mesh_data,

        remaining_sub_mesh_extract_result=remaining_sub_mesh_extract_result,
        remaining_sub_mesh_data=remaining_sub_mesh_data,
    )


@beartype
@dataclasses.dataclass
class ExtractSmplxSubModelDataResult:
    extract_sub_mesh_data_result: ExtractSubMeshDataResult

    subdivide_model_data_result: smplx_utils.ModelDataSubdivideResult

    sub_model_data_extract_result: list[smplx_utils.ModelDataExtractResult]

    remaining_sub_model_data_extract_result: smplx_utils.ModelDataExtractResult


@beartype
def extract_smplx_sub_model_data(
    *,
    model_data: smplx_utils.ModelData,
    blending_coeff_field: typing.Optional[smplx_utils.BlendingCoeffField],
    extract_sub_mesh_data_result: ExtractSubMeshDataResult,
):
    sub_mesh_extract_result = \
        extract_sub_mesh_data_result.sub_mesh_extract_result

    subdivide_model_data_result = model_data.subdivide(
        mesh_subdivide_result=extract_sub_mesh_data_result.mesh_subdivide_result,

        new_vert_t=extract_sub_mesh_data_result.union_sub_vert_t,
    )

    union_sub_model_data = subdivide_model_data_result.model_data

    def _f(extract_result):
        r = union_sub_model_data.extract(
            mesh_extract_result=extract_result,
        )

        if blending_coeff_field is not None:
            r.model_data = blending_coeff_field.query_model_data(r.model_data)

        return r

    return ExtractSmplxSubModelDataResult(
        extract_sub_mesh_data_result=extract_sub_mesh_data_result,

        subdivide_model_data_result=subdivide_model_data_result,

        sub_model_data_extract_result=list(map(_f, sub_mesh_extract_result)),

        remaining_sub_model_data_extract_result=_f(
            extract_sub_mesh_data_result.remaining_sub_mesh_extract_result),
    )
