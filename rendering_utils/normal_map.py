import typing

import torch
from beartype import beartype

from .. import avatar_utils, camera_utils, mesh_utils, transform_utils, utils
from .ras_mesh import *


@beartype
@utils.mem_clear
def make_normal_map(
    *,
    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,  # [...]

    mesh_data: mesh_utils.MeshData,

    pix_to_face: typing.Optional[torch.Tensor],  # [..., H, W]
    bary_coord: typing.Optional[torch.Tensor],  # [..., H, W, 3]
) -> torch.Tensor:  # [..., H, W, 3]
    H, W = camera_config.img_h, camera_config.img_w

    if pix_to_face is None or bary_coord is None:
        mesh_ras_result = rasterize_mesh(
            mesh_data.vert_pos,
            mesh_data.mesh_graph.f_to_vvv,
            camera_config=camera_config,
            camera_transform=camera_transform,
            faces_per_pixel=1,
            cull_backface=True,
        )

        s = camera_transform.shape

        pix_to_face = mesh_ras_result.pix_to_face.view(*s, H, W)
        bary_coord = mesh_ras_result.bary_coord.view(*s, H, W, 3)

    shape = utils.broadcast_shapes(
        camera_transform.shape,
        pix_to_face.shape[:-2],
        bary_coord.shape[:-3],
    )

    camera_transform = camera_transform.expand(shape)
    pix_to_face = pix_to_face.expand(*shape, H, W)
    bary_coord = bary_coord.expand(*shape, H, W, 3)

    hit_map = 0 <= pix_to_face
    # [N, H * W]

    vert_norm = mesh_data.area_weighted_vert_norm
    # [..., V, 3]

    f_vi = torch.where(
        hit_map[..., None].expand(*shape, H, W, 3),

        mesh_data.mesh_graph.f_to_vvv[  # [F, 3]
            pix_to_face,  # [..., H, W]
            :,
        ],  # [..., H, W, 3]

        0,
    ).reshape(*shape, H * W * 3, 1).expand(*shape, H * W * 3, 3)
    # [..., H, W, 3] -> [..., H * W * 3, 3]

    face_vert_norm = vert_norm.expand(*shape, *vert_norm.shape) \
        .gather(-2, f_vi).view(*shape, H, W, 3, 3)
    # [..., H, W, 3, 3]

    raw_pixel_nor = \
        face_vert_norm[..., 0, :] * bary_coord[..., 0, None] + \
        face_vert_norm[..., 1, :] * bary_coord[..., 1, None] + \
        face_vert_norm[..., 2, :] * bary_coord[..., 2, None]
    # [..., H, W, 3]

    nor_view_transform = transform_utils.ObjectTransform.from_matching(
        "RUB",
        dtype=camera_transform.dtype,
        device=camera_transform.device,
    )
    # camera <-> nor_view

    m: torch.Tensor = camera_transform.get_trans_to(
        nor_view_transform).to(raw_pixel_nor)
    # world -> nor_view
    # [..., 4, 4]

    print(f"{m.shape=}")
    print(f"{raw_pixel_nor.shape=}")

    raw_pixel_nor = (
        m[..., None, None, :3, :3] @ raw_pixel_nor[..., None])[..., 0]
    # [..., H, W, 3]

    pixel_nor = torch.where(
        hit_map[..., None].expand(*shape, H, W, 3),

        raw_pixel_nor * 0.5 + 0.5,

        torch.tensor(
            [0.5, 0.5, 1.0],
            dtype=raw_pixel_nor.dtype, device=raw_pixel_nor.device
        ).expand(*shape, H, W, 3)
    )
    # [..., H, W, 3]

    return pixel_nor


@beartype
@utils.mem_clear
def make_light_map(
    *,
    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,  # [...]

    mesh_data: mesh_utils.MeshData,

    pix_to_face: typing.Optional[torch.Tensor],  # [..., H, W]
    bary_coord: typing.Optional[torch.Tensor],  # [..., H, W, 3]
) -> torch.Tensor:  # [..., 4, H, W]
    H, W = camera_config.img_h, camera_config.img_w

    if pix_to_face is None or bary_coord is None:
        mesh_ras_result = rasterize_mesh(
            mesh_data.vert_pos,
            mesh_data.mesh_graph.f_to_vvv,
            camera_config=camera_config,
            camera_transform=camera_transform,
            faces_per_pixel=1,
            cull_backface=True,
        )

        s = camera_transform.shape

        pix_to_face = mesh_ras_result.pix_to_face.view(*s, H, W)
        bary_coord = mesh_ras_result.bary_coord.view(*s, H, W, 3)

    shape = utils.broadcast_shapes(
        camera_transform.shape,
        pix_to_face.shape[:-2],
        bary_coord.shape[:-3],
    )

    camera_transform = camera_transform.expand(shape)
    pix_to_face = pix_to_face.expand(*shape, H, W)
    bary_coord = bary_coord.expand(*shape, H, W, 3)

    hit_map = 0 <= pix_to_face
    # [N, H, W]

    vert_norm = mesh_data.area_weighted_vert_norm
    # [..., V, 3]

    f_vi = torch.where(
        hit_map[..., None].expand(*shape, H, W, 3),

        mesh_data.mesh_graph.f_to_vvv[  # [F, 3]
            pix_to_face,  # [..., H, W]
            :,
        ],  # [..., H, W, 3]

        0,
    ).reshape(*shape, H * W * 3, 1).expand(*shape, H * W * 3, 3)
    # [..., H, W, 3] -> [..., H * W * 3, 3]

    face_vert_norm = vert_norm.expand(*shape, *vert_norm.shape) \
        .gather(-2, f_vi).view(*shape, H, W, 3, 3)
    # [..., H, W, 3, 3]

    raw_pixel_nor = \
        face_vert_norm[..., 0, :] * bary_coord[..., 0, None] + \
        face_vert_norm[..., 1, :] * bary_coord[..., 1, None] + \
        face_vert_norm[..., 2, :] * bary_coord[..., 2, None]
    # [..., H, W, 3]

    sim = utils.vec_dot(
        raw_pixel_nor, camera_transform.vec_b.to(raw_pixel_nor))
    # [..., H, W]

    sim *= hit_map
    # [..., H, W]

    return torch.stack([
        sim,  # R[..., H, W]
        sim,  # G[..., H, W]
        sim,  # B[..., H, W]
        hit_map.to(sim),  # A[..., H, W]
    ], dim=-3)  # [..., 4, H, W]
