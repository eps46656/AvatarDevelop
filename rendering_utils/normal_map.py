import typing

import torch
from beartype import beartype

from .. import avatar_utils, camera_utils, mesh_utils, transform_utils, utils
from .ras_mesh import *


@beartype
def make_normal_map(
    avatar_blender: avatar_utils.AvatarBlender,
    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,
    blending_param: typing.Any,
) -> torch.Tensor:
    H, W = camera_config.img_h, camera_config.img_w

    shape = utils.broadcast_shapes(
        camera_transform,
        blending_param,
    )

    avatar_model: avatar_utils.AvatarModel = avatar_blender(blending_param)

    mesh_data = avatar_model.mesh_data

    frags = rasterize_mesh(
        mesh_data.vert_pos,
        mesh_data.mesh_graph.f_to_vvv,
        camera_config=camera_config,
        camera_transform=camera_transform,
        faces_per_pixel=1,
        cull_backface=True,
    )

    pix_to_face = frags.pix_to_face.view(*shape, H, W)
    bary_coord = frags.bary_coords.view(*shape, H, W, 3)

    hit_map = 0 <= pix_to_face
    # [N, H * W]

    vert_nor: torch.Tensor = mesh_utils.get_area_weighted_vert_nor(
        faces=mesh_data.mesh_graph.f_to_vvv,
        vert_pos=mesh_data.vert_pos,
        vert_pos_a=mesh_data.face_vert_pos[..., 0, :],
        vert_pos_b=mesh_data.face_vert_pos[..., 1, :],
        vert_pos_c=mesh_data.face_vert_pos[..., 2, :],
    )  # [..., V, 3]

    f_vi = torch.where(
        hit_map[..., None].expand(*shape, H, W, 3),

        mesh_data.mesh_graph.f_to_vvv[  # [F, 3]
            pix_to_face,  # [..., H, W]
            :,
        ],  # [..., H, W, 3]

        utils.zeros_like(mesh_data.mesh_graph.f_to_vvv, shape=(1,)).expand(
            *shape, H, W, 3
        )
    ).reshape(*shape, H * W * 3, 1).expand(*shape, H * W * 3, 3)
    # [..., H, W, 3] -> [..., H * W * 3, 3]

    face_vert_nor = vert_nor.gather(-2, f_vi).view(*shape, H, W, 3, 3)
    # [..., H, W, 3, 3]

    raw_pixel_nor = \
        face_vert_nor[..., 0, :] * bary_coord[..., 0, None] + \
        face_vert_nor[..., 1, :] * bary_coord[..., 1, None] + \
        face_vert_nor[..., 2, :] * bary_coord[..., 2, None]
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

    raw_pixel_nor = (m[:3, :3] @ raw_pixel_nor[..., None])[..., 0]
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
