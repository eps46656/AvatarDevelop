import dataclasses

import numpy as np
import torch
from beartype import beartype

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures

from .. import camera_utils, transform_utils, utils, vision_utils


@dataclasses.dataclass
class RasterizeMeshResult:
    pix_to_face: torch.Tensor  # [..., H, W, FPP]
    bary_coord: torch.Tensor  # [..., H, W, FPP, 3]
    zbuf: torch.Tensor  # [..., H, W, FPP]


@beartype
def rasterize_mesh(
    vert_pos: torch.Tensor,  # [..., V, 3]
    faces: torch.Tensor,  # [..., F, 3]

    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,  # [...]
    # camera <-> world

    faces_per_pixel: int,

    cull_backface: bool = True,
) -> RasterizeMeshResult:
    # pix_to_face[..., H, W, FPP]
    # bary_coord[..., H, W, FPP, 3]
    # zbuf[..., H, W, FPP]

    V, F = -1, -2

    V, F = utils.check_shapes(
        vert_pos, (..., V, 3),
        faces, (..., F, 3),
    )

    assert 0 < faces_per_pixel

    img_h, img_w = camera_config.img_h, camera_config.img_w

    device = utils.check_devices(camera_transform, vert_pos)

    shape = utils.broadcast_shapes(
        vert_pos.shape[:-2],
        faces.shape[:-2],
        camera_transform,
    )

    camera_view_transform = transform_utils.ObjectTransform.from_matching(
        "LUF",
        dtype=camera_transform.dtype,
        device=camera_transform.device,
    )
    # camera <-> view

    world_view_mat = camera_transform.get_trans_to(camera_view_transform)
    # world -> view
    # [..., 4, 4]

    camera_proj_mat = camera_utils.make_proj_mat(
        camera_config=camera_config,
        camera_transform=camera_view_transform,
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
    )
    # [..., 4, 4]

    vert_pos = utils.batch_expand(vert_pos, shape, 2).to(torch.float)
    faces = utils.batch_expand(faces, shape, 2)
    world_view_mat = utils.batch_expand(world_view_mat, shape, 2) \
        .to(torch.float)
    camera_proj_mat = utils.batch_expand(camera_proj_mat, shape, 2) \
        .to(torch.float)

    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(img_h, img_w),
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
        max_faces_per_bin=1024 * 32,
        cull_backfaces=cull_backface,
    )

    pix_to_face = torch.empty(
        (*shape, img_h, img_w, faces_per_pixel),
        dtype=torch.int32,
        device=device,
    )

    bary_coord = torch.empty(
        (*shape, img_h, img_w, faces_per_pixel, 3),
        dtype=vert_pos.dtype,
        device=device,
    )

    zbuf = torch.empty(
        (*shape, img_h, img_w, faces_per_pixel),
        dtype=vert_pos.dtype,
        device=device,
    )

    for falttend_batch_idx, batch_idx in utils.get_batch_idxes(shape):
        cur_world_view_mat = world_view_mat[batch_idx]
        # [4, 4]

        cameras = pytorch3d.renderer.PerspectiveCameras(
            R=cur_world_view_mat[:3, :3].T[None, ...],
            T=cur_world_view_mat[:3, 3][None, ...],
            K=camera_proj_mat[batch_idx][None, ...],
            in_ndc=True,
            device=device,
        )

        rasterizer = pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        mesh = pytorch3d.structures.Meshes(
            verts=vert_pos[batch_idx].view(1, V, 3),
            faces=faces[batch_idx].view(1, F, 3),
            textures=None,
        )

        fragments: pytorch3d.renderer.mesh.rasterizer.Fragments = rasterizer(
            mesh)
        # fragments.pix_to_face[N, H, W, FPP]
        # fragments.bary_coords[N, H, W, FPP, 3]

        pix_to_face[batch_idx] = fragments.pix_to_face.view(
            img_h, img_w, faces_per_pixel)

        bary_coord[batch_idx] = fragments.bary_coords.view(
            img_h, img_w, faces_per_pixel, 3)

        zbuf[batch_idx] = fragments.zbuf.view(
            img_h, img_w, faces_per_pixel)

    return RasterizeMeshResult(
        pix_to_face=pix_to_face,
        bary_coord=bary_coord,
        zbuf=zbuf,
    )


@beartype
def render_soft_silhouette(
    vert_pos: torch.Tensor,  # [..., V, 3]
    faces: torch.Tensor,  # [..., F, 3]

    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,  # [...]
    # camera <-> world

    faces_per_pixel: int,

    sigma: float,

    cull_backface: bool,
) -> torch.Tensor:  # [..., 1, H, W]
    V, F = -1, -2

    V, F = utils.check_shapes(
        vert_pos, (..., V, 3),
        faces, (..., F, 3),
    )

    assert 0 < faces_per_pixel

    img_h, img_w = camera_config.img_h, camera_config.img_w

    img_size = (img_h, img_w)

    device = utils.check_devices(camera_transform, vert_pos)

    shape = utils.broadcast_shapes(
        vert_pos.shape[:-2],
        faces.shape[:-2],
        camera_transform,
    )

    camera_view_transform = transform_utils.ObjectTransform.from_matching(
        "LUF",
        dtype=camera_transform.dtype,
        device=camera_transform.device,
    )
    # camera <-> view

    world_view_mat = camera_transform.get_trans_to(camera_view_transform)
    # world -> view
    # [..., 4, 4]

    camera_proj_mat = camera_utils.make_proj_mat(
        camera_config=camera_config,
        camera_transform=camera_view_transform,
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
    )
    # [..., 4, 4]

    vert_pos = utils.batch_expand(vert_pos, shape, 2).to(torch.float)
    faces = utils.batch_expand(faces, shape, 2)
    world_view_mat = utils.batch_expand(world_view_mat, shape, 2) \
        .to(torch.float)
    camera_proj_mat = utils.batch_expand(camera_proj_mat, shape, 2) \
        .to(torch.float)

    mask = torch.empty(
        (*shape, img_h, img_w, faces_per_pixel),
        dtype=vert_pos.dtype,
        device=device,
    )

    mask: list[torch.Tensor] = list()

    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=img_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=faces_per_pixel,
        max_faces_per_bin=1024 * 32,
        cull_backfaces=cull_backface,
    )

    for batch_idx in utils.get_batch_idxes(shape):
        utils.mem_clear()

        cur_world_view_mat = world_view_mat[batch_idx]
        # [4, 4]

        cameras = pytorch3d.renderer.PerspectiveCameras(
            R=cur_world_view_mat[:3, :3].T[None, ...],
            T=cur_world_view_mat[:3, 3][None, ...],
            K=camera_proj_mat[batch_idx][None, ...],
            in_ndc=True,
            device=device,
        )

        rasterizer = pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        renderer = pytorch3d.renderer.MeshRenderer(
            rasterizer=rasterizer,
            shader=pytorch3d.renderer.SoftSilhouetteShader()
        )

        mesh = pytorch3d.structures.Meshes(
            verts=vert_pos[batch_idx].view(1, V, 3),
            faces=faces[batch_idx].view(1, F, 3),
            textures=None,
        )

        cur_mask = renderer(mesh).view(img_h, img_w, 4)[:, :, 3]
        # [H, W]

        mask.append(cur_mask)

    stacked_mask = torch.stack(mask, 0).view(*shape, 1, img_h, img_w)

    print(f"{stacked_mask.shape=}")
    print(f"{stacked_mask.min()=}")
    print(f"{stacked_mask.max()=}")

    return stacked_mask
