import dataclasses

import torch
from beartype import beartype

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures

from .. import camera_utils, transform_utils, utils


@beartype
def rasterize_mesh(
    vert_pos: torch.Tensor,  # [..., V, 3]
    faces: torch.Tensor,  # [..., F, 3]

    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,  # [...]
    # camera <-> world

    faces_per_pixel: int,

    cull_backfaces: bool = True,
) -> pytorch3d.renderer.mesh.rasterizer.Fragments:
    V, F = -1, -2

    V, F = utils.check_shapes(
        vert_pos, (..., V, 3),
        faces, (..., F, 3),
    )

    assert 0 < faces_per_pixel

    img_h, img_w = camera_config.img_h, camera_config.img_w

    image_size = (img_h, img_w)

    device = utils.check_devices(camera_transform, vert_pos)

    batch_shape = utils.broadcast_shapes(
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
        camera_view_transform=camera_view_transform,
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
        dtype=utils.FLOAT,
    )
    # [..., 4, 4]

    vert_pos = utils.batch_expand(vert_pos, batch_shape, -2).to(torch.float)
    faces = utils.batch_expand(faces, batch_shape, -2)
    world_view_mat = utils.batch_expand(world_view_mat, batch_shape, -2) \
        .to(torch.float)
    camera_proj_mat = utils.batch_expand(camera_proj_mat, batch_shape, -2) \
        .to(torch.float)

    N = batch_shape.numel()

    camera_R: list[torch.Tensor] = list()
    camera_T: list[torch.Tensor] = list()
    camera_K: list[torch.Tensor] = list()

    for idxes in utils.get_batch_idxes(batch_shape):
        cur_world_view_mat = world_view_mat[idxes]
        # [4, 4]

        cur_camera_proj_mat = camera_proj_mat[idxes]
        # [4, 4]

        camera_R.append(cur_world_view_mat[:3, :3].T)
        camera_T.append(cur_world_view_mat[:3, 3])
        camera_K.append(cur_camera_proj_mat.T)

    cameras = pytorch3d.renderer.PerspectiveCameras(
        R=torch.stack(camera_R),
        T=torch.stack(camera_T),
        K=torch.stack(camera_K),
        in_ndc=True,
        device=device,
    )

    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
        max_faces_per_bin=1024 * 32,
        cull_backfaces=cull_backfaces,
    )

    rasterizer = pytorch3d.renderer.MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    )

    mesh = pytorch3d.structures.Meshes(
        verts=vert_pos.to(utils.FLOAT).reshape(N, V, 3),
        faces=faces.reshape(N, F, 3),
        textures=None,
    )

    fragments: pytorch3d.renderer.mesh.rasterizer.Fragments = rasterizer(mesh)
    # fragments.pix_to_face[N, H, W, FPP]
    # fragments.bary_coords[N, H, W, FPP, 3]

    return fragments
