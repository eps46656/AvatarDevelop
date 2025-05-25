import dataclasses

import torch
from beartype import beartype

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures

from .. import camera_utils, transform_utils, utils


@dataclasses.dataclass
class MeshRasterizationResult:
    pix_to_face: torch.Tensor  # [..., H, W, FPP]
    bary_coord: torch.Tensor  # [..., H, W, FPP, 3]
    dist: torch.Tensor  # [..., H, W, FPP]


@beartype
def rasterize_mesh(
    vert_pos: torch.Tensor,  # [..., V, 3]
    faces: torch.Tensor,  # [..., F, 3]

    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,  # [...]
    # camera <-> world

    faces_per_pixel: int,

    cull_backface: bool = True,
) -> MeshRasterizationResult:
    # pix_to_face[B, H, W, FPP]
    # bary_coord[B, H, W, FPP, 3]
    # dist[B, H, W, FPP]

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
        camera_view_transform=camera_view_transform,
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
        dtype=utils.FLOAT,
    )
    # [..., 4, 4]

    vert_pos = utils.batch_expand(vert_pos, shape, 2).to(torch.float)
    faces = utils.batch_expand(faces, shape, 2)
    world_view_mat = utils.batch_expand(world_view_mat, shape, 2) \
        .to(torch.float)
    camera_proj_mat = utils.batch_expand(camera_proj_mat, shape, 2) \
        .to(torch.float)

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

    dist = torch.empty(
        (*shape, img_h, img_w, faces_per_pixel),
        dtype=vert_pos.dtype,
        device=device,
    )

    for batch_idx in utils.get_batch_idxes(shape):
        cur_world_view_mat = world_view_mat[batch_idx]
        # [4, 4]

        cameras = pytorch3d.renderer.PerspectiveCameras(
            R=cur_world_view_mat[:3, :3].T[None, ...],
            T=cur_world_view_mat[:3, 3][None, ...],
            K=camera_proj_mat[batch_idx][None, ...],
            in_ndc=True,
            device=device,
        )

        raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=faces_per_pixel,
            max_faces_per_bin=1024 * 32,
            cull_backfaces=cull_backface,
        )

        rasterizer = pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        mesh = pytorch3d.structures.Meshes(
            verts=vert_pos[batch_idx].view(1, V, 3).to(utils.FLOAT),
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

        dist[batch_idx] = fragments.dists.view(
            img_h, img_w, faces_per_pixel)

    return MeshRasterizationResult(
        pix_to_face=pix_to_face,
        bary_coord=bary_coord,
        dist=dist,
    )
