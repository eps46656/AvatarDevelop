import dataclasses

import torch
from beartype import beartype

import pytorch3d
import pytorch3d.structures

from . import camera_utils, transform_utils, utils


@dataclasses.dataclass
class RasterizePointResult:
    pix_to_points: torch.Tensor  # [..., H, W, PPP]
    zbuf: torch.Tensor  # [..., H, W, PPP]
    dist: torch.Tensor  # [..., H, W, PPP]


@beartype
def rasterize_point(
    vert_pos: torch.Tensor,  # [..., V, 3]
    vert_color: torch.Tensor,  # [..., 3]

    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,  # [...]
    # camera <-> world

    radius: float,

    points_per_pixel: int,
) -> dict:
    V, F = -1, -2

    V, F = utils.check_shapes(
        vert_pos, (..., V, 3),
        faces, (..., F, 3),
    )

    assert 0 < points_per_pixel

    img_h, img_w = camera_config.img_h, camera_config.img_w

    device = utils.check_devices(camera_transform, vert_pos)

    shape = utils.broadcast_shapes(
        vert_pos.shape[:-2],
        vert_color.shape[:-1],
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
    vert_color = utils.batch_expand(vert_color, (*shape, V), 1).to(torch.float)
    faces = utils.batch_expand(faces, shape, 2)
    world_view_mat = utils.batch_expand(world_view_mat, shape, 2) \
        .to(torch.float)
    camera_proj_mat = utils.batch_expand(camera_proj_mat, shape, 2) \
        .to(torch.float)

    raster_settings = pytorch3d.renderer.PointsRasterizationSettings(
        image_size=(img_h, img_w),
        radius=radius,
        points_per_pixel=points_per_pixel,
        max_faces_per_bin=1024 * 32,
    )

    pix_to_point = torch.empty(
        (*shape, img_h, img_w, points_per_pixel),
        dtype=torch.int32,
        device=device,
    )

    zbuf = torch.empty(
        (*shape, img_h, img_w, points_per_pixel),
        dtype=vert_pos.dtype,
        device=device,
    )

    dist = torch.empty(
        (*shape, img_h, img_w, points_per_pixel),
        dtype=vert_pos.dtype,
        device=device,
    )

    for flatten_batch_idx, batch_idx in utils.get_batch_idxes(shape):
        cur_world_view_mat = world_view_mat[batch_idx]
        # [4, 4]

        cameras = pytorch3d.renderer.PerspectiveCameras(
            R=cur_world_view_mat[:3, :3].T[None, ...],
            T=cur_world_view_mat[:3, 3][None, ...],
            K=camera_proj_mat[batch_idx][None, ...],
            in_ndc=True,
            device=device,
        )

        point_cloud = pytorch3d.structures.Pointclouds(
            points=[vert_pos[batch_idx]],
            features=[vert_color[batch_idx].view(V, 3)],
        ),

        rasterizer = pytorch3d.renderer.PointsRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        fragments = rasterizer(point_cloud)

        pix_to_point[batch_idx] = fragments.idx.view(
            img_h, img_w, points_per_pixel)

        zbuf[batch_idx] = fragments.zbuf.view(
            img_h, img_w, points_per_pixel)

        dist[batch_idx] = fragments.dist.view(
            img_h, img_w, points_per_pixel)

    return RasterizePointResult(
        pixel_to_points=pix_to_point,
        zbuf=zbuf,
        dist=dist,
    )
