import torch
from beartype import beartype

from .. import camera_utils, transform_utils, utils


import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures


@beartype
def rasterize_mesh(
    vertex_positions: torch.Tensor,  # [V, 3]
    faces: torch.Tensor,  # [F, 3]

    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,
    # camera <-> world

    faces_per_pixel: int,
):
    V, F = -1, -2

    V, F = utils.check_shapes(
        vertex_positions, (V, 3),
        faces, (F, 3),
    )

    assert 0 < faces_per_pixel

    img_h, img_w = camera_config.img_h, camera_config.img_w

    image_size = (img_h, img_w)

    device = utils.check_devices(camera_transform, vertex_positions)

    camera_view_transform = transform_utils.ObjectTransform \
        .from_matching("LUF").to(device)
    # camera <-> view

    camera_view_mat = camera_transform.get_trans_to(camera_view_transform)
    # world -> view

    camera_proj_mat = camera_utils.make_proj_mat(
        camera_config=camera_config,
        camera_view_transform=camera_view_transform,
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
        dtype=utils.FLOAT,
    )

    camera_view_mat = camera_view_mat.to(dtype=torch.float)
    camera_proj_mat = camera_proj_mat.to(dtype=torch.float)
    vertex_positions = vertex_positions.to(dtype=torch.float)

    match camera_config.proj_type:
        case camera_utils.ProjType.ORTH:
            cameras = pytorch3d.renderer.OrthographicCameras(
                R=camera_view_mat[:3, :3].transpose(0, 1).unsqueeze(0),
                T=camera_view_mat[:3, 3].unsqueeze(0),
                K=camera_proj_mat.transpose(0, 1).unsqueeze(0),
                in_ndc=True,
                device=device,
            )

        case camera_utils.ProjType.PERS:
            cameras = pytorch3d.renderer.PerspectiveCameras(
                R=camera_view_mat[:3, :3].transpose(0, 1).unsqueeze(0),
                T=camera_view_mat[:3, 3].unsqueeze(0),
                K=camera_proj_mat.transpose(0, 1).unsqueeze(0),
                in_ndc=True,
                device=device,
            )

        case _:
            assert False, ""

    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
    )

    rasterizer = pytorch3d.renderer.MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    )

    mesh = pytorch3d.structures.Meshes(
        verts=[vertex_positions.to(utils.FLOAT)],
        faces=[faces],
        textures=None,
    )

    fragments = rasterizer(mesh)

    pixel_to_faces = fragments.pix_to_face.squeeze(0)
    # [1, img_h, img_w, faces_per_pixel]
    # [img_h, img_w, faces_per_pixel]

    bary_coords = fragments.bary_coords.squeeze(-1)
    # [1, img_h, img_w, faces_per_pixel, 3]
    # [img_h, img_w, faces_per_pixel, 3]

    return {
        "pixel_to_faces": pixel_to_faces,
        "bary_coords": bary_coords,
    }
