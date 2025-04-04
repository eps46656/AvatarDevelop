import dataclasses
import typing

import torch
from beartype import beartype

import diff_gaussian_rasterization

from . import camera_utils, transform_utils, utils

camera_view_transform = transform_utils.ObjectTransform.from_matching("RDF")
# camera <-> view

camera_ndc_transform = transform_utils.ObjectTransform.from_matching("RDF")
# camera <-> ndc


@beartype
@dataclasses.dataclass
class RenderGaussianResult:
    colors: torch.Tensor  # [..., C, H, W]
    radii: torch.Tensor


@beartype
def render_gaussian(
    camera_config: camera_utils.CameraConfig,

    camera_transform: transform_utils.ObjectTransform,
    # camera <-> world
    # [...]

    sh_degree: int,  # 0 ~ 2

    bg_color: torch.Tensor,  # [..., C]

    gp_means: torch.Tensor,  # [..., N, 3]
    gp_rots: torch.Tensor,  # [..., N, 4] quaternion
    gp_scales: torch.Tensor,  # [..., N, 3]

    gp_shs: typing.Optional[torch.Tensor],  # [..., N, (sh_degress + 1)**2, C]
    gp_colors: typing.Optional[torch.Tensor],  # [..., N, C]

    gp_opacities: torch.Tensor,  # [..., N, 1]

    device: torch.device,
):
    cur_camera_view_transform = camera_view_transform.to(
        camera_transform.device)

    cur_camera_ndc_transform = camera_ndc_transform.to(
        camera_transform.device)

    world_view_mat = camera_transform.get_trans_to(cur_camera_view_transform)
    # world -> view [..., 4, 4]

    view_ndc_mat = camera_utils.make_proj_mat_with_config(
        camera_config=camera_config,

        camera_view_transform=cur_camera_view_transform,

        proj_config=camera_utils.ProjConfig(
            camera_proj_transform=cur_camera_ndc_transform,
            # camera <-> ndc

            delta_u=1.0,
            delta_d=1.0,
            delta_l=1.0,
            delta_r=1.0,
            delta_f=1.0,
            delta_b=0.0,
        ),

        dtype=utils.FLOAT,
    )  # view -> ndc [4, 4]

    world_ndc_mat = view_ndc_mat @ world_view_mat
    # [..., 4, 4]

    # ---

    assert 0 <= sh_degree <= 2

    # ---

    N, C = -1, -2

    N, C = utils.check_shapes(
        bg_color, (..., C),
        gp_means, (..., N, 3),
        gp_rots, (..., N, 4),
        gp_scales, (..., N, 3),
        gp_opacities, (..., N, 1),
    )

    assert gp_shs is not None or gp_colors is not None

    if gp_shs is not None:
        utils.check_shapes(gp_shs, (..., N, (sh_degree + 1)**2, C))

    if gp_colors is not None:
        utils.check_shapes(gp_colors, (..., N, C))

    batch_shape = utils.broadcast_shapes(
        camera_transform.shape,
        bg_color.shape[:-1],
        gp_means.shape[:-2],
        gp_rots.shape[:-2],
        gp_scales.shape[:-2],
        gp_opacities.shape[:-2],
        utils.try_get_batch_shape(gp_shs, -3),
        utils.try_get_batch_shape(gp_colors, -2),
    )

    # ---

    dd = (utils.CUDA_DEVICE, torch.float32)

    world_view_mat = world_view_mat.to(*dd).expand(batch_shape + (4, 4))

    world_ndc_mat = world_ndc_mat.to(*dd).expand(batch_shape + (4, 4))

    camera_pos = camera_transform.pos.to(*dd).expand(batch_shape + (3,))
    # [..., 3]

    bg_color = bg_color.to(*dd).expand(batch_shape + (C,))
    gp_means = gp_means.to(*dd).expand(batch_shape + (N, 3))
    gp_rots = gp_rots.to(*dd).expand(batch_shape + (N, 4))
    gp_scales = gp_scales.to(*dd).expand(batch_shape + (N, 3))
    gp_opacities = gp_opacities.to(*dd).expand(batch_shape + (N, 1))

    if gp_shs is not None:
        gp_shs = gp_shs.to(*dd).expand(batch_shape + (N, C))

    if gp_colors is not None:
        gp_colors = gp_colors.to(*dd).expand(batch_shape + (N, C))

    # ---

    colors = torch.empty(
        batch_shape + (C, camera_config.img_h, camera_config.img_w),
        dtype=utils.FLOAT, device=device)

    for batch_idx in utils.get_idxes(batch_shape):
        renderer_settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
            image_height=camera_config.img_h,  # image height: int
            image_width=camera_config.img_w,  # image width: int

            tanfovx=camera_config.foc_l,  # float
            tanfovy=camera_config.foc_u,  # float

            bg=bg_color[batch_idx],  # torch.Tensor [3]
            scale_modifier=1.0,  # float

            viewmatrix=world_view_mat[batch_idx].transpose(-1, -2),
            # torch.Tensor[4, 4]

            projmatrix=world_ndc_mat[batch_idx].transpose(-1, -2),
            # torch.Tensor[4, 4]

            sh_degree=sh_degree,

            campos=camera_pos[batch_idx],
            # [3]

            prefiltered=False,
            debug=False,
        )

        color, radii = diff_gaussian_rasterization.rasterize_gaussians(
            means3D=gp_means[batch_idx],
            means2D=torch.Tensor([]),

            sh=torch.Tensor([]) if gp_shs is None else gp_shs[batch_idx],

            colors_precomp=torch.Tensor([]) if gp_colors is None else
            gp_colors[batch_idx],

            opacities=gp_opacities[batch_idx],
            scales=gp_scales[batch_idx],
            rotations=gp_rots[batch_idx],
            cov3Ds_precomp=torch.Tensor([]),
            raster_settings=renderer_settings,
        )

        # color[C, H, W]

        colors[batch_idx] = color

    return RenderGaussianResult(
        colors=colors,
        radii=radii,
    )


@beartype
def query_gaussian(
    gp_means: torch.Tensor,  # [N, 3]
    gp_rots: torch.Tensor,  # [N, 4] quaternion
    gp_scales: torch.Tensor,  # [N, 3]
    gp_colors: torch.Tensor,  # [N, C]
    gp_opacities: torch.Tensor,  # [..., N, 1]

    points: torch.Tensor,  # [..., 3]
):
    device = utils.check_devices(
        gp_means,
        gp_rots,
        gp_scales,
        gp_colors,
        gp_opacities,
        points,
    )

    N, C = -1, -2

    N, C = utils.check_shapes(
        gp_means, (N, 3),
        gp_rots, (N, 4),
        gp_scales, (N, 3),
        gp_colors, (N, C),
        gp_opacities, (N, 1),
        points, (..., 3),
    )

    gp_rot_mats = utils.quaternion_to_rot_mat(
        gp_rots,
        order="WXYZ",
        out_shape=(3, 3),
    )  # [N, 3, 3]

    gp_scale_mats = torch.zeros(
        (N, 3, 3), dtype=gp_scales.dtype, device=device)

    gp_scale_mats[:, 0, 0] = gp_scales[:, 0]
    gp_scale_mats[:, 1, 1] = gp_scales[:, 1]
    gp_scale_mats[:, 2, 2] = gp_scales[:, 2]

    gp_rs = gp_rot_mats @ gp_scale_mats
    # [N, 3, 3]

    inv_cov = (gp_rs @ gp_rs.transpose(-1, -2)).inverse()
    # [N, 3, 3]

    rel_points = (points.unsqueeze(-2) - gp_means).unsqueeze(-1)
    # [..., N, 3, 1]

    k = torch.exp(-0.5 * (
        rel_points.transpose(-1, -2) @ inv_cov @ rel_points)) \
        .squeeze(-1).squeeze(-1)
    # [..., N, 1, 1] -> [..., N]

    ret = torch.einsum(
        "...i,ic->...c",
        k,  # [..., N]
        gp_colors * gp_opacities,  # [N, C]
    )  # [..., C]

    return ret
