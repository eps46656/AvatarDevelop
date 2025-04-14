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

    gp_mean: torch.Tensor,  # [..., N, 3]
    gp_rot_q: torch.Tensor,  # [..., N, 4] quaternion
    gp_scale: torch.Tensor,  # [..., N, 3]

    gp_sh: typing.Optional[torch.Tensor],  # [..., N, (sh_degress + 1)**2, C]
    gp_color: typing.Optional[torch.Tensor],  # [..., N, C]

    gp_opacity: torch.Tensor,  # [..., N]

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
        gp_mean, (..., N, 3),
        gp_rot_q, (..., N, 4),
        gp_scale, (..., N, 3),
        gp_opacity, (..., N),
    )

    assert gp_sh is not None or gp_color is not None

    utils.check_shapes(
        gp_sh, (..., N, (sh_degree + 1)**2, C),
        gp_color, (..., N, C),
    )

    batch_shape = utils.broadcast_shapes(
        camera_transform.shape,
        bg_color.shape[:-1],
        gp_mean.shape[:-2],
        gp_rot_q.shape[:-2],
        gp_scale.shape[:-2],
        gp_opacity.shape[:-1],
        utils.try_get_batch_shape(gp_sh, -3),
        utils.try_get_batch_shape(gp_color, -2),
    )

    # ---

    dd = (utils.CUDA_DEVICE, torch.float32)

    world_view_mat = world_view_mat.to(*dd).expand(*batch_shape, 4, 4)

    world_ndc_mat = world_ndc_mat.to(*dd).expand(*batch_shape, 4, 4)

    camera_pos = camera_transform.pos.to(*dd).expand(*batch_shape, 3)
    # [..., 3]

    bg_color = bg_color.to(*dd).expand(*batch_shape, C)
    gp_mean = gp_mean.to(*dd).expand(*batch_shape, N, 3)
    gp_rot_q = gp_rot_q.to(*dd).expand(*batch_shape, N, 4)
    gp_scale = gp_scale.to(*dd).expand(*batch_shape, N, 3)
    gp_opacity = gp_opacity.to(*dd).expand(*batch_shape, N)

    if gp_sh is not None:
        gp_sh = gp_sh.to(*dd).expand(*batch_shape, N, C)

    if gp_color is not None:
        gp_color = gp_color.to(*dd).expand(*batch_shape, N, C)

    # ---

    colors = torch.empty(
        batch_shape + (C, camera_config.img_h, camera_config.img_w),
        dtype=utils.FLOAT, device=device)

    for batch_idx in utils.get_batch_idxes(batch_shape):
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
            means3D=gp_mean[batch_idx],
            means2D=torch.Tensor([]),

            sh=torch.Tensor([]) if gp_sh is None else gp_sh[batch_idx],

            colors_precomp=torch.Tensor([]) if gp_color is None else
            gp_color[batch_idx],

            opacities=gp_opacity[batch_idx].unsqueeze(-1),
            scales=gp_scale[batch_idx],
            rotations=gp_rot_q[batch_idx],
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
    gp_mean: torch.Tensor,  # [N, 3]
    gp_rot_q: torch.Tensor,  # [N, 4] quaternion
    gp_scale: torch.Tensor,  # [N, 3]
    gp_color: torch.Tensor,  # [N, C]
    gp_opacity: torch.Tensor,  # [..., N, 1]

    points: torch.Tensor,  # [..., 3]
):
    device = utils.check_devices(
        gp_mean,
        gp_rot_q,
        gp_scale,
        gp_color,
        gp_opacity,
        points,
    )

    N, C = -1, -2

    N, C = utils.check_shapes(
        gp_mean, (N, 3),
        gp_rot_q, (N, 4),
        gp_scale, (N, 3),
        gp_color, (N, C),
        gp_opacity, (N, 1),
        points, (..., 3),
    )

    gp_rot_mats = utils.quaternion_to_rot_mat(
        gp_rot_q,
        order="WXYZ",
        out_shape=(3, 3),
    )  # [N, 3, 3]

    gp_scale_mats = torch.zeros(
        (N, 3, 3), dtype=gp_scale.dtype, device=device)

    gp_scale_mats[:, 0, 0] = gp_scale[:, 0]
    gp_scale_mats[:, 1, 1] = gp_scale[:, 1]
    gp_scale_mats[:, 2, 2] = gp_scale[:, 2]

    gp_rs = gp_rot_mats @ gp_scale_mats
    # [N, 3, 3]

    inv_cov = (gp_rs @ gp_rs.transpose(-1, -2)).inverse()
    # [N, 3, 3]

    rel_points = (points.unsqueeze(-2) - gp_mean).unsqueeze(-1)
    # [..., N, 3, 1]

    k = (-0.5 * (rel_points.transpose(-1, -2) @ inv_cov @ rel_points)).exp() \
        .squeeze(-1).squeeze(-1)
    # [..., N, 1, 1] -> [..., N]

    ret = torch.einsum(
        "...i,ic->...c",
        k,  # [..., N]
        gp_color * gp_opacity,  # [N, C]
    )  # [..., C]

    return ret
