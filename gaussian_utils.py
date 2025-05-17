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
def get_sh_degree(sh_features_cnt: int) -> int:
    sh_degree = int(sh_features_cnt**0.5 - 1)
    assert (sh_degree + 1)**2 == sh_features_cnt
    return sh_degree


@beartype
@dataclasses.dataclass
class RenderGaussianResult:
    color: torch.Tensor  # [..., C, H, W]
    radii: torch.Tensor


@beartype
def render_gaussian(
    *,
    camera_config: camera_utils.CameraConfig,

    camera_transform: transform_utils.ObjectTransform,
    # (camera <-> world)[...]

    bg_color: torch.Tensor,  # [..., C]

    gp_mean: torch.Tensor,  # [..., N, 3]

    gp_rot_q: typing.Optional[torch.Tensor] = None,  # [..., N, 4] quaternion
    gp_scale: typing.Optional[torch.Tensor] = None,  # [..., N, 3]
    gp_cov3d: typing.Optional[torch.Tensor] = None,  # [..., N, 3, 3]
    gp_cov3d_u: typing.Optional[torch.Tensor] = None,  # [..., N, 6]

    gp_sh: typing.Optional[torch.Tensor] = None,
    # [..., N, (sh_degress + 1)**2, C]

    gp_color: typing.Optional[torch.Tensor] = None,  # [..., N, C]

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

    N, C = -1, -2

    N, C = utils.check_shapes(
        bg_color, (..., C),
        gp_mean, (..., N, 3),

        gp_rot_q, (..., N, 4),
        gp_scale, (..., N, 3),
        gp_cov3d, (..., N, 3, 3),
        gp_cov3d_u, (..., N, 6),

        gp_opacity, (..., N),
    )

    assert gp_sh is not None or gp_color is not None

    if gp_cov3d_u is None:
        if gp_cov3d is None:
            assert gp_scale is not None and gp_rot_q is not None

            gp_sq_scale_mat = utils.make_diag(gp_scale).square()
            # [..., N, 3, 3]

            gp_rot_mat = utils.quaternion_to_rot_mat(
                gp_rot_q, order="WXYZ", out_shape=(3, 3))
            # [..., N, 3, 3]

            gp_cov3d = utils.mat_mul(
                gp_rot_mat, gp_sq_scale_mat, gp_rot_mat.transpose(-2, -1))
            # [..., N, 3, 3]

        gp_cov3d_u = utils.empty_like(
            gp_cov3d, shape=(*gp_cov3d.shape[:-2], 6))

        gp_cov3d_u[..., 0] = gp_cov3d[..., 0, 0]
        gp_cov3d_u[..., 1] = gp_cov3d[..., 0, 1]
        gp_cov3d_u[..., 2] = gp_cov3d[..., 0, 2]
        gp_cov3d_u[..., 3] = gp_cov3d[..., 1, 1]
        gp_cov3d_u[..., 4] = gp_cov3d[..., 1, 2]
        gp_cov3d_u[..., 5] = gp_cov3d[..., 2, 2]

    FT = utils.check_shapes(
        gp_sh, (..., N, -1, C),
        gp_color, (..., N, C),
    )  # (sh_degree + 1)**2

    if gp_color is None:
        sh_degree = get_sh_degree(FT)
    else:
        sh_degree = 0

    assert 0 <= sh_degree <= 2, f"{sh_degree=}"

    batch_shape = utils.broadcast_shapes(
        camera_transform.shape,
        bg_color.shape[:-1],
        gp_mean.shape[:-2],

        gp_cov3d_u.shape[:-2],

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
    gp_cov3d_u = gp_cov3d_u.to(*dd).expand(*batch_shape, N, 6)
    gp_opacity = gp_opacity.to(*dd).expand(*batch_shape, N)

    if gp_sh is not None:
        gp_sh = gp_sh.to(*dd).expand(*batch_shape, N, C)

    if gp_color is not None:
        gp_color = gp_color.to(*dd).expand(*batch_shape, N, C)

    # ---

    color = torch.empty(
        (*batch_shape, C, camera_config.img_h, camera_config.img_w),
        dtype=gp_color.dtype, device=device)

    radii = torch.empty(
        (*batch_shape, camera_config.img_h, camera_config.img_w),
        dtype=torch.int32, device=device,
    )

    for batch_idx in utils.get_batch_idxes(batch_shape):
        renderer_settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
            image_height=camera_config.img_h,  # image height: int
            image_width=camera_config.img_w,  # image width: int

            tanfovx=camera_config.foc_l,  # float
            tanfovy=camera_config.foc_u,  # float

            bg=bg_color[batch_idx],  # torch.Tensor [3]
            scale_modifier=1.0,  # float

            viewmatrix=world_view_mat[batch_idx].transpose(-2, -1),
            # torch.Tensor[4, 4]

            projmatrix=world_ndc_mat[batch_idx].transpose(-2, -1),
            # torch.Tensor[4, 4]

            sh_degree=sh_degree,

            campos=camera_pos[batch_idx],
            # [3]

            prefiltered=False,
            debug=False,
        )

        cur_color, cur_radii = diff_gaussian_rasterization.rasterize_gaussians(
            means3D=gp_mean[batch_idx].contiguous(),

            means2D=torch.Tensor([]),

            sh=torch.Tensor([]) if gp_sh is None else
            gp_sh[batch_idx].contiguous(),

            colors_precomp=torch.Tensor([]) if gp_color is None else
            gp_color[batch_idx].contiguous(),

            opacities=gp_opacity[*batch_idx, :, None].contiguous(),

            scales=torch.Tensor([]),
            rotations=torch.Tensor([]),
            cov3Ds_precomp=gp_cov3d_u[batch_idx].contiguous(),

            raster_settings=renderer_settings,
        )
        # cur_color[C, H, W]

        color[batch_idx] = cur_color

    return RenderGaussianResult(
        color=color,
        radii=radii,
    )
