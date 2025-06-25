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
def trans_to_cov3d(
    trans: torch.Tensor
) -> torch.Tensor:
    D = utils.check_shapes(trans, (..., -1, -1))
    return trans @ trans.transpose(-2, -1)


@beartype
@dataclasses.dataclass
class RenderGaussianResult:
    color: torch.Tensor  # [..., C, H, W]
    alpha: typing.Optional[torch.Tensor]  # [..., H, W]


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

    calc_alpha: bool,
):
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

    # ---

    foc_ud = camera_config.foc_u + camera_config.foc_d
    foc_lr = camera_config.foc_l + camera_config.foc_r

    ext_foc_h = max(camera_config.foc_u, camera_config.foc_d)
    ext_foc_w = max(camera_config.foc_l, camera_config.foc_r)

    ext_img_u_diff = round(camera_config.img_h * (
        (ext_foc_h - camera_config.foc_u) / foc_ud))

    ext_img_d_diff = round(camera_config.img_h * (
        (ext_foc_h - camera_config.foc_d) / foc_ud))

    ext_img_l_diff = round(camera_config.img_w * (
        (ext_foc_w - camera_config.foc_l) / foc_lr))

    ext_img_r_diff = round(camera_config.img_w * (
        (ext_foc_w - camera_config.foc_r) / foc_lr))

    ext_camera_config = camera_utils.CameraConfig(
        proj_type=camera_config.proj_type,

        foc_u=ext_foc_h,
        foc_d=ext_foc_h,

        foc_l=ext_foc_w,
        foc_r=ext_foc_w,

        depth_near=camera_config.depth_near,
        depth_far=camera_config.depth_far,

        img_h=camera_config.img_h + ext_img_u_diff + ext_img_d_diff,
        img_w=camera_config.img_w + ext_img_l_diff + ext_img_r_diff,
    )

    cur_camera_view_transform = camera_view_transform.to(
        camera_transform.device)

    cur_camera_ndc_transform = camera_ndc_transform.to(
        camera_transform.device)

    world_view_mat = camera_transform.get_trans_to(cur_camera_view_transform)
    # world -> view [..., 4, 4]

    view_ndc_mat = camera_utils.make_proj_mat_with_config(
        camera_config=ext_camera_config,

        camera_transform=cur_camera_view_transform,

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
    )  # view -> ndc [4, 4]

    world_ndc_mat = view_ndc_mat @ world_view_mat
    # [..., 4, 4]

    # ---

    assert gp_sh is not None or gp_color is not None

    if gp_cov3d_u is None:
        if gp_cov3d is None:
            assert gp_scale is not None and gp_rot_q is not None

            gp_scale_mat = utils.make_diag(gp_scale)
            # [..., N, 3, 3]

            gp_rot_mat = utils.quaternion_to_rot_mat(
                gp_rot_q, order="WXYZ", out_shape=(3, 3))
            # [..., N, 3, 3]

            gp_cov3d = trans_to_cov3d(utils.mat_mul(gp_rot_mat, gp_scale_mat))

        gp_cov3d_u = torch.stack([
            gp_cov3d[..., 0, 0],
            gp_cov3d[..., 0, 1],
            gp_cov3d[..., 0, 2],
            gp_cov3d[..., 1, 1],
            gp_cov3d[..., 1, 2],
            gp_cov3d[..., 2, 2],
        ], -1)
        # [..., N, 6]

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
        utils.try_get_batch_shape(gp_sh, 3),
        utils.try_get_batch_shape(gp_color, 2),
    )

    # ---

    world_view_mat = world_view_mat.expand(*batch_shape, 4, 4)

    world_ndc_mat = world_ndc_mat.expand(*batch_shape, 4, 4)

    camera_pos = camera_transform.pos.expand(*batch_shape, 3)

    bg_color = bg_color.expand(*batch_shape, C)
    gp_mean = gp_mean.expand(*batch_shape, N, 3)
    gp_cov3d_u = gp_cov3d_u.expand(*batch_shape, N, 6)
    gp_opacity = gp_opacity.expand(*batch_shape, N)

    gp_sh = utils.try_expand(gp_sh, (*batch_shape, N, C))
    gp_color = utils.try_expand(gp_color, (*batch_shape, N, C))

    color: list[torch.Tensor] = list()
    alpha: typing.Optional[list[torch.Tensor]] = list() if calc_alpha else None

    def preprocess(x):
        return x.to(utils.CUDA_DEVICE, torch.float32).contiguous()

    if calc_alpha:
        black_bg = torch.zeros(
            (3,), dtype=torch.float32, device=utils.CUDA_DEVICE)

        white_gp_color = torch.ones(
            (N, 3), dtype=torch.float32, device=utils.CUDA_DEVICE)

    for flatten_batch_idx, batch_idx in utils.get_batch_idxes(batch_shape):
        renderer_settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
            image_height=ext_camera_config.img_h,  # image height: int
            image_width=ext_camera_config.img_w,  # image width: int

            tanfovx=ext_camera_config.foc_l,  # float
            tanfovy=ext_camera_config.foc_u,  # float

            bg=preprocess(bg_color[batch_idx]),  # torch.Tensor [3]
            scale_modifier=1.0,  # float

            viewmatrix=preprocess(world_view_mat[batch_idx].T),
            # torch.Tensor[4, 4]

            projmatrix=preprocess(world_ndc_mat[batch_idx].T),
            # torch.Tensor[4, 4]

            sh_degree=sh_degree,

            campos=preprocess(camera_pos[batch_idx]),
            # [3]

            prefiltered=False,
            debug=False,
        )

        cur_color, cur_radii = diff_gaussian_rasterization.rasterize_gaussians(
            means3D=preprocess(gp_mean[batch_idx]),

            means2D=torch.Tensor([]),

            sh=torch.Tensor([]) if gp_sh is None else
            preprocess(gp_sh[batch_idx]),

            colors_precomp=torch.Tensor([]) if gp_color is None else
            preprocess(gp_color[batch_idx]),

            opacities=preprocess(gp_opacity[*batch_idx, :, None]),

            scales=torch.Tensor([]),
            rotations=torch.Tensor([]),
            cov3Ds_precomp=preprocess(gp_cov3d_u[batch_idx]),

            raster_settings=renderer_settings,
        )
        # cur_color[C, H, W]

        color.append(cur_color[
            :,
            ext_img_u_diff:(ext_camera_config.img_h - ext_img_d_diff),
            ext_img_l_diff:(ext_camera_config.img_w - ext_img_r_diff),
        ])

        if not calc_alpha:
            continue

        renderer_settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
            image_height=ext_camera_config.img_h,  # image height: int
            image_width=ext_camera_config.img_w,  # image width: int

            tanfovx=ext_camera_config.foc_l,  # float
            tanfovy=ext_camera_config.foc_u,  # float

            bg=black_bg,  # torch.Tensor [3]
            scale_modifier=1.0,  # float

            viewmatrix=preprocess(world_view_mat[batch_idx].T),
            # torch.Tensor[4, 4]

            projmatrix=preprocess(world_ndc_mat[batch_idx].T),
            # torch.Tensor[4, 4]

            sh_degree=sh_degree,

            campos=preprocess(camera_pos[batch_idx]),
            # [3]

            prefiltered=False,
            debug=False,
        )

        cur_alpha, cur_radii = diff_gaussian_rasterization.rasterize_gaussians(
            means3D=preprocess(gp_mean[batch_idx]),

            means2D=torch.Tensor([]),

            sh=torch.Tensor([]),

            colors_precomp=white_gp_color,

            opacities=preprocess(gp_opacity[*batch_idx, :, None]),

            scales=torch.Tensor([]),
            rotations=torch.Tensor([]),
            cov3Ds_precomp=preprocess(gp_cov3d_u[batch_idx]),

            raster_settings=renderer_settings,
        )
        # cur_alpha[3, H, W]

        alpha.append(cur_alpha[
            :1,
            ext_img_u_diff:(ext_camera_config.img_h - ext_img_d_diff),
            ext_img_l_diff:(ext_camera_config.img_w - ext_img_r_diff),
        ])

    stacked_color = torch.stack(color, 0).view(
        *batch_shape, C, camera_config.img_h, camera_config.img_w)

    stacked_alpha = torch.stack(alpha, 0).view(
        *batch_shape, 1, camera_config.img_h, camera_config.img_w) \
        if calc_alpha else None

    return RenderGaussianResult(
        color=stacked_color,
        alpha=stacked_alpha,
    )
