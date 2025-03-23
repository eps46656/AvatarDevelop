import dataclasses

import einops
import torch
from beartype import beartype

import diff_gaussian_rasterization

from . import camera_utils, transform_utils, utils

camera_view_transform = transform_utils.ObjectTransform.FromMatching("RDF")
# camera <-> view

camera_ndc_transform = transform_utils.ObjectTransform.FromMatching("RDF")
# camera <-> ndc


@beartype
@dataclasses.dataclass
class RenderGaussianResult:
    color: torch.Tensor  # [C, H, W]
    radii: torch.Tensor


@beartype
def RenderGaussian(
    camera_transform: transform_utils.ObjectTransform,
    # camera <-> world
    camera_config: camera_utils.CameraConfig,

    sh_degree: int,  # 0 ~ 2

    bg_color: torch.Tensor,  # [C]

    gp_means: torch.Tensor,  # [N, 3]
    gp_rots: torch.Tensor,  # [N, 4] quaternion
    gp_scales: torch.Tensor,  # [N, 3]

    gp_shs: torch.Tensor,  # [N, (sh_degress + 1)**2, C]
    gp_colors: torch.Tensor,  # [N, C]

    gp_opacities: torch.Tensor,  # [N, 1]

    device: torch.device,
):
    world_view_mat = camera_transform.GetTransTo(camera_view_transform)
    # world -> view [4, 4]

    view_ndc_mat = camera_utils.MakeProjMatWithConfig(
        camera_config=camera_config,

        camera_view_transform=camera_view_transform,

        proj_config=camera_utils.ProjConfig(
            camera_proj_transform=camera_ndc_transform,
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

    # ---

    assert 0 <= sh_degree <= 2

    # ---

    N, C = -1, -2

    N, C = utils.CheckShapes(
        bg_color, (C,),
        gp_means, (N, 3),
        gp_rots, (N, 4),
        gp_scales, (N, 3),
        gp_opacities, (N, 1),
    )

    assert gp_shs is not None or gp_colors is not None

    if gp_shs is None:
        gp_shs = torch.Tensor([])
    else:
        utils.CheckShapes(gp_shs, (N, (sh_degree + 1)**2), C)

    if gp_colors is None:
        gp_colors = torch.Tensor([])
    else:
        utils.CheckShapes(gp_colors, (N, C))

    # ---

    renderer_settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
        image_height=camera_config.img_h,  # image height: int
        image_width=camera_config.img_w,  # image width: int

        tanfovx=camera_config.foc_l,  # float
        tanfovy=camera_config.foc_u,  # float

        bg=bg_color.to(device=device),  # torch.Tensor [3]
        scale_modifier=1.0,  # float

        viewmatrix=world_view_mat.transpose(0, 1).to(device=device),
        # torch.Tensor[4, 4]

        projmatrix=(view_ndc_mat @ world_view_mat).transpose(0, 1)
        .to(device=device),
        # torch.Tensor[4, 4]

        sh_degree=sh_degree,

        campos=camera_transform.GetPos().to(device=device),
        # [3]

        prefiltered=True,
        debug=False,
    )

    color, radii = diff_gaussian_rasterization.rasterize_gaussians(
        means3D=gp_means.to(device=device),
        means2D=torch.Tensor([]),
        sh=gp_shs,
        colors_precomp=gp_colors.to(device=device),
        opacities=gp_opacities.to(device=device),
        scales=gp_scales.to(device=device),
        rotations=gp_rots.to(device=device),
        cov3Ds_precomp=torch.Tensor([]),
        raster_settings=renderer_settings,
    )

    # color[H, W, C]

    return RenderGaussianResult(
        color=einops.rearrange(color, f"h w c -> c h w"),
        radii=radii,
    )
