import math
import pathlib
import typing

import torch
from beartype import beartype


from . import camera_utils, gaussian_utils, transform_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")

CUDA_DEVICE = torch.device("cuda")

camera_view_transform = transform_utils.ObjectTransform.from_matching(
    "RUB")
# camera <-> view


def main1():
    radius = 10.0
    theta = 60.0 * utils.DEG
    phi = (180.0 + 45.0) * utils.DEG

    camera_transform = camera_utils.make_view(
        origin=torch.tensor(utils.sph_to_cart(radius, theta, phi),
                            dtype=utils.FLOAT),
        aim=utils.ORIGIN,
        quasi_u_dir=utils.Z_AXIS,
    )  # camera <-> world

    camera_config = camera_utils.CameraConfig.from_fov_diag(
        fov_diag=90 * utils.DEG,
        depth_near=utils.DEPTH_NEAR,
        depth_far=utils.DEPTH_FAR,
        img_h=768,
        img_w=1280,
    )

    SH_DEG = 1

    N = 3

    # gp_means = torch.rand((N, 3), dtype=utils.FLOAT)
    gp_means = torch.tensor([
        # [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ], dtype=utils.FLOAT, device=DEVICE)

    gp_opacities = torch.ones(
        (N, 1), dtype=utils.FLOAT, device=DEVICE)

    gp_scales = torch.tensor([
        # [1.0, 1.0, 1.0],
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ], dtype=utils.FLOAT, device=DEVICE)

    # gp_rots = utils.RandQuaternion((N,), dtype=utils.FLOAT)
    gp_rots = torch.tensor([
        # [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=utils.FLOAT, device=DEVICE)

    gp_shs = torch.rand(
        (N, 3, (SH_DEG + 1)**2),
        dtype=utils.FLOAT, device=DEVICE)

    gp_colors = torch.rand(
        (N, 3),
        dtype=utils.FLOAT, device=DEVICE)

    print(f"{gp_means.shape=}")
    print(f"{gp_opacities.shape=}")
    print(f"{gp_scales.shape=}")
    print(f"{gp_rots.shape=}")
    print(f"{gp_shs.shape=}")
    print(f"{gp_colors.shape=}")

    result = gaussian_utils.render_gaussian(
        camera_transform=camera_transform,

        camera_config=camera_config,

        sh_degree=SH_DEG,

        bg_color=torch.tensor(
            [1.0, 1.0, 1.0], dtype=utils.FLOAT, device=DEVICE),

        gp_mean=gp_means,
        gp_opacity=gp_opacities,
        gp_scale=gp_scales,
        gp_rot_q=gp_rots,

        gp_sh=None,
        gp_color=gp_colors,

        device=CUDA_DEVICE,
    )

    img = result["color"]

    print(f"{type(img)=}")

    if isinstance(img, torch.Tensor):
        print(f"{img.shape=}")

    vision_utils.write_image(
        DIR / "out.png",
        img * 255,
    )


if __name__ == "__main__":
    main1()
