import dataclasses
import itertools
import math
import typing

import beartype
import einops
import torch
import torchvision
import tqdm
from beartype import beartype

from . import (camera_utils, config, rendering_utils, smplx_utils,
               transform_utils, utils, video_seg_utils, vision_utils)

SEG_DIR = config.DIR / "video_seg_2025_0517_1"

MESH_SEG_DIR = config.DIR / "mesh_seg_2025_0514_1"

DTYPE = torch.float64
DEVICE = utils.CUDA_DEVICE


def main1():
    model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_utils.StaticModelBuilder(
            model_data=smplx_utils.ModelData.from_origin_file(
                model_data_path=config.SMPL_NEUTRAL_MODEL_PATH,
                model_config=smplx_utils.smpl_model_config,
                dtype=DTYPE,
                device=DEVICE,
            ),
        ),
    )

    camera_config = camera_utils.CameraConfig.from_fov_diag(
        fov_diag=50.0 * utils.DEG,
        depth_near=utils.DEPTH_NEAR,
        depth_far=utils.DEPTH_FAR,
        img_h=1080,
        img_w=1080,
    )

    model_model_space_transform = smplx_utils.smpl_model_transform.to(
        DEVICE, DTYPE)
    # model <-> model space

    model_world_transform = transform_utils.ObjectTransform.from_matching(
        dirs="RFU", dtype=DTYPE, device=DEVICE)
    # model <-> world

    world_to_model_space = model_world_transform.get_trans_to(
        model_model_space_transform)
    # world -> model space

    radius = 4.0
    theta = 1.10 * 90.0 * utils.DEG
    phi = (1 / 8) * 360.0 * utils.DEG

    point = torch.tensor(utils.sph_to_cart(
        radius=radius,
        theta=theta,
        phi=phi,
    ), dtype=DTYPE, device=DEVICE)

    print(f"{point=}")

    f_vec = utils.vec_normed(-point)
    quasi_u_vec = torch.tensor([0, 0, 1], dtype=DTYPE, device=DEVICE)
    l_vec = utils.vec_normed(utils.vec_cross(quasi_u_vec, f_vec))
    u_vec = utils.vec_cross(f_vec, l_vec)

    print(f"{f_vec=}")
    print(f"{l_vec=}")
    print(f"{u_vec=}")

    camera_world_transform = transform_utils.ObjectTransform.from_matching(
        dirs="LUF",
        pos=point,
        vecs=(l_vec, u_vec, f_vec),
    )
    # camera <-> world

    camera_model_space_transform = camera_world_transform.collapse(
        world_to_model_space)
    # camera <-> model space

    blending_param = smplx_utils.BlendingParam(
        global_transl=model_model_space_transform.vec_u * 0.3,
    )

    model: smplx_utils.Model = model_blender(blending_param)
    # [V, 3]

    img = rendering_utils.make_normal_map(
        camera_config=camera_config,
        camera_transform=camera_model_space_transform,

        mesh_data=model.mesh_data,

        pix_to_face=None,
        bary_coord=None,
    )

    vision_utils.write_image(
        config.DIR / f"nor_{utils.timestamp_sec()}.png",
        utils.rct(einops.rearrange(img, "h w c -> c h w") * 255,
                  dtype=torch.uint8, device=utils.CPU_DEVICE),
    )


def main2():
    model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_utils.StaticModelBuilder(
            model_data=smplx_utils.ModelData.from_origin_file(
                model_data_path=config.SMPL_NEUTRAL_MODEL_PATH,
                model_config=smplx_utils.smpl_model_config,
                dtype=DTYPE,
                device=DEVICE,
            ),
        ),
    )

    model_transform = smplx_utils.smpl_model_transform.to(
        DEVICE, DTYPE)

    camera_config = rendering_utils.get_nice_camera_config(1080, 1080)

    camera_transform = rendering_utils.get_nice_camera_transform(
        model_transform, 4.0)

    blending_param = smplx_utils.BlendingParam(
        global_transl=model_transform.vec_u * 0.3,
    )

    model: smplx_utils.Model = model_blender(blending_param)
    # [V, 3]

    img = rendering_utils.make_normal_map(
        camera_config=camera_config,
        camera_transform=camera_transform,

        mesh_data=model.mesh_data,

        pix_to_face=None,
        bary_coord=None,
    )

    for i, cur_img in enumerate(img):
        vision_utils.write_image(
            config.DIR / f"nor_{utils.timestamp_sec()}-{i}.png",
            utils.rct(einops.rearrange(cur_img, "h w c -> c h w") * 255,
                      dtype=torch.uint8, device=utils.CPU_DEVICE),
        )


def main3():
    pass


if __name__ == "__main__":
    main2()
