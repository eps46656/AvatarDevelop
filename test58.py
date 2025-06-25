import dataclasses
import itertools
import math
import typing

import beartype
import cv2
import einops
import numpy as np
import torch
import torchvision
import tqdm
from beartype import beartype

from . import (camera_utils, config, mesh_utils, rendering_utils, smplx_utils,
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
        model_transform, 4.0)[0]

    blending_param = smplx_utils.BlendingParam(
        global_transl=model_transform.vec_u * 0.3,
    )

    model: smplx_utils.Model = model_blender(blending_param)
    # [V, 3]

    joint_T = model.joint_T
    # [J, 4, 4]

    joint_pos = joint_T[:, :, 3]
    # [J, 4]

    H = 1080
    W = 1080

    proj_mat = camera_utils.make_proj_mat_with_config(
        camera_config=camera_config,
        camera_transform=camera_transform,
        proj_config=camera_utils.ProjConfig(
            camera_proj_transform=transform_utils.ObjectTransform.from_matching
            ("DRF"),
            delta_u=0.0,
            delta_d=H * 1.0,
            delta_l=0.0,
            delta_r=W * 1.0,
            delta_f=+1.0 / utils.DEPTH_FAR,
            delta_b=-1.0 / utils.DEPTH_NEAR,
        ),
    )
    # [4, 4]

    img_joint_pos = utils.do_homo(proj_mat, joint_pos)[:, :2]
    # [J, 2]

    print(f"{img_joint_pos=}")

    img = np.full((H, W, 3), 255, dtype=np.uint8)

    def draw(beg, end, color):
        p1 = (round(float(beg[1])), round(float(beg[0])))
        p2 = (round(float(end[1])), round(float(end[0])))

        cv2.line(img, p1, p2, color=color, thickness=4)

    SMPL_BONES = [
        (0, 1), (1, 4), (4, 7), (7, 10),       # left leg
        (0, 2), (2, 5), (5, 8), (8, 11),       # right leg
        (0, 3), (3, 6), (6, 9), (9, 12),       # spine to neck
        (12, 15),                              # neck to head
        (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),  # left arm
        (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),  # right arm
    ]

    SMPL_BONE_COLORS = [
        (255, 0, 0),   # red      left leg
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (0, 0, 255),   # blue     right leg
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 255),
        (0, 255, 0),   # green    spine
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0),
        (255, 128, 0),  # orange   left arm
        (255, 128, 0),
        (255, 128, 0),
        (255, 128, 0),
        (255, 128, 0),
        (0, 128, 255),  # light blue right arm
        (0, 128, 255),
        (0, 128, 255),
        (0, 128, 255),
        (0, 128, 255),
    ]

    for bone, color in zip(SMPL_BONES, SMPL_BONE_COLORS):
        beg = img_joint_pos[bone[0]]
        end = img_joint_pos[bone[1]]

        draw(beg, end, color)

    vision_utils.write_image(
        config.DIR /
        f"_images/canonical_image.png",
        utils.rct(rendering_utils.make_light_map(
            camera_config=camera_config,
            camera_transform=camera_transform,
            mesh_data=mesh_utils.MeshData(
                mesh_graph=model.mesh_graph,
                vert_pos=model.vert_pos.to(DEVICE)
            ),
            pix_to_face=None,
            bary_coord=None,
        ) * 255, dtype=torch.uint8),
    )

    vision_utils.write_image(
        config.DIR /
        f"_images/canonical_skeleton.png",
        einops.rearrange(torch.from_numpy(img), "h w c -> c h w"),
    )


if __name__ == "__main__":
    main3()
