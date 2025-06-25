import os

import einops
import numpy as np
import torch
import tqdm
import cv2

from . import (avatar_utils, camera_utils, config, mesh_utils,
               people_snapshot_utils, rendering_utils, smplx_utils,
               transform_utils, utils, video_seg_utils, vision_utils)

DTYPE = torch.float64
DEVICE = utils.CPU_DEVICE

SUBJECT_NAME = "female-3-casual"
SUBJECT_SHORT_NAME = "f3c"

VIDEO_SEG_DIR = config.DIR / f"video_seg_f3c_2025_0619_1"


BARE_AVATAR_DIR = config.DIR / "bare_avatar_f3c_2025_0527_1"


def main1():
    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        None,
        DTYPE,
        DEVICE,
    )

    T, C, H, W = subject_data.video.shape

    obj_list = [
        "upper_garment",
        "lower_garment",
    ]

    obj_model_data_path = [
        config.DIR / "mesh_seg_f3c_2025_0624_1/obj_model_data_upper_garment_1750768751.pkl",

        config.DIR / "mesh_seg_f3c_2025_0624_1/obj_model_data_lower_garment_1750768753.pkl",
    ]

    texture = init_tex = vision_utils.read_image(
        config.DIR / "tex_avatar_f3c_2025_0624_1/tex_1750765250.png",
        "RGB"
    ).image.to(DEVICE, DTYPE) / 255.0

    obj_model_data = [
        smplx_utils.ModelData.from_state_dict(
            state_dict=utils.read_pickle(cur_obj_model_data_path),
            dtype=torch.float64,
            device=DEVICE,
        )

        for cur_obj_model_data_path in obj_model_data_path
    ]

    frame_idx = 30

    camera_config = subject_data.camera_config

    camera_transform = subject_data.camera_transform

    blending_param = subject_data.blending_param[frame_idx]

    for cur_obj_name, cur_obj_model_data in zip(obj_list, obj_model_data):
        smplx_model_builder = smplx_utils.StaticModelBuilder(
            model_data=cur_obj_model_data,
        )

        smplx_model_blender = smplx_utils.ModelBlender(
            model_builder=smplx_model_builder,
        )

        avatar_model = smplx_model_blender(blending_param)

        ret_img = rendering_utils.make_light_map(
            camera_config=camera_config,
            camera_transform=camera_transform,
            mesh_data=mesh_utils.MeshData(
                mesh_graph=avatar_model.mesh_graph,
                vert_pos=avatar_model.vert_pos.to(DEVICE)
            ),
            pix_to_face=None,
            bary_coord=None,
        )

        """
        mesh_ras_result = rendering_utils.rasterize_mesh(
            vert_pos=avatar_model.vert_pos.to(DEVICE),

            faces=avatar_model.mesh_graph.f_to_vvv.to(DEVICE),

            camera_config=camera_config,
            camera_transform=camera_transform,
            faces_per_pixel=1,
        )

        pix_to_face = mesh_ras_result.pix_to_face[..., 0]
        # [H, W]

        bary_coord = mesh_ras_result.bary_coord[..., 0, :]
        # [H, W, 3]

        tex_f_to_vvv = avatar_model.tex_mesh_graph.f_to_vvv.to(DEVICE)
        tex_vert_pos = avatar_model.tex_vert_pos.to(DEVICE)

        tex_coord = rendering_utils.calc_tex_coord(
            pix_to_face=pix_to_face,
            bary_coord=bary_coord,
            tex_f_to_vvv=tex_f_to_vvv,
            tex_vert_pos=tex_vert_pos,
        )
        # [..., H, W, 2]

        rendered_img = rendering_utils.sample_texture(
            texture=einops.rearrange(texture, "c h w -> h w c"),
            tex_coord=tex_coord,
            wrap_mode=rendering_utils.WrapMode.MIRROR,
            sampling_mode=rendering_utils.SamplingMode.LINEAR,
        ).to(torch.float64)
        # [H, W, C]

        hit_map = 0 <= pix_to_face
        # [H, W]

        rendered_img = rendered_img * hit_map[..., None]

        ret_img = torch.cat([
            rendered_img,  # [H, W, C]
            hit_map[..., None],
        ], dim=-1)
        # [H, W, C + 1]
        """

        vision_utils.write_image(
            config.DIR /
            f"_images/obj_image_{cur_obj_name}.png",
            utils.rct(ret_img * 255, dtype=torch.uint8),
        )


def main2():
    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        None,
        DTYPE,
        DEVICE,
    )

    T, C, H, W = subject_data.video.shape

    obj_list = [
        "upper_garment",
        "lower_garment",
    ]

    obj_model_data_path = [
        config.DIR / "mesh_seg_f3c_2025_0624_1/obj_model_data_upper_garment_1750768751.pkl",

        config.DIR / "mesh_seg_f3c_2025_0624_1/obj_model_data_lower_garment_1750768753.pkl",
    ]

    texture = init_tex = vision_utils.read_image(
        BARE_AVATAR_DIR / "tex_1748286865.png",
        "RGB"
    ).image.to(DEVICE, DTYPE) / 255.0

    frame_idx = 30

    camera_config = subject_data.camera_config

    camera_transform = subject_data.camera_transform

    blending_param = subject_data.blending_param[frame_idx]

    smplx_model_builder = smplx_utils.StaticModelBuilder(
        model_data=subject_data.model_data,
    )

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    avatar_model = smplx_model_blender(blending_param)

    mesh_ras_result = rendering_utils.rasterize_mesh(
        vert_pos=avatar_model.vert_pos.to(DEVICE),

        faces=avatar_model.mesh_graph.f_to_vvv.to(DEVICE),

        camera_config=camera_config,
        camera_transform=camera_transform,
        faces_per_pixel=1,
    )

    pix_to_face = mesh_ras_result.pix_to_face[..., 0]
    # [H, W]

    bary_coord = mesh_ras_result.bary_coord[..., 0, :]
    # [H, W, 3]

    tex_f_to_vvv = avatar_model.tex_mesh_graph.f_to_vvv.to(DEVICE)
    tex_vert_pos = avatar_model.tex_vert_pos.to(DEVICE)

    tex_coord = rendering_utils.calc_tex_coord(
        pix_to_face=pix_to_face,
        bary_coord=bary_coord,
        tex_f_to_vvv=tex_f_to_vvv,
        tex_vert_pos=tex_vert_pos,
    )
    # [..., H, W, 2]

    rendered_img = rendering_utils.sample_texture(
        texture=einops.rearrange(texture, "c h w -> h w c"),
        tex_coord=tex_coord,
        wrap_mode=rendering_utils.WrapMode.MIRROR,
        sampling_mode=rendering_utils.SamplingMode.LINEAR,
    ).to(torch.float64)
    # [H, W, C]

    hit_map = 0 <= pix_to_face
    # [H, W]

    rendered_img = rendered_img * hit_map[..., None]

    ret_img = torch.cat([
        rendered_img,  # [H, W, C]
        hit_map[..., None],
    ], dim=-1)
    # [H, W, C + 1]

    vision_utils.write_image(
        config.DIR /
        f"_images/bare_avatar.png",
        utils.rct(einops.rearrange(
            ret_img, "h w c -> c h w") * 255, dtype=torch.uint8),
    )


def main3():
    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        None,
        DTYPE,
        DEVICE,
    )

    T, C, H, W = subject_data.video.shape

    camera_config = subject_data.camera_config

    camera_transform = subject_data.camera_transform

    smplx_model_builder = smplx_utils.StaticModelBuilder(
        model_data=subject_data.model_data,
    )

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    frame_idx = 30

    blending_param = subject_data.blending_param[frame_idx]
    blending_param.body_shape = None
    blending_param.expr_shape = None
    blending_param.body_pose = None
    blending_param.jaw_pose = None
    blending_param.leye_pose = None
    blending_param.reye_pose = None
    blending_param.lhand_pose = None
    blending_param.rhand_pose = None

    avatar_model = smplx_model_blender(blending_param)

    ret_img = rendering_utils.make_light_map(
        camera_config=camera_config,
        camera_transform=camera_transform,
        mesh_data=mesh_utils.MeshData(
            mesh_graph=avatar_model.mesh_graph,
            vert_pos=avatar_model.vert_pos.to(DEVICE)
        ),
        pix_to_face=None,
        bary_coord=None,
    )

    vision_utils.write_image(
        config.DIR /
        f"_images/canonical_image.png",
        utils.rct(ret_img * 255, dtype=torch.uint8),
    )


def main4():
    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        None,
        DTYPE,
        DEVICE,
    )

    T, C, H, W = subject_data.video.shape

    obj_list = [
        "upper_garment",
        "lower_garment",
    ]

    obj_model_data_path = [
        config.DIR / "mesh_seg_f3c_2025_0624_1/obj_model_data_upper_garment_1750768751.pkl",

        config.DIR / "mesh_seg_f3c_2025_0624_1/obj_model_data_lower_garment_1750768753.pkl",
    ]

    texture = init_tex = vision_utils.read_image(
        BARE_AVATAR_DIR / "tex_1748286865.png",
        "RGB"
    ).image.to(DEVICE, DTYPE) / 255.0

    frame_idx = 30

    camera_config = subject_data.camera_config

    camera_transform = subject_data.camera_transform

    blending_param = subject_data.blending_param[frame_idx]

    smplx_model_builder = smplx_utils.StaticModelBuilder(
        model_data=subject_data.model_data,
    )

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    avatar_model: avatar_utils.AvatarModel = \
        smplx_model_blender(blending_param)

    joint_T = avatar_model.joint_T
    # [J, 4, 4]

    joint_pos = joint_T[:, :, 3]
    # [J, 4]

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
        f"_images/skeleton.png",
        einops.rearrange(torch.from_numpy(img), "h w c -> c h w"),
    )


if __name__ == "__main__":
    main4()
