import copy
import itertools
import math
import pathlib
import time
import typing

import einops
import torch
import tqdm
from beartype import beartype

from . import (avatar_utils, camera_utils, config, gom_avatar_utils,
               people_snapshot_utils, rendering_utils, sdf_utils, smplx_utils,
               training_utils, transform_utils, utils, vision_utils)

DTYPE = torch.float64
DEVICE = torch.device("cuda")

SDF_MODULE_DIR = config.DIR / "sdf_module_2025_0522_1"


ALPHA_SIGNED_DIST = 1.0
ALPHA_EIKONAL = 1.0

BATCH_SIZE = 64
BATCHES_CNT = 32


LR = 1e-3

SUBJECT_NAME = "female-3-casual"


def read_subject(model_data: typing.Optional[smplx_utils.ModelData] = None):
    if model_data is None:
        model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_NEUTRAL_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=utils.FLOAT,
            device=utils.CPU_DEVICE,
        )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME, model_data, utils.CPU_DEVICE)

    return subject_data


def load_trainer():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_NEUTRAL_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=DTYPE,
        device=DEVICE,
    )

    # ---

    dataset = sdf_utils.Dataset(
        mesh_graph=model_data.mesh_graph,
        vert_pos=model_data.vert_pos,

        std=50e-3,

        shape=torch.Size((BATCH_SIZE * BATCHES_CNT,)),
    )

    # ---

    module = sdf_utils.Module(
        range_min=(-2.0, -2.0, -2.0),
        range_max=(+2.0, +2.0, +2.0),
        dtype=DTYPE,
        device=DEVICE,
    ).train()

    # ---

    trainer_core = sdf_utils.TrainerCore(
        config=sdf_utils.TrainerCoreConfig(
            proj_dir=SDF_MODULE_DIR,
            device=DEVICE,

            batch_size=BATCH_SIZE,

            lr=LR,
            betas=(0.9, 0.99),
            gamma=0.1 ** (1 / 64),

            alpha_signed_dist=ALPHA_SIGNED_DIST,
            alpha_eikonal=ALPHA_EIKONAL,
        ),
        module=module,
        dataset=dataset,
    )

    # ---

    trainer = training_utils.Trainer(
        proj_dir=SDF_MODULE_DIR,
        trainer_core=trainer_core,
    )

    return trainer


@beartype
def F(
    avatar_blender: list[avatar_utils.AvatarBlender],
    camera_config: camera_utils.CameraConfig,
    camera_transform: transform_utils.ObjectTransform,
    blending_param: list[typing.Any],
    texture: list[torch.Tensor],
) -> tuple[
    torch.Tensor,  # dist[H, W]
    torch.Tensor,  # tex[H, W, C]
]:
    N = utils.all_same(
        len(avatar_blender),
        len(blending_param),
        len(texture),
    )

    H, W = camera_config.img_h, camera_config.img_w

    acc_r_dist: torch.Tensor = None
    acc_tex: torch.Tensor = None

    for i in range(N):
        cur_avatar_blender = avatar_blender[i]
        cur_blending_param = blending_param[i]
        cur_texture = texture[i]

        avatar_model: avatar_utils.AvatarModel = \
            cur_avatar_blender(cur_blending_param)

        cur_rasterize_mesh_result = rendering_utils.rasterize_mesh(
            vert_pos=avatar_model.vert_pos,
            faces=avatar_model.mesh_graph.f_to_vvv,

            camera_config=camera_config,
            camera_transform=camera_transform,

            faces_per_pixel=1,

            cull_backface=True,
        )

        cur_pix_to_face = cur_rasterize_mesh_result.pix_to_face.view(H, W)
        cur_bary_coord = cur_rasterize_mesh_result.bary_coord.view(H, W, 3)
        cur_dist = cur_rasterize_mesh_result.dist.view(H, W)

        cur_tex_coord = rendering_utils.calc_tex_coord(
            pix_to_face=cur_pix_to_face,
            bary_coord=cur_bary_coord,
            tex_f_to_vvv=avatar_model.tex_mesh_graph.f_to_vvv,
            tex_vert_pos=avatar_model.tex_vert_pos,
        )

        cur_tex = rendering_utils.sample_texture(
            texture=cur_texture,
            tex_coord=cur_tex_coord,
        )

        cur_tex = torch.where(
            (cur_pix_to_face == -1)[..., None],
            1.0,
            cur_tex,
        )

        cur_r_dist = 1 / cur_dist

        if acc_r_dist is None:
            acc_r_dist = cur_r_dist
            acc_tex = cur_tex
        else:
            p = acc_r_dist < cur_r_dist
            # [H, W]

            acc_r_dist = torch.where(
                p,
                cur_r_dist,
                acc_r_dist,
            )
            acc_tex = torch.where(
                p[..., None],
                cur_tex,
                acc_tex,
            )

    return 1 / acc_r_dist, acc_tex


def main1():
    model_data = [
        smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=DTYPE,
            device=DEVICE,
        ),  # naked model


    ]

    tex_path = [
        config.DIR / "tex_avatar_2025_0518_4" / "tex_1747581876.png",
        # naked model texture
    ]

    N = utils.all_same(len(model_data), tex_path)

    model_blender = [
        smplx_utils.ModelBlender(
            model_builder=smplx_utils.StaticModelBuilder(
                model_data=cur_model_data),
        )
        for cur_model_data in model_data
    ]

    tex = [
        (vision_utils.read_image(cur_tex_path) / 255.0).to(DEVICE, DTYPE)
        for cur_tex_path in tex_path
    ]

    subject_data = read_subject()

    T, C, H, W = subject_data.video.shape

    video_writer = vision_utils.VideoWriter(
        path=config.DIR / f"rgb_{utils.timestamp_sec()}.avi",
        height=H,
        width=W,
        color_type=vision_utils.ColorType.RGB,
        fps=subject_data.fps,
    )

    for t in range(T):
        cur_blending_param = subject_data.blending_param[t]

        cur_img = F(
            avatar_blender=model_blender,
            camera_config=subject_data.camera_config,
            camera_transform=subject_data.camera_transform,
            blending_param=[cur_blending_param] * N,
            texture=tex,
        )

        video_writer.write(utils.rct(cur_img * 255, torch.uint8))

    video_writer.close()


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(False, False):
        main1()

    print("ok")
