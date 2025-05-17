import copy
import itertools
import pathlib

import torch
from beartype import beartype

from . import (camera_utils, config, gom_avatar_utils, people_snapshot_utils,
               smplx_utils, transform_utils, utils, vision_utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")


def main1():
    smpl_model_data_path_dict = {
        "male": config.SMPL_MALE_MODEL_PATH,
        "female": config.SMPL_FEMALE_MODEL_PATH,
        "neutral": config.SMPL_NEUTRAL_MODEL_PATH,
    }

    smplx_model_data_path_dict = {
        "male": config.SMPLX_MALE_MODEL_PATH,
        "female": config.SMPLX_FEMALE_MODEL_PATH,
        "neutral": config.SMPLX_NEUTRAL_MODEL_PATH,
    }

    body_shapes_cnt = 10
    expr_shapes_cnt = 0
    body_joints_cnt = 24
    jaw_joints_cnt = 0
    eye_joints_cnt = 0
    hand_joints_cnt = 0

    model_data_dict = {
        key: smplx_utils.Core.from_file(
            model_data_path=value,
            body_shapes_cnt=body_shapes_cnt,
            expr_shapes_cnt=expr_shapes_cnt,
            body_joints_cnt=body_joints_cnt,
            jaw_joints_cnt=jaw_joints_cnt,
            eye_joints_cnt=eye_joints_cnt,
            hand_joints_cnt=hand_joints_cnt,
            device=DEVICE,
        )
        for key, value in smpl_model_data_path_dict.items()
    }

    people_snapshot_dir = DIR / "people_snapshot_public"

    subject_name = "female-1-casual"

    subject_dir = people_snapshot_dir / subject_name

    subject_data = people_snapshot_utils.read_subject(
        subject_dir=subject_dir,
        model_data_dict=model_data_dict,
        device=DEVICE,
    )

    subject_data.video = vision_utils.normalize_image(subject_data.video)

    camera_config = subject_data.camera_config

    # ---

    model_data = subject_data.model_data

    model_data.vert_pos = torch.nn.Parameter(
        model_data.vert_pos)

    smplx_model_builder = smplx_utils.ModelBlender(
        model_data=model_data,
        device=DEVICE,
    )

    gom_avatar_module = gom_avatar_utils.Module(
        avatar_blender=smplx_model_builder,
        color_channels_cnt=3,
    ).train()

    T = subject_data.video.shape[0]

    frames = torch.empty(
        (T, 3,
         subject_data.camera_config.img_h, subject_data.camera_config.img_w)
    )

    optimizer = torch.optim.Adam(
        gom_avatar_module.parameters(),
        lr=1e-4,
    )

    for epoch_i in range(10):
        for frame_i in range(T):
            optimizer.zero_grad()

            print(f"{epoch_i=}\t\t{frame_i=}")

            result: gom_avatar_utils.ModuleForwardResult =\
                gom_avatar_module(
                    subject_data.camera_transform,
                    subject_data.camera_config,

                    subject_data.video[frame_i],

                    subject_data.mask[frame_i],

                    smplx_utils.BlendingParam(
                        body_shape=subject_data.blending_param.
                        body_shape,

                        global_transl=subject_data.blending_param.
                        global_transl[frame_i],

                        global_rot=subject_data.blending_param.
                        global_rot[frame_i],

                        body_pose=subject_data.blending_param.
                        body_pose[frame_i],
                    )
                )

            frames[frame_i] = result.gp_render_img.detach()

            mean_rgb_loss = result.img_diff.mean()
            mean_lap_loss = result.lap_diff.mean()
            mean_normal_sim_loss = result.nor_sim.mean()
            mean_color_diff_loss = result.gp_color_diff.mean()

            print(f"{mean_rgb_loss=}")
            print(f"{mean_lap_loss=}")
            print(f"{mean_normal_sim_loss=}")
            print(f"{mean_color_diff_loss=}")

            loss = mean_rgb_loss + mean_lap_loss + \
                mean_normal_sim_loss + mean_color_diff_loss

            loss.backward()
            optimizer.step()

        torch.save(gom_avatar_module.state_dict(),
                   DIR / f"gom_avatar_model_{epoch_i}.pth")

        utils.write_video(
            path=DIR / f"output_{epoch_i}.mp4",
            video=vision_utils.denormalize_image(frames),
            fps=30,
        )


if __name__ == "__main__":
    main1()

    print("ok")
