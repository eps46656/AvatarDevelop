import dataclasses
import pathlib

import torch
from beartype import beartype

from . import (config, gom_avatar_utils, people_snapshot_utils, smplx_utils,
               utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")


def main1():
    smpl_model_data_path_dict = {
        "male": config.SMPL_MALE_MODEL,
        "female": config.SMPL_FEMALE_MODEL,
        "neutral": config.SMPL_NEUTRAL_MODEL,
    }

    smplx_model_data_path_dict = {
        "male": config.SMPLX_MALE_MODEL,
        "female": config.SMPLX_FEMALE_MODEL,
        "neutral": config.SMPLX_NEUTRAL_MODEL,
    }

    model_data_dict = {
        key: smplx_utils.ModelData.FromFile(
            model_data_path=value,
            model_config=smplx_utils.smpl_model_config,
            device=DEVICE,
        )
        for key, value in smpl_model_data_path_dict.items()
    }

    people_snapshot_dir = DIR / "people_snapshot_public"

    subject_name = "female-1-casual"

    dataset = people_snapshot_utils.Dataset(
        dataset_dir=people_snapshot_dir,
        subject_name=subject_name,
        model_data_dict=model_data_dict,
        device=DEVICE,
    )

    subject_dir = people_snapshot_dir / subject_name

    subject_data = people_snapshot_utils.ReadSubject(
        subject_dir=subject_dir,
        model_data_dict=model_data_dict,
        device=DEVICE,
    )

    subject_data.video = utils.ImageNormalize(subject_data.video)

    camera_config = subject_data.camera_config

    # ---

    smplx_model_builder = smplx_utils.StaticModelBuilder(
        model_data=subject_data.model_data,
    )

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
        device=DEVICE,
    )

    gom_avatar_module = gom_avatar_utils.Module(
        avatar_blender=smplx_model_blender,
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

            result: gom_avatar_utils.model.GoMAvatarModelForwardResult =\
                gom_avatar_module(
                    subject_data.camera_transform,
                    subject_data.camera_config,

                    subject_data.video[frame_i],

                    subject_data.mask[frame_i],

                    smplx_utils.BlendingParam(
                        body_shapes=subject_data.blending_param.
                        body_shapes,

                        global_transl=subject_data.blending_param.
                        global_transl[frame_i],

                        global_rot=subject_data.blending_param.
                        global_rot[frame_i],

                        body_poses=subject_data.blending_param.
                        body_poses[frame_i],
                    )
                )

            frames[frame_i] = result.rendered_img.detach()

            mean_rgb_loss = result.rgb_loss.mean()
            mean_lap_loss = result.lap_loss.mean()
            mean_normal_sim_loss = result.normal_sim_loss.mean()
            mean_color_diff_loss = result.color_diff_loss.mean()

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

        utils.WriteVideo(
            path=DIR / f"output_{epoch_i}.mp4",
            video=utils.ImageDenormalize(frames),
            fps=30,
        )


if __name__ == "__main__":
    main1()

    print("ok")
