import copy
import itertools
import math
import pathlib
import typing

import torch
import tqdm
from beartype import beartype

from . import (camera_utils, config, dataset_utils, gom_avatar_utils,
               people_snapshot_utils, smplx_utils, training_utils,
               transform_utils, utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")


ALPHA_RGB = 1.0
ALPHA_LAP = 1.0
ALPHA_NORMAL_SIM = 1.0
ALPHA_COLOR_DIFF = 1.0


@beartype
def LossFunc(
    rendered_img: torch.Tensor,  # [..., C, H, W]

    rgb_loss: typing.Optional[torch.Tensor],
    lap_loss: typing.Optional[torch.Tensor],
    normal_sim_loss: typing.Optional[torch.Tensor],
    color_diff_loss: typing.Optional[torch.Tensor],
):
    weighted_rgb_loss = ALPHA_RGB * rgb_loss.mean()
    weighted_lap_loss = ALPHA_LAP * lap_loss.mean()
    weighted_normal_sim_loss = ALPHA_NORMAL_SIM * normal_sim_loss.mean()
    weighted_color_diff_loss = ALPHA_COLOR_DIFF * color_diff_loss.mean()

    return weighted_rgb_loss + weighted_lap_loss + weighted_normal_sim_loss + weighted_color_diff_loss


class MyTrainingCore(training_utils.TrainingCore):
    def Train(self) -> training_utils.TrainingResult:
        assert self.scheduler is None or isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        sum_loss = 0.0

        for idxes in tqdm.tqdm(self.dataset_loader):
            idxes: torch.Tensor

            sample = self.dataset.BatchGet(idxes)
            result = self.module(**sample)

            loss: torch.Tensor = self.loss_func(**result)

            sum_loss += float(loss) * idxes.numel()

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        avg_loss = sum_loss / len(self.dataset)

        if self.scheduler is not None:
            self.scheduler.step()

        return training_utils.TrainingResult(
            avg_loss=avg_loss
        )


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

    body_shapes_cnt = 10
    expr_shapes_cnt = 0
    body_joints_cnt = 24
    jaw_joints_cnt = 0
    eye_joints_cnt = 0
    hand_joints_cnt = 0

    model_data_dict = {
        key: smplx_utils.ReadSMPLXModelData(
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

    dataset = people_snapshot_utils.Dataset(
        dataset_dir=people_snapshot_dir,
        subject_name=subject_name,
        model_data_dict=model_data_dict,
        device=DEVICE,
    )

    dataset_loader = dataset_utils.DatasetLoader(
        dataset,
        batch_size=8,
    )

    # ---

    subject_data = dataset.subject_data

    model_data = copy.copy(subject_data.model_data)

    model_data.vertex_positions = torch.nn.Parameter(
        model_data.vertex_positions)

    smplx_model_builder = smplx_utils.SMPLXModelBuilder(
        model_data=model_data,
        device=DEVICE,
    )

    gom_avatar_model = gom_avatar_utils.model.GoMAvatarModel(
        avatar_blending_layer=smplx_model_builder,
        color_channels_cnt=3,
    ).train()

    T = len(dataset)

    frames = torch.empty(
        (T, 3,
         subject_data.camera_config.img_h, subject_data.camera_config.img_w)
    )

    optimizer = torch.optim.Adam(
        gom_avatar_model.parameters(),
        lr=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=pow(0.1, 1/4),
        patience=5,
        threshold=0.05,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-7,
    )

    for epoch_i in range(10):
        for frame_i in range(T):
            optimizer.zero_grad()

            print(f"{epoch_i=}\t\t{frame_i=}")

            result: gom_avatar_utils.model.GoMAvatarModelForwardResult =\
                gom_avatar_model(
                    subject_data.camera_transform,
                    subject_data.camera_config,

                    subject_data.video[frame_i],

                    subject_data.mask[frame_i],

                    smplx_utils.SMPLXBlendingParam(
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

        torch.save(gom_avatar_model.state_dict(),
                   DIR / f"gom_avatar_model_{epoch_i}.pth")

        utils.WriteVideo(
            path=DIR / f"output_{epoch_i}.mp4",
            video=utils.ImageDenormalize(frames),
            fps=30,
        )


if __name__ == "__main__":
    main1()

    print("ok")
