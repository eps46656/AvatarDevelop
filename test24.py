import copy
import itertools
import math
import pathlib
import typing

import torch
import tqdm
from beartype import beartype

from . import (camera_utils, config, dataset_utils, gom_utils,
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
def MyLossFunc(
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


class MyTrainingCore(training_utils.TrainerCore):
    def train(self) -> training_utils.TrainingResult:
        assert self.scheduler is None or isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        sum_loss = 0.0

        for idxes, sample in tqdm.tqdm(self.dataset_loader):
            idxes: torch.Tensor

            sample: people_snapshot_utils.SubjectData = self.dataset[idxes]

            result = self.module(
                camera_transform=sample.camera_transform,
                camera_config=sample.camera_config,
                img=sample.video,
                mask=sample.mask,
                blending_param=sample.blending_param,
            )

            if not isinstance(result, dict):
                result = result.__dict__

            loss: torch.Tensor = self.loss_func(**result)

            sum_loss += float(loss) * idxes.numel()

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        avg_loss = sum_loss / len(self.dataset)

        if self.scheduler is not None:
            self.scheduler.step(avg_loss)

        return training_utils.TrainingResult(
            avg_loss=avg_loss
        )


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
        key: smplx_utils.ReadModelData(
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

    smplx_model_builder = smplx_utils.ModelBlender(
        model_data=model_data,
        device=DEVICE,
    )

    module = gom_utils.model.GoMAvatarModel(
        avatar_blending_layer=smplx_model_builder,
        color_channels_cnt=3,
    ).train()

    T = len(dataset)

    frames = torch.empty(
        (T, 3,
         subject_data.camera_config.img_h, subject_data.camera_config.img_w)
    )

    optimizer = torch.optim.Adam(
        module.parameters(),
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

    training_core = MyTrainingCore(
        module=module,
        dataset=dataset,
        dataset_loader=dataset_loader,
        loss_func=MyLossFunc,
        optimizer=optimizer,
        scheduler=scheduler
    )

    trainer = training_utils.Trainer(
        proj_dir=DIR / "train_2025_0325"
    )

    trainer.set_training_core(training_core)

    trainer.save("init", True)

    trainer.train(10)

    trainer.save("0", False)


if __name__ == "__main__":
    main1()

    print("ok")
