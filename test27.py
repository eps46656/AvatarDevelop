import pathlib
import time
import typing

import torch
import tqdm
from beartype import beartype

from . import (config, dataset_utils, gom_avatar_utils, people_snapshot_utils,
               smplx_utils, training_utils, utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = utils.CUDA_DEVICE


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


@beartype
class MyTrainingCore(training_utils.TrainingCore):
    def Train(self) -> training_utils.TrainingResult:
        assert self.scheduler is None or isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        sum_loss = 0.0

        for batch_idxes, sample in tqdm.tqdm(self.dataset_loader):
            batch_idxes: torch.Tensor

            sample: people_snapshot_utils.SubjectData = \
                self.dataset.BatchGet(batch_idxes)

            result: gom_avatar_utils.ModuleForwardResult = self.module(
                camera_transform=sample.camera_transform,
                camera_config=sample.camera_config,
                img=sample.video,
                mask=sample.mask,
                blending_param=sample.blending_param,
            )

            loss: torch.Tensor = self.loss_func(**result.__dict__)

            sum_loss += float(loss) * batch_idxes.numel()

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        avg_loss = sum_loss / len(self.dataset)

        if self.scheduler is not None:
            self.scheduler.step(avg_loss)

        return training_utils.TrainingResult(
            avg_loss=avg_loss
        )

    def Eval(self):
        self.dataset: gom_avatar_utils.Dataset

        out_frames = torch.empty_like(self.dataset.sample.img)
        # [T, C, H, W]

        T, C, H, W = self.dataset.sample.img.shape

        batch_shape = self.dataset.GetBatchShape()

        with torch.no_grad():
            for batch_idxes, sample in tqdm.tqdm(self.dataset_loader):
                batch_idxes: tuple[torch.Tensor, ...]

                idxes = utils.RavelIdxes(
                    batch_idxes, self.dataset.GetBatchShape())
                # [K]

                idxes = idxes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                    idxes.shape + (C, H, W))
                # [K, C, H, W]

                sample: people_snapshot_utils.SubjectData = \
                    self.dataset.BatchGet(batch_idxes)

                result: gom_avatar_utils.ModuleForwardResult = self.module(
                    camera_transform=sample.camera_transform,
                    camera_config=sample.camera_config,
                    img=sample.video,
                    mask=sample.mask,
                    blending_param=sample.blending_param,
                )

                rendered_img = result.rendered_img.reshape((-1, C, H, W))
                # [K, C, H, W]

                out_frames.scatter_(
                    0, idxes, rendered_img)

                """

                out_frames[idxes[k, c, h, w], c, h, w] =
                    rendered_img[k, c, h, w]

                """

        utils.WriteVideo(
            path=DIR / f"output_{int(time.time())}.mp4",
            video=utils.ImageDenormalize(out_frames),
            fps=25,
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
        subject_dir, model_data_dict, DEVICE)

    dataset = gom_avatar_utils.Dataset(gom_avatar_utils.Sample(
        camera_transform=subject_data.camera_transform,
        camera_config=subject_data.camera_config,
        img=subject_data.video,
        mask=subject_data.mask,
        blending_param=subject_data.blending_param,
    ))

    dataset_loader = dataset_utils.DatasetLoader(
        dataset,
        batch_size=8,
    )

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

    optimizer = torch.optim.Adam(
        gom_avatar_module.parameters(),
        lr=1e-4,
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
        module=gom_avatar_module,
        dataset=dataset,
        dataset_loader=dataset_loader,
        loss_func=MyLossFunc,
        optimizer=optimizer,
        scheduler=scheduler
    )

    trainer = training_utils.Trainer(
        proj_dir=DIR / "train_2025_0325"
    )

    pass

    """
    torch.save(gom_avatar_module.state_dict(),
               DIR / f"gom_avatar_model_{epoch_i}.pth")

    utils.WriteVideo(
        path=DIR / f"output_{epoch_i}.mp4",
        video=utils.ImageDenormalize(frames),
        fps=subject_data.fps,
    )
    """


if __name__ == "__main__":
    main1()

    print("ok")
