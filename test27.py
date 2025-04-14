import pathlib
import time
import typing

import torch
import tqdm
from beartype import beartype

from . import (config, dataset_utils, gom_utils, people_snapshot_utils,
               smplx_utils, training_utils, utils, vision_utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = utils.CUDA_DEVICE

PROJ_DIR = DIR / "train_2025_0328"


ALPHA_RGB = 1.0
ALPHA_LAP = 1.0
ALPHA_NORMAL_SIM = 1.0
ALPHA_COLOR_DIFF = 1.0


@beartype
def MyLossFunc(
    rendered_img: torch.Tensor,  # [..., C, H, W]

    rgb_loss: float | torch.Tensor,
    lap_loss: float | torch.Tensor,
    normal_sim_loss: float | torch.Tensor,
    color_diff_loss: float | torch.Tensor,
):
    weighted_rgb_loss = ALPHA_RGB * rgb_loss
    weighted_lap_loss = ALPHA_LAP * lap_loss
    weighted_normal_sim_loss = ALPHA_NORMAL_SIM * normal_sim_loss
    weighted_color_diff_loss = ALPHA_COLOR_DIFF * color_diff_loss

    return weighted_rgb_loss + weighted_lap_loss + weighted_normal_sim_loss + weighted_color_diff_loss


@beartype
class MyTrainingCore(training_utils.TrainingCore):
    def train(self) -> training_utils.TrainingResult:
        assert self.scheduler is None or isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        sum_loss = 0.0

        for batch_idxes, sample in tqdm.tqdm(self.dataset_loader):
            batch_idxes: tuple[torch.Tensor, ...]

            sample: gom_utils.Sample = self.dataset[batch_idxes]

            result: gom_utils.ModuleForwardResult = self.module(
                camera_transform=sample.camera_transform,
                camera_config=sample.camera_config,
                img=sample.img,
                mask=sample.mask,
                blending_param=sample.blending_param,
            )

            loss: torch.Tensor = self.loss_func(**result.__dict__)

            sum_loss += float(loss) * batch_idxes[0].numel()

            self.optimizer.zero_grad()

            loss.backward()

            """
            for name, param in self.module.named_parameters():

                print(f"{name=}")
                print(f"{param=}")
                print(f"{param.grad=}")

                assert param.isfinite().all()
                assert param.grad.isfinite().all()
            """

            self.optimizer.step()

        avg_loss = sum_loss / self.dataset.shape.numel()

        print(f"{avg_loss=}")

        if self.scheduler is not None:
            self.scheduler.step(avg_loss)

        return training_utils.TrainingResult(
            avg_loss=avg_loss
        )

    def eval(self):
        self.dataset: gom_utils.Dataset

        out_frames = utils.empty_like(
            self.dataset.sample.img,
            dtype=utils.FLOAT,
            device=utils.CPU_DEVICE,
        )
        # [T, C, H, W]

        T, C, H, W = self.dataset.sample.img.shape

        batch_shape = self.dataset.shape

        with torch.no_grad():
            for batch_idxes, sample in tqdm.tqdm(self.dataset_loader):
                batch_idxes: tuple[torch.Tensor, ...]

                idxes = utils.ravel_idxes(
                    batch_idxes, self.dataset.shape)
                # [K]

                idxes = idxes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                    idxes.shape + (C, H, W))
                # [K, C, H, W]

                sample: gom_utils.Sample = self.dataset[batch_idxes]

                result: gom_utils.ModuleForwardResult = self.module(
                    camera_transform=sample.camera_transform,
                    camera_config=sample.camera_config,
                    img=sample.img,
                    mask=sample.mask,
                    blending_param=sample.blending_param,
                )

                rendered_img = result.gp_render_img.reshape((-1, C, H, W))
                # [K, C, H, W]

                out_frames.scatter_(
                    0,
                    idxes.to(out_frames.device),
                    rendered_img.to(out_frames.device))

                """

                out_frames[idxes[k, c, h, w], c, h, w] =
                    rendered_img[k, c, h, w]

                """

        utils.write_video(
            path=PROJ_DIR / f"output_{int(time.time())}.mp4",
            video=vision_utils.denormalize_image(out_frames),
            fps=25,
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

    model_data_dict = {
        key: smplx_utils.Core.from_file(
            model_data_path=value,
            model_config=smplx_utils.smpl_model_config,
            device=DEVICE,
        )
        for key, value in smpl_model_data_path_dict.items()
    }

    people_snapshot_dir = DIR / "people_snapshot_public"

    subject_name = "female-1-casual"

    subject_dir = people_snapshot_dir / subject_name

    subject_data = people_snapshot_utils.read_subject(
        subject_dir, model_data_dict, DEVICE)

    print(f"{subject_data.camera_transform.shape=}")
    print(f"{subject_data.video.shape=}")
    print(f"{subject_data.mask.shape=}")
    print(f"{subject_data.blending_param.shape=}")
    print(f"{subject_data.model_data.vert_pos.shape=}")

    dataset = gom_utils.Dataset(gom_utils.Sample(
        camera_transform=subject_data.camera_transform,
        camera_config=subject_data.camera_config,
        img=subject_data.video,
        mask=subject_data.mask,
        blending_param=subject_data.blending_param,
    )).to(DEVICE)

    dataset_loader = dataset_utils.DatasetLoader(
        dataset,
        batch_size=4,
    )

    # ---

    smplx_model_builder = smplx_utils.DeformableModelBuilder(
        model_data=subject_data.model_data,
    ).to(DEVICE)

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    gom_avatar_module = gom_utils.Module(
        avatar_blender=smplx_model_blender,
        color_channels_cnt=3,
    ).to(DEVICE).train()

    optimizer = torch.optim.Adam(
        gom_avatar_module.parameters(),
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
        module=gom_avatar_module,
        dataset=dataset,
        dataset_loader=dataset_loader,
        loss_func=MyLossFunc,
        optimizer=optimizer,
        scheduler=scheduler
    )

    trainer = training_utils.Trainer(
        proj_dir=PROJ_DIR,
        device=DEVICE,
    )

    trainer.set_training_core(training_core)

    # trainer.load_latest()

    trainer.enter_cli()

    return

    for epoch_i in range(10):
        trainer.train(epochs_cnt=1)
        trainer.save()
        trainer.eval()

    print(f"###")

    for name, param in gom_avatar_module.named_parameters():
        print(f"{name=}")
        print(f"{param=}")

    print(f"###")


if __name__ == "__main__":
    main1()

    print("ok")
