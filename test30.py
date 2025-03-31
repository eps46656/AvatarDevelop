import copy
import itertools
import math
import pathlib
import time
import typing

import torch
import tqdm
from beartype import beartype

from . import (avatar_utils, camera_utils, config, dataset_utils, gom_utils,
               people_snapshot_utils, smplx_utils, texture_utils,
               training_utils, transform_utils, utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")

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

            sample: gom_utils.Sample = \
                self.dataset.batch_get(batch_idxes)

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

        out_frames = torch.empty_like(
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

                sample: gom_utils.Sample = \
                    self.dataset.batch_get(batch_idxes)

                result: gom_utils.ModuleForwardResult = self.module(
                    camera_transform=sample.camera_transform,
                    camera_config=sample.camera_config,
                    img=sample.img,
                    mask=sample.mask,
                    blending_param=sample.blending_param,
                )

                rendered_img = result.rendered_img.reshape((-1, C, H, W))
                # [K, C, H, W]

                out_frames.scatter_(
                    0,
                    idxes.to(device=out_frames.device),
                    rendered_img.to(device=out_frames.device))

                """

                out_frames[idxes[k, c, h, w], c, h, w] =
                    rendered_img[k, c, h, w]

                """

        utils.write_video(
            path=PROJ_DIR / f"output_{int(time.time())}.mp4",
            video=utils.image_denormalize(out_frames),
            fps=25,
        )


@beartype
def map(
    gom_module: gom_utils.Module,
    mapping_blending_param: object,
    img_h: int,
    img_w: int,
):
    assert 0 < img_h
    assert 0 < img_w

    avatar_blender = gom_module.avatar_blender

    avatar_model: avatar_utils.AvatarModel = avatar_blender(
        mapping_blending_param)

    vertex_positions = avatar_model.vertex_positions
    # [V, 3]

    texture_faces = avatar_model.texture_faces
    # [F, 3]

    texture_vertex_positions = avatar_model.texture_vertex_positions
    # [TV, 2]

    gp_colors = gom_module.gp_colors
    # [F, C]

    V, TV, F = -1, -2, -3

    V, TV, F = utils.check_shapes(
        vertex_positions, (V, 3),
        texture_vertex_positions, (TV, 2),
        texture_faces, (F, 3),
    )

    m = texture_utils.position_to_map(
        vertex_positions,

        texture_faces=texture_faces,
        texture_vertex_positions=texture_vertex_positions,

        img_h=img_h,
        img_w=img_w,
    )
    # [img_h, img_w]

    ret = torch.zeros((3, img_h, img_w), dtype=utils.FLOAT)

    for pixel_i in range(img_h):
        for pixel_j in range(img_w):
            l = m[pixel_i][pixel_j]

            if l is None:
                continue

            if 1 < len(l):
                print(f"multi mapping at ({pixel_i}, {pixel_j})")

            face_i, _ = l[0]

            ret[:, pixel_i, pixel_j] = gp_colors[face_i, :]

    utils.write_image(
        DIR / " tex_map.ong",
        ret
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
        key: smplx_utils.ModelData.from_file(
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

    dataset = None

    dataset_loader = None

    # ---

    smplx_model_builder = smplx_utils.DeformableModelBuilder(
        model_data=subject_data.model_data,
    ).to(device=DEVICE)

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    gom_avatar_module = gom_utils.Module(
        avatar_blender=smplx_model_blender,
        color_channels_cnt=3,
    ).to(device=DEVICE).train()

    optimizer = None

    scheduler = None

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

    trainer.load_latest()

    map(
        training_core.module,
        smplx_utils.BlendingParam(),
        1000,
        1000,
    )


if __name__ == "__main__":
    main1()

    print("ok")
