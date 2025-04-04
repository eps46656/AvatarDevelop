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

from . import (avatar_utils, camera_utils, config, dataset_utils,
               dw_interp_utils, gaussian_utils,
               people_snapshot_utils, rendering_utils, sdf_utils, smplx_utils,
               texture_utils, training_utils, transform_utils, utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")

PROJ_DIR = DIR / "train_2025_0404_2"

BATCH_SIZE = 4

MODEL_TYPE = "smplx"
MODEL_SUBTYPE = "female"


@beartype
class MyTrainingCore(training_utils.TrainingCore):
    def __init__(
        self,
        module: sdf_utils.Module,
        dataset: sdf_utils.Dataset,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
    ):
        self.__module = module
        self.__dataset = dataset
        self.__optimizer = optimizer
        self.__scheduler = scheduler

    # --

    @property
    def module(self) -> sdf_utils.Module:
        return self.__module

    @module.setter
    def module(self, module: sdf_utils.Module) -> None:
        self.__module = module

    # --

    @property
    def dataset(self) -> sdf_utils.Dataset:
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset: sdf_utils.Dataset) -> None:
        self.__dataset = dataset

    # --

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.__optimizer = optimizer

    # --

    @property
    def scheduler(self) -> object:
        return self.__scheduler

    @scheduler.setter
    def scheduler(self, scheduler: object) -> None:
        self.__scheduler = scheduler

    # --

    def train(self) \
            -> training_utils.TrainingResult:
        assert self.scheduler is None or isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        sum_loss = 0.0

        for batch_idxes, sample in tqdm.tqdm(
                dataset_utils.load(self.dataset, batch_size=BATCH_SIZE)):
            batch_idxes: tuple[torch.Tensor, ...]

            sample: sdf_utils.Sample = self.dataset[batch_idxes]

            result: sdf_utils.ModuleForwardResult = self.module(
                point_pos=sample.point_pos,
                signed_dist=sample.signed_dist,
            )

            loss = result.diff_loss
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

        avg_loss = sum_loss / self.dataset.shape.numel()

        print(f"{avg_loss=}")

        if self.scheduler is not None:
            self.scheduler.step(avg_loss)

        return training_utils.TrainingResult(
            avg_loss=avg_loss
        )

    @utils.mem_clear
    def show_params(self):
        for name, param in self.module.named_parameters():
            print(f"{name}: {param}")

        print(self.optimizer.param_groups)


def main1():
    torch.autograd.set_detect_anomaly(True, True)

    model_data_path_dict = {
        "smpl": {
            "male": config.SMPL_MALE_MODEL,
            "female": config.SMPL_FEMALE_MODEL,
            "neutral": config.SMPL_NEUTRAL_MODEL,
        },
        "smplx": {
            "male": config.SMPLX_MALE_MODEL,
            "female": config.SMPLX_FEMALE_MODEL,
            "neutral": config.SMPLX_NEUTRAL_MODEL,
        },
    }

    # ---

    model_data = smplx_utils.ModelData.from_file(
        model_data_path=model_data_path_dict[MODEL_TYPE][MODEL_SUBTYPE],
        model_config=smplx_utils.smpl_model_config,
        device=DEVICE,
    )

    range_min = (-2.0, -2.0, -2.0)
    range_max = (+2.0, +2.0, +2.0)

    module = sdf_utils.Module(
        range_min=range_min,
        range_max=range_max,
    ).to(DEVICE).train()

    dataset = sdf_utils.Dataset(
        mean_x=(range_min[0] + range_max[0]) / 2,
        mean_y=(range_min[1] + range_max[1]) / 2,
        mean_z=(range_min[2] + range_max[2]) / 2,

        std_x=(range_max[0] - range_min[0]) / 4,
        std_y=(range_max[1] - range_min[1]) / 4,
        std_z=(range_max[2] - range_min[2]) / 4,
    )

    lr = 1e-3

    param_groups = utils.get_param_groups(module, lr)

    print(param_groups)

    optimizer = torch.optim.Adam(
        param_groups,
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=pow(0.1, 1/4),
        patience=5,
        threshold=0.05,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-8,
    )

    training_core = MyTrainingCore(
        module=module,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler
    )

    trainer = training_utils.Trainer(
        proj_dir=PROJ_DIR,
        device=DEVICE,
    )

    trainer.set_training_core(training_core)

    # trainer.load_latest()

    # trainer.training_core.bake_texture_face(1000, 1000)

    trainer.enter_cli()


if __name__ == "__main__":
    main1()

    print("ok")
