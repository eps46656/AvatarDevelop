import collections
import dataclasses
import os
import typing

import torch
import tqdm
from beartype import beartype

from .. import dataset_utils, training_utils, utils
from .Dataset import Dataset, Sample
from .Module import Module, ModuleForwardResult


@dataclasses.dataclass
class TrainerCoreConfig:
    proj_dir: os.PathLike

    device: torch.device

    batch_size: int

    lr: float
    betas: tuple[float, float]
    gamma: float


@beartype
class TrainerCore(training_utils.TrainerCore):
    def __init__(
        self,
        config: TrainerCoreConfig,
        module: Module,
        dataset: Dataset,
    ):
        self.config = dataclasses.replace(config)
        self.config.proj_dir = utils.to_pathlib_path(self.config.proj_dir)

        self.epoch = 0

        self.module = module
        self.dataset = dataset

        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()

    # --

    def get_epoch(self) -> int:
        return self.epoch

    # --

    def _make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.module.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
        )

    def _make_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR:
        return torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.config.gamma,
        )

    # --

    def state_dict(self, full: bool) \
            -> collections.OrderedDict[str, typing.Any]:
        l = [
            ("epoch", self.epoch),
            ("full", full),
        ]

        if full:
            l.append(("module", self.module))
        else:
            l.append(("module", self.module.state_dict()))

        l.append(("optimizer", self.optimizer.state_dict()))
        l.append(("scheduler", self.scheduler.state_dict()))

        return collections.OrderedDict(l)

    def load_state_dict(self, state_dict: typing.Mapping[str, typing.Any]) \
            -> None:
        self.epoch = state_dict["epoch"]

        full = state_dict["full"]

        if full:
            self.module = state_dict["module"]
        else:
            self.module.load_state_dict(state_dict["module"])

        self.optimizer = self._make_optimizer()
        self.optimizer.load_state_dict(state_dict["optimizer"])

        self.scheduler = self._make_scheduler()
        self.scheduler.load_state_dict(state_dict["scheduler"])

    # --

    def calc_loss(self, forward_result: ModuleForwardResult) \
            -> torch.Tensor:
        # print(f"{forward_result.gt_signed_dist=}")
        # print(f"{forward_result.pr_signed_dist=}")

        diff = forward_result.pr_signed_dist - forward_result.gt_signed_dist

        diff_rms = diff.square().mean().sqrt()

        print(f"{diff_rms.item()=:+.6e}")

        rel_diff = diff.square() / (
            1e-3 + forward_result.gt_signed_dist.square())

        return rel_diff.mean()

    # --

    @utils.mem_clear
    def train(self) -> training_utils.TrainingResult:
        sum_loss = 0.0

        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(
                self.dataset, batch_size=self.config.batch_size)):
            batch_idxes: tuple[torch.Tensor, ...]

            sample: Sample

            result: ModuleForwardResult = self.module(
                point_pos=sample.point_pos,
                signed_dist=sample.signed_dist,
            )

            loss = self.calc_loss(result)

            sum_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

        avg_loss = sum_loss / self.dataset.shape.numel()

        print(f"{avg_loss=}")

        self.epoch += 1
        self.scheduler.step(avg_loss)

        return training_utils.TrainingResult(message=str(avg_loss))

    @utils.mem_clear
    def show_params(self):
        for name, param in self.module.named_parameters():
            print(f"{name}: {param}")

        print(self.optimizer.param_groups)
