import collections
import dataclasses
import os
import typing

import tabulate
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

    alpha_signed_dist: float
    alpha_eikonal: float


@beartype
def call_eikonal_loss(
    point_pos: torch.Tensor,  # [..., D]
    signed_dist: torch.Tensor,  # [...]
) -> torch.Tensor:  # []
    grads = torch.autograd.grad(
        outputs=signed_dist,
        inputs=point_pos,

        grad_outputs=utils.dummy_ones(like=signed_dist),

        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # [..., D]

    grad_norm = utils.vec_norm(grads)  # [...]

    loss = (grad_norm - 1).square().mean()

    return loss


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
            utils.get_param_groups(self.module, self.config.lr),
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

    def calc_loss(
        self,
        point_pos: torch.Tensor,  # [..., D]
        forward_result: ModuleForwardResult
    ) -> torch.Tensor:
        D = point_pos.shape[-1]

        def _f(x):
            return f"{x.item():+.4e}"

        loss_tab = [("", "loss", "weighted loss")]

        pr_signed_dist = forward_result.pr_signed_dist
        gt_signed_dist = forward_result.gt_signed_dist
        # [...]

        # ---

        """
        diff = pr_signed_dist - gt_signed_dist

        signed_dist_diff = (diff.square() / (
            1e-4 + gt_signed_dist.square())).mean()
        """

        signed_dist_diff = (pr_signed_dist - gt_signed_dist).square().mean()

        w_signed_dist_diff = self.config.alpha_signed_dist * signed_dist_diff

        loss_tab.append(
            ("signed_dist_diff", _f(signed_dist_diff), _f(w_signed_dist_diff)))

        # ---

        eikonal_loss = call_eikonal_loss(point_pos, pr_signed_dist)
        w_eikonal_loss = self.config.alpha_eikonal * eikonal_loss

        loss_tab.append(("eikonal_loss", _f(eikonal_loss), _f(w_eikonal_loss)))

        # ---

        print(tabulate.tabulate(zip(*loss_tab), tablefmt="grid"))

        return w_signed_dist_diff + w_eikonal_loss

    # --

    @utils.mem_clear
    def train(self) -> training_utils.TrainingResult:
        sum_loss = 0.0

        self.dataset.refresh()

        for batch_size, batch_idx, sample in tqdm.tqdm(dataset_utils.load(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )):
            sample: Sample

            self.optimizer.zero_grad()

            point_pos = sample.point_pos.clone().requires_grad_()
            signed_dist = sample.signed_dist

            result: ModuleForwardResult = self.module(
                point_pos=point_pos,
                signed_dist=signed_dist,
            )

            loss = self.calc_loss(point_pos, result)

            sum_loss += float(loss) * batch_size

            loss.backward()

            self.optimizer.step()

        avg_loss = sum_loss / self.dataset.shape.numel()

        self.scheduler.step()

        self.epoch += 1

        return training_utils.TrainingResult(
            message=str(avg_loss)
        )

    @utils.mem_clear
    def show_params(self):
        for name, param in self.module.named_parameters():
            mean = param.mean().item()
            std = param.std().item()
            grad = None if param.grad is None else param.grad.square().sum().sqrt().item()

            print(f"{name}: {mean=} {std=} {grad}")

        # print(self.optimizer.param_groups)
