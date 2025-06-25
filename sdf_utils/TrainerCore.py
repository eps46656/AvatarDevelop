import collections
import dataclasses
import typing

import tabulate
import torch
import tqdm
from beartype import beartype

from .. import dataset_utils, training_utils, utils
from .Dataset import *
from .Module import *


@dataclasses.dataclass
class TrainerCoreConfig:
    proj_dir: utils.PathLike

    device: torch.device

    batch_size: int

    lr: typing.Callable[[int], float]
    betas: tuple[float, float]

    alpha_signed_dist: typing.Callable[[int], float]
    alpha_eikonal: typing.Callable[[int], float]


@beartype
def calc_eikonal_loss(
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
        dataset: typing.Optional[Dataset],
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
        base_lr = self.config.lr(0)

        return torch.optim.Adam(
            utils.get_param_groups(self.module, base_lr),
            lr=base_lr,
            betas=self.config.betas,
        )

    def _make_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: self.config.lr(epoch) / self.config.lr(0),
            last_epoch=self.epoch - 1,
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
        pr_signed_dist: torch.Tensor,  # [...]
        gt_signed_dist: torch.Tensor,  # [...]
    ) -> torch.Tensor:
        D = point_pos.shape[-1]

        loss_table = utils.LossTable()

        # ---

        """
        diff = pr_signed_dist - gt_signed_dist

        signed_dist_diff = (diff.square() / (
            1e-4 + gt_signed_dist.square())).mean()
        """

        signed_dist_diff = (pr_signed_dist - gt_signed_dist).square().mean()

        w_signed_dist_diff = \
            self.config.alpha_signed_dist(self.epoch) * signed_dist_diff

        loss_table.add(
            "signed_dist_diff", signed_dist_diff, w_signed_dist_diff)

        # ---

        eikonal_loss = calc_eikonal_loss(point_pos, pr_signed_dist)

        w_eikonal_loss = self.config.alpha_eikonal(self.epoch) * eikonal_loss

        loss_table.add(
            "eikonal_loss", eikonal_loss, w_eikonal_loss)

        # ---

        loss_table.show()

        loss = loss_table.get_weighted_sum_loss()

        assert loss.isfinite().all()

        print(f"{loss=}")

        return loss

    # --

    @utils.mem_clear
    def train(self) -> training_utils.TrainingResult:
        assert self.dataset is not None

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
            gt_signed_dist = sample.signed_dist

            pr_signed_dist = self.module(point_pos)

            loss = self.calc_loss(point_pos, pr_signed_dist, gt_signed_dist)

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
    def show_params(self) -> None:
        utils.show_tensor_info(self.module.named_parameters())
