

import dataclasses

import torch
from beartype import beartype

from .. import utils


@beartype
@dataclasses.dataclass
class ModuleForwardResult:
    pr_signed_dist: torch.Tensor  # [...]
    diff_loss: torch.Tensor  # []


@beartype
class Module(torch.nn.Module):
    def __init__(
        self,
        range_min: tuple[float, float, float],
        range_max: tuple[float, float, float],
    ):
        super().__init__()

        assert range_min[0] < range_max[0]
        assert range_min[1] < range_max[1]
        assert range_min[2] < range_max[2]

        self.mean = tuple((l + r) / 2 for l, r in zip(range_min, range_max))
        self.std = tuple((r - l) / 2 for l, r in zip(range_min, range_max))

        layers: list[torch.nn.Module] = list()

        layers.append(torch.nn.Linear(3, 128))
        layers.append(torch.nn.LeakyReLU(0.05))

        for _ in range(12):
            layers.append(torch.nn.Linear(128, 128))
            layers.append(torch.nn.LeakyReLU(0.05))

        layers.append(torch.nn.Linear(128, 1))

        self.module = torch.nn.Sequential(*layers)

    def forward(
        self,
        point_pos: torch.Tensor,  # [..., 3]
        signed_dist: torch.Tensor,  # [...]
    ) -> ModuleForwardResult:
        utils.check_shapes(
            point_pos, (..., 3),
            signed_dist, (...,),
        )

        buffer = torch.empty_like(point_pos)
        # [..., 3]

        buffer[..., 0] = (point_pos[..., 0] - self.mean[0]) / self.std[0]
        buffer[..., 1] = (point_pos[..., 1] - self.mean[1]) / self.std[1]
        buffer[..., 2] = (point_pos[..., 2] - self.mean[2]) / self.std[2]

        pr_signed_dist: torch.Tensor = self.module(buffer).squeeze(-1)
        # [...]

        assert pr_signed_dist.shape == point_pos.shape[:-1]

        if self.training:
            diff = pr_signed_dist - signed_dist

            diff_loss = (diff.square().sum(-1) /
                         (1e-3 + signed_dist.square().sum(-1))).mean()
        else:
            diff_loss = torch.tensor(0.0, dtype=diff.dtype, device=diff.device)

        return ModuleForwardResult(
            pr_signed_dist=pr_signed_dist,
            diff_loss=diff_loss,
        )
