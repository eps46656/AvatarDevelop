from __future__ import annotations

import collections
import dataclasses
import typing

import torch
from beartype import beartype

from .. import utils


@beartype
@dataclasses.dataclass
class ModuleForwardResult:
    gt_signed_dist: torch.Tensor  # [...]
    pr_signed_dist: torch.Tensor  # [...]


@beartype
class Module(torch.nn.Module):
    def __init__(
        self,
        range_min: tuple[float, float, float],
        range_max: tuple[float, float, float],
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        assert range_min[0] < range_max[0]
        assert range_min[1] < range_max[1]
        assert range_min[2] < range_max[2]

        mean = tuple((l + r) / 2 for l, r in zip(range_min, range_max))
        std = tuple((r - l) / 2 for l, r in zip(range_min, range_max))

        self.register_buffer("norm_m", torch.tensor([
            [1 / std[0], 0, 0, -mean[0] / std[0]],
            [0, 1 / std[1], 0, -mean[1] / std[1]],
            [0, 0, 1 / std[2], -mean[2] / std[2]],
        ], dtype=dtype, device=device))

        self.norm_m: torch.Tensor  # [3, 4]

        layers: list[torch.nn.Module] = list()

        layers.append(torch.nn.Linear(3, 128, dtype=dtype, device=device))
        layers.append(torch.nn.LeakyReLU(0.05))

        for _ in range(24):
            layers.append(torch.nn.Linear(
                128, 128, dtype=dtype, device=device))

            layers.append(torch.nn.LeakyReLU(0.05))

        layers.append(torch.nn.Linear(128, 1, dtype=dtype, device=device))

        self.module = torch.nn.Sequential(*layers)

    def state_dict(self) -> collections.OrderedDict[str, typing.Any]:
        return collections.OrderedDict([
            ("norm_m", self.norm_m),
            ("module", self.module.state_dict()),
        ])

    def load_state_dict(
        self, state_dict: typing.Mapping[str, typing.Any]
    ) -> None:
        self.norm_m = state_dict["norm_m"]
        self.module.load_state_dict(state_dict["module"])

    def forward(
        self,
        point_pos: torch.Tensor,  # [..., 3]
        signed_dist: torch.Tensor,  # [...]
    ) -> ModuleForwardResult:
        utils.check_shapes(
            point_pos, (..., 3),
            signed_dist, (...,),
        )

        norm_m = self.norm_m.to(point_pos)

        pr_signed_dist: torch.Tensor = self.module(utils.do_rt(
            norm_m[:, :3], norm_m[:, 3], point_pos))[..., 0]
        # [...]

        assert pr_signed_dist.shape == point_pos.shape[:-1]

        return ModuleForwardResult(
            gt_signed_dist=signed_dist,
            pr_signed_dist=pr_signed_dist,
        )
