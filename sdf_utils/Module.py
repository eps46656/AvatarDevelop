from __future__ import annotations

import collections
import dataclasses
import math
import typing

import torch
from beartype import beartype

from .. import utils


@beartype
def init(
    layer: torch.nn.Linear,
) -> torch.nn.Linear:
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(layer.weight)

        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

    return layer


@beartype
@dataclasses.dataclass
class ModuleForwardResult:
    gt_signed_dist: torch.Tensor  # [...]
    pr_signed_dist: torch.Tensor  # [...]


@beartype
class Block(torch.nn.Module):
    def __init__(
        self,
        io_features: int,
        m_features: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self.m = torch.nn.Sequential(
            init(torch.nn.Linear(
                io_features, m_features, dtype=dtype, device=device)),

            torch.nn.SiLU(),

            init(torch.nn.Linear(
                m_features, io_features, dtype=dtype, device=device)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.m(x)


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

        layers.append(init(torch.nn.Linear(
            3, 1024, dtype=dtype, device=device)))

        layers.append(torch.nn.SiLU())

        for _ in range(24):
            layers.append(Block(1024, 128, dtype, device))

        layers.append(init(torch.nn.Linear(
            1024, 1, dtype=dtype, device=device)))

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

    def get_param_groups(self, base_lr: float) -> list[dict]:
        return [{
            "params": list(self.module.parameters()),
            "lr": base_lr,
        }]

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

        return ModuleForwardResult(
            gt_signed_dist=signed_dist,
            pr_signed_dist=pr_signed_dist,
        )
