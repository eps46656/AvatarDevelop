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
def get_positional_encoding_dim(L: int) -> int:
    return 2 * L + 1


@beartype
def positional_encoding(
    x: torch.Tensor,  # [..., D]
    L: int,
) -> torch.Tensor:
    out = [x]

    for i in range(L):
        out.append(((2 ** i) * torch.pi * x).sin())
        out.append(((2 ** i) * torch.pi * x).cos())

    return torch.cat(out, dim=-1)


@beartype
class ResBlock(torch.nn.Module):
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

            torch.nn.Softplus(beta=100),

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

        self.L = 6

        positional_encoding_dim = 3 * get_positional_encoding_dim(self.L)

        layers: list[torch.nn.Module] = list()

        layers.append(init(torch.nn.Linear(
            positional_encoding_dim, 256, dtype=dtype, device=device)))

        layers.append(torch.nn.Softplus(beta=100))

        for _ in range(8):
            layers.append(ResBlock(256, 256, dtype, device))

        layers.append(init(torch.nn.Linear(
            256, 1, dtype=dtype, device=device)))

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
    ) -> torch.Tensor:  # [...]
        utils.check_shapes(point_pos, (..., 3))

        norm_m = self.norm_m.to(point_pos)

        normed_x = utils.do_rt(norm_m[:, :3], norm_m[:, 3], point_pos)

        pr_signed_dist: torch.Tensor = self.module(
            positional_encoding(normed_x, self.L)
        )[..., 0]
        # [...]

        return pr_signed_dist
