from __future__ import annotations

import collections
import typing

import torch
from beartype import beartype

from . import utils


@beartype
def calc_density(
    point_pos: torch.Tensor,  # [N, D]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:  # [N]
    N, D = utils.check_shapes(point_pos, (-1, -2))
    return kernel(torch.cdist(point_pos, point_pos)).sum(-1)


@beartype
def _query(
    d_pos: torch.Tensor,  # [N, D]
    d_val: torch.Tensor,  # [N, C]
    d_weight: typing.Optional[torch.Tensor],  # [N]
    q_pos: torch.Tensor,  # [..., D]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
) -> tuple[
    torch.Tensor,  # [..., C] gathered val
    torch.Tensor,  # [...] gathered weight
]:
    weight = kernel(utils.vec_norm(q_pos[..., None, :] - d_pos))
    # [..., N]

    if d_weight is not None:
        weight = weight * d_weight
        # [..., N]

    dtype = utils.promote_dtypes(weight, d_val)

    val = utils.einsum(
        "...n, nc -> ...c",
        weight.to(dtype),
        d_val.to(dtype),
    )
    # [..., C]

    return val, weight.sum(-1)


@beartype
def query(
    d_pos: torch.Tensor,  # [N, D]
    d_val: torch.Tensor,  # [N, C]
    d_weight: typing.Optional[torch.Tensor],  # [N]
    q_pos: torch.Tensor,  # [..., D]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
) -> tuple[
    torch.Tensor,  # [..., C] gathered val
    torch.Tensor,  # [..., C] normalized gathered val
    torch.Tensor,  # [...] gathered weight
]:
    N, D, C = -1, -2, -3

    N, D, C = utils.check_shapes(
        d_pos, (N, D),
        d_val, (N, C),
        q_pos, (..., D),
    )

    q_cnt = q_pos.shape[:-1].numel()

    batch_size = utils.clamp(10**8 // max(1, q_cnt), 1, N)

    val = utils.zeros(like=d_val, shape=(*q_pos.shape[:-1], C))

    weight = utils.zeros(like=d_pos, shape=q_pos.shape[:-1])

    for data_beg in range(0, N, batch_size):
        data_end = min(data_beg + batch_size, N)

        cur_val, cur_weight = _query(
            d_pos[data_beg:data_end, :],
            d_val[data_beg:data_end, :],

            None if d_weight is None else
            d_weight[data_beg:data_end],

            q_pos,
            kernel,
        )

        val += cur_val
        weight += cur_weight

    return val, val / (1e-4 + weight[..., None]), weight


@beartype
class KernelSplattingInterpolator(torch.nn.Module):
    def __init__(
        self,
        d_pos: torch.Tensor,  # [N, D]
        d_val: torch.Tensor,  # [N, C]
        kernel: typing.Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()

        N, D, C = -1, -2, -3

        N, D, C = utils.check_shapes(
            d_pos, (N, D),
            d_val, (N, C),
        )

        self.d_pos = d_pos
        self.d_val = d_val
        self.kernel = kernel

    @staticmethod
    def from_data_point(
        d_pos: torch.Tensor,  # [N, D]
        d_val: torch.Tensor,  # [N, C]
        kernel: typing.Callable[[torch.Tensor], torch.Tensor],
    ) -> KernelSplattingInterpolator:
        return KernelSplattingInterpolator(
            d_pos, d_val, kernel)

    @staticmethod
    def from_state_dict(
        state_dict: typing.Mapping[str, object],
        device: typing.Optional[torch.device] = None,
    ) -> KernelSplattingInterpolator:
        d_pos = utils.deserialize_tensor(
            state_dict["d_pos"], device=device)

        d_val = utils.deserialize_tensor(
            state_dict["d_val"], device=device)

        kernel = state_dict["kernel"]

        return KernelSplattingInterpolator(d_pos, d_val, kernel)

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("d_pos", utils.serialize_tensor(self.d_pos)),
            ("d_val", utils.serialize_tensor(self.d_val)),
            ("kernel", self.kernel),
        ])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        self.d_pos = utils.deserialize_tensor(
            state_dict["d_pos"], device=self.d_pos.device)

        self.d_val = utils.deserialize_tensor(
            state_dict["d_val"], device=self.d_val.device)

        self.kernel = state_dict["kernel"]

    def to(self, *args, **kwargs) -> KernelSplattingInterpolator:
        self.d_pos = self.d_pos.to(*args, **kwargs)
        self.d_val = self.d_val.to(*args, **kwargs)

        return self

    def forward(
        self,
        q_pos: torch.Tensor,  # [..., D]
    ) -> torch.Tensor:  # q_val[..., C]
        return query(
            d_pos=self.d_pos,
            d_val=self.d_val,
            d_weight=None,

            q_pos=q_pos,
            kernel=self.kernel,
        )[1]
