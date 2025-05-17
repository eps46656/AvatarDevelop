import typing

import torch
from beartype import beartype

from . import utils


@beartype
def calc_density(
    point_pos: torch.Tensor,  # [N, P]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:  # [N]
    N, P = -1, -2
    N, P = utils.check_shapes(point_pos, (N, P))

    return kernel(torch.cdist(point_pos, point_pos)).sum(-1)


@beartype
def _splat(
    data_point_pos: torch.Tensor,  # [N, P]
    data_point_val: torch.Tensor,  # [N, Q]
    data_point_weight: typing.Optional[torch.Tensor],  # [N]
    q_point_pos: torch.Tensor,  # [..., P]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
) -> tuple[
    torch.Tensor,  # [..., Q] gathered val
    torch.Tensor,  # [...] gathered weight
]:
    weight = kernel(utils.vec_norm(q_point_pos[..., None, :] - data_point_pos))
    # [..., N]

    if data_point_weight is not None:
        weight = weight * data_point_weight
        # [..., N]

    dtype = utils.promote_dtypes(weight, data_point_val)

    val = utils.einsum(
        "...n, nq -> ...q",
        weight.to(dtype),
        data_point_val.to(dtype),
    )

    return val, weight.sum(-1)


@beartype
def splat(
    data_point_pos: torch.Tensor,  # [N, P]
    data_point_val: torch.Tensor,  # [N, Q]
    data_point_weight: typing.Optional[torch.Tensor],  # [N]
    q_point_pos: torch.Tensor,  # [..., P]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
) -> tuple[
    torch.Tensor,  # [..., Q] gathered val
    torch.Tensor,  # [...] gathered weight
]:
    N, P, Q = -1, -2, -3

    N, P, Q = utils.check_shapes(
        data_point_pos, (N, P),
        data_point_val, (N, Q),
        q_point_pos, (..., P),
    )

    q_points_cnt = q_point_pos.shape[:-1].numel()

    batch_size = utils.clamp(10**8 // q_points_cnt, 1, N)

    val = utils.zeros_like(
        data_point_val, shape=(*q_point_pos.shape[:-1], Q))

    weight = utils.zeros_like(data_point_pos, shape=q_point_pos.shape[:-1])

    for data_beg in range(0, N, batch_size):
        data_end = min(data_beg + batch_size, N)

        cur_val, cur_weight = _splat(
            data_point_pos[data_beg:data_end, :],
            data_point_val[data_beg:data_end, :],
            data_point_weight[data_beg:data_end],
            q_point_pos,
            kernel,
        )

        val += cur_val
        weight += cur_weight

    return val, weight


@beartype
def interp(
    data_point_pos: torch.Tensor,  # [N, P]
    data_point_val: torch.Tensor,  # [N, Q]
    data_point_weight: typing.Optional[torch.Tensor],  # [N]
    q_point_pos: torch.Tensor,  # [..., P]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
):
    val, weight = splat(
        data_point_pos, data_point_val, data_point_weight, q_point_pos, kernel)
    # val[..., Q]
    # weight[...]

    return val / weight[..., None]
