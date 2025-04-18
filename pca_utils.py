

import torch
from beartype import beartype

from . import utils


@beartype
def feed(
    x: torch.Tensor,  # [..., D]
    inplace: bool,

    dst_sum_x: torch.Tensor,  # [D]
    dst_sum_xxt: torch.Tensor,  # [D, D]
) -> tuple[
    torch.Tensor,  # dst_sum_x[D]
    torch.Tensor,  # dst_sum_xxt[D, D]
]:
    D = -1

    D = utils.check_shapes(
        x, (..., D),
        dst_sum_x, (D,),
        dst_sum_xxt, (D, D),
    )

    inc_sum_x = torch.einsum("...i->i", x)
    inc_sum_xxt = torch.einsum("...i,...j->ij", x, x)

    if inplace:
        dst_sum_x += inc_sum_x
        dst_sum_xxt += inc_sum_xxt
    else:
        dst_sum_x = dst_sum_x + inc_sum_x
        dst_sum_xxt = dst_sum_xxt + inc_sum_xxt

    return dst_sum_x, dst_sum_xxt


@beartype
def scatter_feed(
    idx: torch.Tensor,  # [B]
    x: torch.Tensor,  # [B, D]
    inplace: bool,

    dst_cnt: torch.Tensor,  # [N]
    dst_sum_x: torch.Tensor,  # [N, D]
    dst_sum_xxt: torch.Tensor,  # [N, D, D]
) -> tuple[
    torch.Tensor,  # dst_cnts[N]
    torch.Tensor,  # dst_sum_x[N, D]
    torch.Tensor,  # dst_sum_xxt[N, D, D]
]:
    B, N, D = -1, -2, -3

    B, N, D = utils.check_shapes(
        idx, (B,),
        x, (B, D),
        dst_cnt, (N,),
        dst_sum_x, (N, D),
        dst_sum_xxt, (N, D, D),
    )

    xxt = x.unsqueeze(-1) @ x.unsqueeze(-2)
    # [..., D, D]

    if inplace:
        dst_cnt += idx.bincount(minlength=N)

        dst_sum_x.index_add_(
            0, idx, x.to(dst_sum_x.device, dst_sum_x.dtype))

        dst_sum_xxt.index_add_(
            0, idx, xxt.to(dst_sum_xxt.device, dst_sum_xxt.dtype))
    else:
        dst_cnt = dst_cnt + idx.bincount(minlength=N)

        dst_sum_x = dst_sum_x.index_add(
            0, idx, x.to(dst_sum_x.device, dst_sum_x.dtype))

        dst_sum_xxt = dst_sum_xxt.index_add(
            0, idx, xxt.to(dst_sum_xxt.device, dst_sum_xxt.dtype))

    return dst_cnt, dst_sum_x, dst_sum_xxt


@beartype
def get_pca(
    cnt: torch.Tensor,  # [...]
    sum_x: torch.Tensor,  # [..., D]
    sum_xxt: torch.Tensor,  # [..., D, D]
):
    D = utils.check_shapes(
        sum_x, (..., -1),
        sum_xxt, (..., -1, -1),
    )

    cnt = cnt.clamp(2, None)

    mean_x = sum_x / cnt.unsqueeze(-1)
    # [..., D]

    cov = (sum_xxt - mean_x.unsqueeze(-1) @ sum_x.unsqueeze(-2)) \
        / (cnt - 1).unsqueeze(-1).unsqueeze(-1)
    # [..., D, D]

    eig_val, eig_vec = torch.linalg.eigh(cov)

    eig_val: torch.Tensor  # [..., D]
    eig_vec: torch.Tensor  # [..., D, D]

    eig_val = eig_val.flip(-1)
    eig_vec = eig_vec.transpose(-1, -2).flip(-2)
    # [..., D, D]

    std = eig_val.sqrt()
    # [..., D]

    return mean_x, eig_vec, std


@beartype
class Calculator:
    def __init__(
        self,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.dim = dim

        self.x_cnt = 0

        self.sum_x = torch.zeros(
            (self.dim,), dtype=dtype, device=device)

        self.sum_xxt = torch.zeros(
            (self.dim, self.dim), dtype=dtype, device=device)

    @property
    def avg_x(self) -> torch.Tensor:
        if self.x_cnt == 0:
            return self.sum_x

        return self.sum_x / self.x_cnt

    def reset(self) -> None:
        self.x_cnt = 0
        self.sum_x[:, :] = 0
        self.sum_xxt[:, :] = 0

    def feed(
        self,
        x: torch.Tensor,  # [..., D]
    ) -> None:
        D = self.dim

        utils.check_shapes(x, (..., D))

        cur_x_cnt = x.numel() / D

        if 0 < cur_x_cnt:
            self.x_cnt += cur_x_cnt
            feed(x, True, self.sum_x, self.sum_xxt)

    def get_pca(self) -> tuple[
        torch.Tensor,  # mean[D],
        torch.Tensor,  # pca[D, D],
        torch.Tensor,  # std[D],
    ]:
        assert 1 < self.x_cnt

        return get_pca(self.x_cnt, self.sum_x, self.sum_xxt)
