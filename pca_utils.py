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
    idx: torch.Tensor,  # [...]
    w: torch.Tensor,  # [...]
    x: torch.Tensor,  # [..., D]
    inplace: bool,

    dst_sum_w: torch.Tensor,  # [N]
    dst_sum_sq_w: torch.Tensor,  # [N]
    dst_sum_w_x: torch.Tensor,  # [N, D]
    dst_sum_w_xxt: torch.Tensor,  # [N, D, D]
) -> tuple[
    torch.Tensor,  # dst_sum_w[N]
    torch.Tensor,  # dst_sum_sq_w[N]
    torch.Tensor,  # dst_sum_w_x[N, D]
    torch.Tensor,  # dst_sum_w_xxt[N, D, D]
]:
    N, D = -1, -2

    N, D = utils.check_shapes(
        idx, (...,),
        w, (...,),
        x, (..., D),

        dst_sum_w, (N,),
        dst_sum_w_x, (N, D),
        dst_sum_w_xxt, (N, D, D),
    )

    shapes = utils.broadcast_shapes(
        idx,
        w,
        x.shape[:-1],
    )

    B = shapes.numel()

    idx = idx.expand(shapes).reshape(B)
    w = w.expand(shapes).reshape(B)
    x = x.expand(*shapes, D).reshape(B, D)

    w_x = w[..., None] * x

    w_xxt = w[..., None, None] * (x[..., :, None] @ x[..., None, :])
    # [B, D, D]

    sq_w = w.square()

    if inplace:
        dst_sum_w.index_add_(
            0, idx, w.to(dst_sum_w))

        dst_sum_sq_w.index_add_(
            0, idx, sq_w.to(dst_sum_sq_w))

        dst_sum_w_x.index_add_(
            0, idx, w_x.to(dst_sum_w_x))

        dst_sum_w_xxt.index_add_(
            0, idx, w_xxt.to(dst_sum_w_xxt))
    else:
        dst_sum_w = dst_sum_w.index_add(
            0, idx, w.to(dst_sum_w))

        dst_sum_sq_w.index_add_(
            0, idx, sq_w.to(dst_sum_sq_w))

        dst_sum_w_x = dst_sum_w_x.index_add(
            0, idx, w_x.to(dst_sum_w_x))

        dst_sum_w_xxt = dst_sum_w_xxt.index_add(
            0, idx, w_xxt.to(dst_sum_w_xxt))

    return dst_sum_w, dst_sum_sq_w, dst_sum_w_x, dst_sum_w_xxt


@beartype
def get_pca(
    sum_w: torch.Tensor,  # [...]
    sum_sq_w: torch.Tensor,  # [...]
    sum_w_x: torch.Tensor,  # [..., D]
    sum_w_xxt: torch.Tensor,  # [..., D, D]
    *,
    biased: bool = False,
) -> tuple[
    torch.Tensor,  # mean[..., D]
    torch.Tensor,  # pca[..., D, D]
    torch.Tensor,  # std[..., D]
]:
    utils.check_shapes(
        sum_w, (...,),
        sum_sq_w, (...,),
        sum_w_x, (..., -1),
        sum_w_xxt, (..., -1, -1),
    )

    sum_w = 1e-6 + sum_w

    mean_x = sum_w_x / sum_w[..., None]
    # [..., D]

    eff_w = sum_w if biased else sum_w - sum_sq_w / sum_w

    cov = (sum_w_xxt - mean_x[..., :, None] @ sum_w_x[..., None, :]) \
        / eff_w[..., None, None]
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
class PCACalculator:
    def __init__(
        self,
        n: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.sum_w = torch.zeros(
            (n,), dtype=dtype, device=device)

        self.sum_sq_w = torch.zeros(
            (n,), dtype=dtype, device=device)

        self.sum_w_x = torch.zeros(
            (n, dim), dtype=dtype, device=device)

        self.sum_w_xxt = torch.zeros(
            (n, dim, dim), dtype=dtype, device=device)

    def scatter_feed(
        self,
        idx: torch.Tensor,  # [...]
        w: torch.Tensor,  # [...]
        x: torch.Tensor,  # [..., D]
    ) -> None:
        scatter_feed(
            idx,
            w,
            x,
            True,
            self.sum_w,
            self.sum_sq_w,
            self.sum_w_x,
            self.sum_w_xxt,
        )

    def get_pca(self, biased: bool) -> tuple[
        torch.Tensor,  # mean[N, D]
        torch.Tensor,  # pca[N, D, D]
        torch.Tensor,  # std[N, D]
    ]:
        return get_pca(
            self.sum_w,
            self.sum_sq_w,
            self.sum_w_x,
            self.sum_w_xxt,
            biased=biased,
        )
