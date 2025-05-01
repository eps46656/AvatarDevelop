from __future__ import annotations

import math
import typing

import torch
import torchrbf
import tqdm
from beartype import beartype

from .. import utils
from .blending_utils import get_shape_vert_dir
from .ModelData import ModelData


@beartype
def shape_canon(
    model_data: ModelData,
    body_shape: torch.Tensor,  # [..., BS]
    expr_shape: torch.Tensor,  # [..., ES]
    shaped_vert_pos: torch.Tensor,  # [... 3]
) -> tuple[
    torch.Tensor,  # vert_pos[..., 3]
    float,  # err
]:
    BS = model_data.body_shapes_cnt
    ES = model_data.expr_shapes_cnt
    V = model_data.verts_cnt

    utils.check_shapes(
        body_shape, (..., BS),
        expr_shape, (..., ES),
        shaped_vert_pos, (..., 3),
    )

    temp_vert_pos_cpu = model_data.vert_pos.cpu()

    temp_body_shape_vert_dir_cpu = \
        model_data.body_shape_vert_dir.cpu().reshape(V, 3 * BS)

    temp_expr_shape_vert_dir_cpu = \
        model_data.expr_shape_vert_dir.cpu().reshape(V, 3 * ES)

    body_shape_vert_dir_interp = torchrbf.RBFInterpolator(
        y=temp_vert_pos_cpu,  # [V, 3]
        d=temp_body_shape_vert_dir_cpu,  # [V, 3 * BS]
        smoothing=1.0,
        kernel="thin_plate_spline",
    ).to(shaped_vert_pos)

    expr_shape_vert_dir_interp = torchrbf.RBFInterpolator(
        y=temp_vert_pos_cpu,  # [V, 3]
        d=temp_expr_shape_vert_dir_cpu,  # [V, 3 * ES]
        smoothing=1.0,
        kernel="thin_plate_spline",
    ).to(shaped_vert_pos)

    temp_shaped_vert_pos = model_data.vert_pos + \
        get_shape_vert_dir(
            shape_vert_dir=model_data.body_shape_vert_dir,
            shape=body_shape,
        ) + \
        get_shape_vert_dir(
            shape_vert_dir=model_data.expr_shape_vert_dir,
            shape=expr_shape,
        )
    # [..., 3]

    inv_shaped_vert_pos_interp = torchrbf.RBFInterpolator(
        y=temp_shaped_vert_pos.cpu(),  # [V, 3]
        d=temp_vert_pos_cpu,  # [V, 3]
        smoothing=1.0,
        kernel="thin_plate_spline",
    ).to(shaped_vert_pos)

    def get_shaped_vert_pos(
        x: torch.Tensor,  # [..., 3]
    ) -> torch.Tensor:  # [..., 3 * BS]
        nonlocal body_shape
        nonlocal expr_shape

        nonlocal body_shape_vert_dir_interp
        nonlocal expr_shape_vert_dir_interp

        body_shapr_vert_dir = body_shape_vert_dir_interp(x).view(
            *x.shape[:-1], 3, BS)

        expr_shapr_vert_dir = expr_shape_vert_dir_interp(x).view(
            *x.shape[:-1], 3, ES)

        return x + \
            get_shape_vert_dir(body_shapr_vert_dir, body_shape) + \
            get_shape_vert_dir(expr_shapr_vert_dir, expr_shape)

    err = math.inf
    target_err = 1e-3
    max_iters_cnt = 100

    @beartype
    def _regress_adam(
        field: typing.Callable[[torch.Tensor], torch.Tensor],
        u: torch.Tensor,  # [..., D]
        v: torch.Tensor,  # [..., D]
    ):
        nonlocal err

        v.requires_grad = True

        assert v.shape == u.shape

        optimizer = torch.optim.Adam([v], lr=1e-4, betas=(0.5, 0.5))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 0.98 ** epoch,
        )

        iters_cnt = 0

        while target_err < err and iters_cnt <= max_iters_cnt:
            print(f"{iters_cnt=}")

            optimizer.zero_grad()

            loss = (field(v) - u).square()
            err = math.sqrt(loss.sqrt().max().item())

            loss.mean().backward()

            print(f"\t\terr: {err:.6e}")

            optimizer.step()
            scheduler.step()

            iters_cnt += 1

        return v

    vert_pos = inv_shaped_vert_pos_interp(shaped_vert_pos)
    # [..., 3]

    vert_pos = _regress_adam(get_shaped_vert_pos, shaped_vert_pos, vert_pos)

    vert_pos = vert_pos.detach()
    vert_pos.requires_grad = False

    return vert_pos, err
