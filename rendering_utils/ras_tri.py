
import math
import typing

import torch
from beartype import beartype

from .. import utils


@beartype
def rasterize_triangle(
    points: torch.Tensor,  # [3, 2]
    x_range: tuple[int, int],  # [x_min, x_max]
    y_range: tuple[int, int],  # [y_min, y_max]
) -> typing.Iterable[tuple[
    tuple[int, int],  # pixel position
    tuple[float, float, float],  # barycentric coordinate
]]:
    assert x_range[0] <= x_range[1]
    assert y_range[0] <= y_range[1]

    utils.check_shapes(points, (3, 2))

    if x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
        return

    pax, pay = points[0]
    pbx, pby = points[1]
    pcx, pcy = points[2]

    x_min = utils.clamp(float(min(pax, pbx, pcx)), x_range[0], x_range[1] - 1)
    x_max = utils.clamp(float(max(pax, pbx, pcx)), x_range[0], x_range[1] - 1)
    y_min = utils.clamp(float(min(pay, pby, pcy)), y_range[0], y_range[1] - 1)
    y_max = utils.clamp(float(max(pay, pby, pcy)), y_range[0], y_range[1] - 1)

    if x_max <= x_min or y_max <= y_min:
        return

    scan_type = 1 if x_max - x_min <= y_max - y_min else 2
    # 1 : iterate x, find y
    # 2 : iterate y, find x

    if scan_type == 1:
        pah, paw = pax, pay
        pbh, pbw = pbx, pby
        pch, pcw = pcx, pcy

        u_min, u_max = x_min, x_max
        v_min, v_max = y_min, y_max
    else:
        pah, paw = pay, pax
        pbh, pbw = pby, pbx
        pch, pcw = pcy, pcx

        u_min, u_max = y_min, y_max
        v_min, v_max = x_min, x_max

    mat = torch.tensor([
        [pbh - pah, pch - pah, pah],
        [pbw - paw, pcw - paw, paw],
        [0, 0, 1],
    ], dtype=float)

    inv_mat = torch.inverse(mat)

    c0s = [
        (1 - inv_mat[0, 2] - inv_mat[1, 2]),
        inv_mat[0, 2],
        inv_mat[1, 2],
    ]

    c1s = [
        -inv_mat[0, 1] - inv_mat[1, 1],
        inv_mat[0, 1],
        inv_mat[1, 1],
    ]

    if any(c0 < 0 and c1 == 0 for c0, c1 in zip(c0s, c1s)):
        return

    b0s = [
        None if c1s[0] == 0 else -c0s[0] / c1s[0],
        None if c1s[1] == 0 else -c0s[1] / c1s[1],
        None if c1s[2] == 0 else -c0s[2] / c1s[2],
    ]

    b1s = [
        None if c1s[0] == 0 else (inv_mat[0, 0] + inv_mat[1, 0]) / c1s[0],
        None if c1s[1] == 0 else -inv_mat[0, 0] / c1s[1],
        None if c1s[2] == 0 else -inv_mat[1, 0] / c1s[2],
    ]

    for ui in range(math.ceil(float(u_min)),
                    math.floor(float(u_max)) + 1):
        cur_v_min = v_min
        cur_v_max = v_max

        for i in range(3):
            c1 = c1s[i]

            if c1 < 0:
                cur_v_max = min(cur_v_max, float(b0s[i] + b1s[i] * ui))
            elif 0 < c1:
                cur_v_min = max(cur_v_min, float(b0s[i] + b1s[i] * ui))

        for vi in range(math.ceil(cur_v_min),
                        math.floor(cur_v_max) + 1):
            k1 = inv_mat[0, 0] * ui + inv_mat[0, 1] * vi + inv_mat[0, 2]
            k2 = inv_mat[1, 0] * ui + inv_mat[1, 1] * vi + inv_mat[1, 2]
            k0 = 1 - k1 - k2

            if scan_type == 1:
                yield (ui, vi), (k0, k1, k2)
            else:
                yield (vi, ui), (k0, k2, k1)


@beartype
def rasterize_triangle_naive(
    points: torch.Tensor,  # [3, 2]
    x_range: tuple[int, int],  # [x_min, x_max]
    y_range: tuple[int, int],  # [y_min, y_max]
) -> typing.Iterable[tuple[
    tuple[int, int],  # pixel position
    tuple[float, float, float],  # barycentric coordinate
]]:
    assert x_range[0] <= x_range[1]
    assert y_range[0] <= y_range[1]

    utils.check_shapes(points, (3, 2))

    if x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
        return

    pa = points[0]
    pb = points[1]
    pc = points[2]

    mat = torch.tensor([
        [pb[0] - pa[0], pc[0] - pa[0], pa[0]],
        [pb[1] - pa[1], pc[1] - pa[1], pa[1]],
        [0, 0, 1],
    ], dtype=float)

    inv_mat = mat.inverse()

    x_min = math.ceil(utils.clamp(
        min(pa[0], pb[0], pc[0]), x_range[0], x_range[1] - 1))
    x_max = math.floor(utils.clamp(
        max(pa[0], pb[0], pc[0]), x_range[0], x_range[1] - 1)) + 1
    y_min = math.ceil(utils.clamp(
        min(pa[1], pb[1], pc[1]), y_range[0], y_range[1] - 1))
    y_max = math.floor(utils.clamp(
        max(pa[1], pb[1], pc[1]), y_range[0], y_range[1] - 1)) + 1

    if x_max <= x_min or y_max <= y_min:
        return

    for cur_x in range(x_min, x_max):
        for cur_y in range(y_min, y_max):
            k = inv_mat @ torch.tensor([cur_x, cur_y, 1],
                                       dtype=float).unsqueeze(-1)

            kb = k[0, 0]
            kc = k[1, 0]
            ka = 1 - kb - kc

            if 0 <= ka and 0 <= kb and 0 <= kc:
                yield (cur_x, cur_y), (ka, kb, kc)
