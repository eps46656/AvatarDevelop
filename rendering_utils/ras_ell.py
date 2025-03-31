
import math

import numpy as np

import utils

import torch

from typeguard import beartype


"""

(c_y2) * y**2 +
(c_xy * x + c_y) * y +
(c_x2 * x0**2 + c_x * x + c_1)
=


det = (c_xy * x + c_y)**2 - 4 * (c_y2) * (c_x2 * x0**2 + c_x * x + c_1)

"""


@beartype
def NaiveRasterizeEllipseWithCoeff(
    c_x2: float,
    c_xy: float,
    c_y2: float,
    c_x: float,
    c_y: float,
    c_1: float,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
):
    for x0 in range(x_range[0], x_range[1]):
        for y0 in range(y_range[0], y_range[1]):
            r = c_x2 * x0**2 + c_xy * x0 * y0 + c_y2 * y0**2 + c_x * x0 + c_y * y0 + c_1

            if r <= 0:
                yield (x0, y0)


"""

(c_y2) * y**2 +
(c_xy * x + c_y) * y +
(c_x2 * x**2 + c_x * x + c_1) +


(c_xy * x + c_y)**2 == 4 * (c_y2) * (c_x2 * x**2 + c_x * x + c_1)

(c_xy**2) * x**2
(2 * c_xy * c_y) * x
(c_y**2)
==
(4 * c_y2 * c_x2) * x**2
(4 * c_y2 * c_x) * x
(4 * c_y2 * c_1)


"""


@beartype
def GetEllipseRange(
    c_x2: float,
    c_xy: float,
    c_y2: float,
    c_x: float,
    c_y: float,
    c_1: float,
):
    a = c_xy**2 - 4 * c_y2 * c_x2
    b = 2 * c_xy * c_y - 4 * c_y2 * c_x
    c = c_y**2 - 4 * c_y2 * c_1

    det = b**2 - 4 * a * c
    assert 0 <= det

    k = math.sqrt(det)

    root_a = (-b - k) / (2 * a)
    root_b = (-b + k) / (2 * a)

    return min(root_a, root_b), max(root_a, root_b)


@beartype
def RasterizeEllipseWithCoeff(
    c_x2: float,
    c_xy: float,
    c_y2: float,
    c_x: float,
    c_y: float,
    c_1: float,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
):
    assert x_range[0] <= x_range[1]
    assert y_range[0] <= y_range[1]

    if x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
        return

    assert c_xy**2 - 4 * c_x2 * c_y2 < 0

    cur_x_min, cur_x_max = GetEllipseRange(c_x2, c_xy, c_y2, c_x, c_y, c_1)
    cur_y_min, cur_y_max = GetEllipseRange(c_y2, c_xy, c_x2, c_y, c_x, c_1)

    x_min = utils.clamp(cur_x_min, x_range[0], x_range[1] - 1)
    x_max = utils.clamp(cur_x_max, x_range[0], x_range[1] - 1)
    y_min = utils.clamp(cur_y_min, y_range[0], y_range[1] - 1)
    y_max = utils.clamp(cur_y_max, y_range[0], y_range[1] - 1)

    scan_type = 1 if x_max - x_min <= y_max - y_min else 2

    if scan_type == 1:
        c_h2, c_hw, c_w2, c_h, c_w = c_x2, c_xy, c_y2, c_x, c_y
        h_min, h_max = x_min, x_max
        w_min, w_max = y_min, y_max
    else:
        c_h2, c_hw, c_w2, c_h, c_w = c_y2, c_xy, c_x2, c_y, c_x
        h_min, h_max = y_min, y_max
        w_min, w_max = x_min, x_max

    for h0 in range(math.ceil(h_min), math.floor(h_max) + 1):
        a = c_w2
        b = c_hw * h0 + c_w
        c = c_h2 * h0**2 + c_h * h0 + c_1

        cur_det = b**2 - 4 * a * c

        if cur_det < 0:
            continue

        k = math.sqrt(cur_det)

        cur_w_min = max((-b - k) / (2 * a), w_min)
        cur_w_max = min((-b + k) / (2 * a), w_max)

        w_generator = range(math.ceil(cur_w_min), math.floor(cur_w_max) + 1)

        if scan_type == 1:
            yield from ((h0, w0) for w0 in w_generator)
        else:
            yield from ((w0, h0) for w0 in w_generator)

    """
    for h0 in range(math.ceil(h_min), math.floor(h_max) + 1):
        for w0 in range(math.ceil(w_min), math.floor(w_max) + 1):
            r = c_h2 * h0**2 + c_xy * h0 * w0 + c_w2 * w0**2 + c_h * h0 + c_w * w0 + c_1

            if 0 < r:
                continue

            if scan_type == 1:
                yield (h0, w0)
            else:
                yield (w0, h0)
    """


@beartype
def TransEllipseAxisToCoeff(
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
):
    assert center.shape == (2,)
    assert axis_u.shape == (2,)
    assert axis_v.shape == (2,)

    cx, cy = float(center[0]), float(center[1])

    aux, auy = float(axis_u[0]), float(axis_u[1])
    avx, avy = float(axis_v[0]), float(axis_v[1])

    A = np.linalg.inv(np.array([
        [aux, avx],
        [auy, avy],
    ], dtype=float))

    b = A @ np.array([[-cx], [-cy]], dtype=float)

    ux, uy, uc = A[0, 0], A[0, 1], b[0, 0]
    vx, vy, vc = A[1, 0], A[1, 1], b[1, 0]

    c_x2 = ux**2 + vx**2
    c_xy = 2 * ux * uy + 2 * vx * vy
    c_y2 = uy**2 + vy**2
    c_x = 2 * ux * uc + 2 * vx * vc
    c_y = 2 * uy * uc + 2 * vy * vc
    c_1 = uc**2 + vc**2 - 1

    return A, b, c_x2, c_xy, c_y2, c_x, c_y, c_1


@beartype
def RasterizeEllipseWithAxis(
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
):
    A, b, c_x2, c_xy, c_y2, c_x, c_y, c_1 = TransEllipseAxisToCoeff(
        center, axis_u, axis_v)

    gen = RasterizeEllipseWithCoeff(
        c_x2, c_xy, c_y2, c_x, c_y, c_1, x_range, y_range)

    for x, y in gen:
        u = A[0, 0] * x + A[0, 1] * y + b[0, 0]
        v = A[1, 0] * x + A[1, 1] * y + b[1, 0]

        yield (x, y), (u, v)


@beartype
def NaiveRasterizeEllipseWithAxis(
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
):
    A, b, c_x2, c_xy, c_y2, c_x, c_y, c_1 = TransEllipseAxisToCoeff(
        center, axis_u, axis_v)

    gen = NaiveRasterizeEllipseWithCoeff(
        c_x2, c_xy, c_y2, c_x, c_y, c_1, x_range, y_range)

    for x, y in gen:
        u = A[0, 0] * x + A[0, 1] * y + b[0, 0]
        v = A[1, 0] * x + A[1, 1] * y + b[1, 0]

        yield (x, y), (u, v)
