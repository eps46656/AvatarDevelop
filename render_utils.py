
import math

import numpy as np

import utils


def NaiveRasterizeTriangle(points: np.ndarray,
                           x_range: tuple[int, int],
                           y_range: tuple[int, int],):
    assert x_range[0] <= x_range[1]
    assert y_range[0] <= y_range[1]

    assert points.shape == (2, 3)

    if x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
        return list()

    pa = np.array(points[:, 0], dtype=float, copy=True)
    pb = np.array(points[:, 1], dtype=float, copy=True)
    pc = np.array(points[:, 2], dtype=float, copy=True)

    mat = np.array([
        [pb[0] - pa[0], pc[0] - pa[0], pa[0]],
        [pb[1] - pa[1], pc[1] - pa[1], pa[1]],
        [0, 0, 1],
    ], dtype=float)

    inv_mat = np.linalg.inv(mat)

    x_min = math.ceil(utils.Clamp(
        min(pa[0], pb[0], pc[0]), x_range[0], x_range[1] - 1))
    x_max = math.floor(utils.Clamp(
        max(pa[0], pb[0], pc[0]), x_range[0], x_range[1] - 1)) + 1
    y_min = math.ceil(utils.Clamp(
        min(pa[1], pb[1], pc[1]), y_range[0], y_range[1] - 1))
    y_max = math.floor(utils.Clamp(
        max(pa[1], pb[1], pc[1]), y_range[0], y_range[1] - 1)) + 1

    if x_max <= x_min or y_max <= y_min:
        return list()

    ret: list[tuple[int, int]] = list()

    for cur_x in range(x_min, x_max):
        for cur_y in range(y_min, y_max):
            l = inv_mat @ np.array([[cur_x], [cur_y], [1]])

            lb = l[0, 0]
            lc = l[1, 0]
            la = 1 - lb - lc

            if 0 <= la and 0 <= lb and 0 <= lc:
                ret.append((cur_x, cur_y))

    return ret


def RasterizeTriangle(points: np.ndarray,
                      x_range: tuple[int, int],
                      y_range: tuple[int, int],):
    assert x_range[0] <= x_range[1]
    assert y_range[0] <= y_range[1]

    assert points.shape == (2, 3)

    if x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
        return list()

    pax, pay = points[:, 0]
    pbx, pby = points[:, 1]
    pcx, pcy = points[:, 2]

    x_min = utils.Clamp(min(pax, pbx, pcx), x_range[0], x_range[1] - 1)
    x_max = utils.Clamp(max(pax, pbx, pcx), x_range[0], x_range[1] - 1)
    y_min = utils.Clamp(min(pay, pby, pcy), y_range[0], y_range[1] - 1)
    y_max = utils.Clamp(max(pay, pby, pcy), y_range[0], y_range[1] - 1)

    if x_max <= x_min or y_max <= y_min:
        return list()

    ret: list[tuple[int, int]] = list()

    scan_type = 1 if x_max - x_min <= y_max - y_min else 2
    # 1 : iterate x, find y
    # 2 : iterate y, find x

    if scan_type == 1:
        pah, paw = pax, pay
        pbh, pbw = pbx, pby
        pch, pcw = pcx, pcy

        h_min, h_max = x_min, x_max
        w_min, w_max = y_min, y_max
    else:
        pah, paw = pay, pax
        pbh, pbw = pby, pbx
        pch, pcw = pcy, pcx

        h_min, h_max = y_min, y_max
        w_min, w_max = x_min, x_max

    mat = np.array([
        [pbh - pah, pch - pah, pah],
        [pbw - paw, pcw - paw, paw],
        [0, 0, 1],
    ], dtype=float)

    inv_mat = np.linalg.inv(mat)

    c1s = [-inv_mat[0, 1] - inv_mat[1, 1],
           inv_mat[0, 1],
           inv_mat[1, 1],]

    c0s = [
        (1 - inv_mat[0, 2] - inv_mat[1, 2]),
        inv_mat[0, 2],
        inv_mat[1, 2],
    ]

    for i in range(3):
        if c1s[i] == 0 and c0s[i] < 0:
            return list()

    b0s = [
        -c0s[0] / c1s[0],
        -c0s[1] / c1s[1],
        -c0s[2] / c1s[2],
    ]

    b1s = [
        (inv_mat[0, 0] + inv_mat[1, 0]) / c1s[0],
        -inv_mat[0, 0] / c1s[1],
        -inv_mat[1, 0] / c1s[2],
    ]

    for cur_h in range(math.ceil(h_min), math.floor(h_max) + 1):
        cur_w_min = w_min
        cur_w_max = w_max

        for i in range(3):
            c1 = c1s[i]
            b = b0s[i] + b1s[i] * cur_h

            if c1 < 0:
                cur_w_max = min(cur_w_max, b)
            elif 0 < c1:
                cur_w_min = max(cur_w_min, b)

        if scan_type == 1:
            ret.extend((cur_h, cur_w) for cur_w in range(
                math.ceil(cur_w_min), math.floor(cur_w_max) + 1))
        else:
            ret.extend((cur_w, cur_h) for cur_w in range(
                math.ceil(cur_w_min), math.floor(cur_w_max) + 1))

    return ret
