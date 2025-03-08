
import pathlib

import numpy as np

import AbstractMesh
import utils
import rendering_utils
import typing

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


def IsUnique(x: typing.Iterable[object]):
    ret = set()

    for obj in x:
        assert utils.SetAdd(ret, obj)

    return ret


def main1():
    H = 720
    W = 1280

    img = np.zeros((H, W, 3), dtype=np.uint8)

    for _ in range(1):
        center = np.random.rand(2)

        center[0] *= H
        center[1] *= W

        axis_u = np.random.rand(2) * 16
        axis_y = np.random.rand(2) * 16

        timer_a = utils.Timer()

        with timer_a:
            naive_pixels = IsUnique(
                rendering_utils.RasterizeEllipseWithAxis(
                    center, axis_u, axis_y, (0, H), (0, W), True))

        for pixel in naive_pixels:
            img[pixel[0], pixel[1], :] = (255, 0, 0)

    print(f"{img.shape=}")

    utils.WriteImage(DIR / "test.png", img)


def main2():
    H = 720
    W = 1280

    img = np.zeros((H, W, 3), dtype=np.uint8)

    for _ in range(128):
        center = np.random.rand(2)

        center[0] *= H
        center[1] *= W

        axis_u = (np.random.rand(2) * 2 - 1) * 16
        axis_y = (np.random.rand(2) * 2 - 1) * 16

        timer_a = utils.Timer()
        timer_b = utils.Timer()

        with timer_a:
            naive_pixels = IsUnique(
                rendering_utils.NaiveRasterizeEllipseWithAxis(
                    center, axis_u, axis_y, (0, H), (0, W)))

        with timer_b:
            pixels = IsUnique(
                rendering_utils.RasterizeEllipseWithAxis(
                    center, axis_u, axis_y, (0, H), (0, W)))

        boost_rate = timer_a.duration() / (timer_b.duration() + 1e-3)

        print(f"{boost_rate=}")

        intersection_pixels = naive_pixels.intersection(pixels)

        ratio = 1 - len(intersection_pixels) / (
            1e-3 + len(naive_pixels) + len(pixels) - len(intersection_pixels))

        print(f"{ratio*100=}")

        for (x, y), (u, v) in pixels:
            img[x, y, :] = (round(255 * (u**2 + v**2)), 0, 0)

        assert ratio <= 0.04

    utils.WriteImage(DIR / "test.png", img)


if __name__ == "__main__":
    main2()
