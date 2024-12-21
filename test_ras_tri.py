
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

    for _ in range(10):
        points = np.random.rand(2, 3)

        points[0, :] *= H
        points[1, :] *= W

        timer_a = utils.Timer()
        timer_b = utils.Timer()

        with timer_a:
            naive_pixels = IsUnique(
                rendering_utils.NaiveRasterizeTriangle(points, (0, H), (0, W)))

        with timer_b:
            pixels = IsUnique(
                rendering_utils.RasterizeTriangle(points, (0, H), (0, W)))

        print(f"{timer_b.duration() / timer_a.duration() * 100=}")

        intersection_pixels = naive_pixels.intersection(pixels)

        ratio = 1 - len(intersection_pixels) / (
            len(naive_pixels) + len(pixels) - len(intersection_pixels))

        print(f"{ratio*100=}")

        for pixel in pixels:
            img[pixel[0], pixel[1], :] = (255, 0, 0)

        assert ratio <= 0.04

    utils.WriteImage(DIR / "test.png", img)


if __name__ == "__main__":
    main1()
