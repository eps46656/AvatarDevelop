
import pathlib
import typing

import torch

from . import rendering_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


def main1():
    H = 720
    W = 1280

    img = torch.zeros((3, H, W), dtype=utils.FLOAT)

    for _ in range(10):
        points = torch.rand(3, 2)

        points[0, :] *= H
        points[1, :] *= W

        timer_a = utils.Timer()
        timer_b = utils.Timer()

        with timer_a:
            it = rendering_utils.rasterize_triangle_naive(
                points, (0, H), (0, W))

            naive_bcs: dict[
                tuple[int, int],
                tuple[float, float, float]
            ] = dict()

            for naive_pixel, naive_bc in it:
                assert naive_pixel not in naive_bcs
                naive_bcs[naive_pixel] = naive_bc

        with timer_b:
            it = rendering_utils.rasterize_triangle(points, (0, H), (0, W))

            bcs: dict[
                tuple[int, int],
                tuple[float, float, float]
            ] = dict()

            for pixel,  bc in it:
                assert pixel not in bcs
                bcs[pixel] = bc

        print(f"{timer_b.duration / timer_a.duration * 100=}")

        intersection_pixels = naive_bcs.keys() & bcs.keys()

        ratio = 1 - len(intersection_pixels) / (
            len(naive_bcs) + len(bcs) - len(intersection_pixels))

        print(f"{len(naive_bcs)=}")
        print(f"{len(bcs)=}")
        print(f"{len(intersection_pixels)=}")
        print(f"{ratio*100=}")

        for pixel in bcs:
            img[0, pixel[0], pixel[1]] = 1

        assert ratio <= 0.04

    utils.write_image(DIR / "test.png", img)


if __name__ == "__main__":
    main1()
