import pathlib

import numpy as np

import AbstractMesh
import utils
import rendering_utils
import typing
import torch
import sampling_utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


def main1():
    mesh = AbstractMesh.AbstractMesh()

    with utils.Timer():
        va = mesh.AddVertex("va")

    with utils.Timer():
        vb = mesh.AddVertex("vb")

    with utils.Timer():
        vc = mesh.AddVertex("vb")

    with utils.Timer():
        fa, _ = mesh.AddFace((va, vb, vc), "fa")

    print(f"{va=}")
    print(f"{vb=}")
    print(f"{vc=}")
    print(f"{fa=}")


def main2():
    img = torch.Tensor(utils.read_image(DIR / "origin.png", "chw"))
    # [c, h, w]

    scale_factor = 1.3

    tex_sampler = sampling_utils.TextureSampler(
        2,
        img,
        sampling_utils.WrapModeEnum.MIRROR_REPEAT,
        sampling_utils.InterpModeEnum.CUBIC,
    )

    c, h, w = img.shape

    print(f"{h=}")
    print(f"{w=}")
    print(f"{c=}")

    dst_h = int(h * scale_factor)
    dst_w = int(w * scale_factor)

    points = torch.empty(dst_h, dst_w, 2)

    for i in range(dst_h):
        for j in range(dst_w):
            points[i, j, 0] = i / (dst_h - 1) + 0.3
            points[i, j, 1] = j / (dst_w - 1) - 0.2

    dst_img = tex_sampler.Sample(points)
    # [c, dst_h, dst_w]

    print(f"{dst_img.shape}")

    vision_utils.write_image(DIR / "new.png", dst_img.numpy(), "chw")


if __name__ == "__main__":
    main2()
