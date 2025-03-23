import pathlib

import torch
import itertools

from beartype import beartype

from . import camera_utils, utils, transform_utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")


@beartype
def GetIdxes(shape):
    return itertools.product(*(range(s) for s in shape))


def main1():
    for i in GetIdxes(()):
        print(i)


if __name__ == "__main__":
    main1()

    print("ok")
