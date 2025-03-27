import dataclasses
import json
import math
import pathlib
import pickle

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch

import blending_utils
import camera_utils
import config
from smplx import smplx
import utils
from kin_utils import KinTree

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


INT = torch.int32
FLOAT = torch.float32
DEVICE = utils.CPU_DEVICE


def main1():
    pass


if __name__ == "__main__":
    main1()
