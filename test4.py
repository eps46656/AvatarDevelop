import numpy as np
import utils
import pickle

import torch
import sampling_utils

import pathlib

import json

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


def main1():
    with open(DIR / "smplx_param.json") as f:
        data = json.load(f)

    print(data)

    print(len(data["body_pose"]))  # 63
    print(len(data["lhand_pose"]))  # 45
    print(len(data["rhand_pose"]))  # 45
    print(len(data["shape"]))  # 10

    pass


if __name__ == "__main__":
    main1()
