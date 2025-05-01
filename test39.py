
import pathlib

import matplotlib.pyplot as plt
import torch

from . import (config, people_snapshot_utils, segment_utils, smplx_utils,
               utils, vision_utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")

SUBJECT_NAME = "female-1-casual"


def print_tensor(x: torch.Tensor):
    print(f"[")

    for val in x.flatten().tolist():
        print(f"{val:+.6e}f", end=", ")

    print(f"]")


def main1():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPLX_NEUTRAL_MODEL_PATH,
        model_config=smplx_utils.smplx_model_config,
        dtype=utils.FLOAT,
        device=DEVICE,
    )

    print_tensor(model_data.lhand_pose_mean)
    print_tensor(model_data.rhand_pose_mean)


if __name__ == "__main__":
    main1()
