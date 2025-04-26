
import os
import pathlib

import matplotlib.pyplot as plt
import torch
import trimesh

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


def get_subdirs(path: os.PathLike) -> list[os.PathLike]:
    return [name for name in utils.to_pathlib_path(path).glob("*") if name.is_dir()]


def get_train_dir(train_base_dir: os.PathLike, train_name: str) -> os.PathLike:
    dirs = get_subdirs(train_base_dir / train_name)

    assert len(dirs) == 1

    return dirs[0]


def main1():
    k_dir = "seq=female-3-casual_prof=people_2m_data=people_snapshot"

    train_name = "train_2025_0426_1"

    train_base_dir = DIR / f"GART/logs"

    print(f"{train_base_dir=}")

    train_dir = get_train_dir(train_base_dir, train_name)

    print(f"{train_dir=}")

    state_dict = torch.load(train_dir / "model.pth")

    for key, val in state_dict.items():
        print(f"{key}: {val.shape}    {val.dtype}")

    vert_pos = state_dict["_xyz"].cpu().reshape(-1, 3)
    V = vert_pos.shape[0]

    scene = trimesh.Scene()

    cloud = trimesh.points.PointCloud(vert_pos)

    scene.add_geometry(cloud)

    scene.show()


if __name__ == "__main__":
    main1()
