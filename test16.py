import pathlib

import torch
import h5py
import pickle
import dataclasses

from . import people_snapshot_utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")


@beartype
@dataclasses.dataclass
class PeopleSnapshotData:
    pass


def main1():
    people_snapshot_dir = DIR / "people_snapshot_public"

    subject_name = "female-1-casual"

    subject_dir = people_snapshot_dir / subject_name

    print(f"{subject_dir / "camera.pkl"=}")

    camera_f = read_pickle(subject_dir / "camera.pkl")

    K, R, T, D = get_KRTD(camera_f)

    print(f"{K=}")
    print(f"{R=}")
    print(f"{T=}")
    print(f"{D=}")


def main2():
    people_snapshot_dir = DIR / "people_snapshot_public"

    subject_name = "female-1-casual"

    subject_dir = people_snapshot_dir / subject_name

    people_snapshot_utils.ReadSubject(subject_dir)


if __name__ == "__main__":
    main2()
