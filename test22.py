import copy
import itertools
import pathlib

import torch
from beartype import beartype

from . import (utils, training_utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")


def main1():
    checkpoint_record_path = DIR / "test_checkpoint_record.json"

    init_checkpoint, checkpoint_records = training_utils.LoadCheckpointRecords(
        checkpoint_record_path)

    print(f"{checkpoint_records}")

    checkpoint_records[999] = training_utils.CheckpointMeta(
        timestamp=999,
        prv=1,
        epoch=5,
        message="",
    )

    training_utils.SaveCheckpointRecords(
        checkpoint_records,
        checkpoint_record_path
    )


if __name__ == "__main__":
    main1()
