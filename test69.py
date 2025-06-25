import torch
import typing
from beartype import beartype

from . import (config, smplx_utils, tex_avatar_utils, training_utils,
               transform_utils, utils, vision_utils, people_snapshot_utils)

DTYPE = torch.float64
DEVICE = torch.device("cuda")

SUBJECT_NAME = "female-3-casual"


def read_subject(model_data: typing.Optional[smplx_utils.ModelData] = None):
    if model_data is None:
        model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=DTYPE,
            device=DEVICE,
        )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        model_data,
        DTYPE,
        utils.CPU_DEVICE,
    )

    return subject_data


def main1():
    pass


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True, True):
        main1()

    print("ok")
