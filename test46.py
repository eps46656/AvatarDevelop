import pathlib

import torch
from beartype import beartype

from . import config, sdf_utils, smplx_utils, training_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DTYPE = torch.float32
DEVICE = torch.device("cuda")

PROJ_DIR = DIR / "sdf_train_2025_0503_2"

BATCH_SIZE = 8

MODEL_TYPE = "smplx"
MODEL_SUBTYPE = "female"


def main1():
    torch.autograd.set_detect_anomaly(True, True)

    # ---

    model_data: smplx_utils.ModelData = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_FEMALE_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        device=DEVICE,
    )

    range_min = (-2.0, -2.0, -2.0)
    range_max = (+2.0, +2.0, +2.0)

    module = sdf_utils.Module(
        range_min=range_min,
        range_max=range_max,
        dtype=DTYPE,
        device=DEVICE,
    ).train()

    dataset = sdf_utils.Dataset(
        mean=(
            (range_min[0] + range_max[0]) / 2,
            (range_min[1] + range_max[1]) / 2,
            (range_min[2] + range_max[2]) / 2,
        ),

        std=(
            (range_max[0] - range_min[0]) / 4,
            (range_max[1] - range_min[1]) / 4,
            (range_max[2] - range_min[2]) / 4,
        ),

        epoch_size=1024,
        mesh_graph=model_data.mesh_graph,
        vert_pos=model_data.vert_pos,

        dtype=DTYPE,
        device=DEVICE,
    )

    lr = 1e-3

    trainer = training_utils.Trainer(
        proj_dir=PROJ_DIR,
        trainer_core=sdf_utils.TrainerCore(
            sdf_utils.TrainerCoreConfig(
                proj_dir=PROJ_DIR,
                device=DEVICE,
                batch_size=BATCH_SIZE,

                lr=lr,
                gamma=0.1**(1 / 100),
                betas=(0.5, 0.5),
            ),

            module=module,
            dataset=dataset,
        ),
    )

    trainer.enter_cli()


if __name__ == "__main__":
    main1()

    print("ok")
