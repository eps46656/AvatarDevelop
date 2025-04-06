import dataclasses
import pathlib

import torch
from beartype import beartype

from . import (camera_utils, config, dataset_utils, face_seg_utils,
               gom_avatar_training_utils, gom_utils, people_snapshot_utils,
               rendering_utils, smplx_utils, training_utils, transform_utils,
               utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")

PROJ_DIR = DIR / "train_2025_0404_2"

ALPHA_RGB = 1.0
ALPHA_LAP_SMOOTHING = 10.0
ALPHA_NOR_SIM = 1.0
ALPHA_COLOR_DIFF = 1.0

BATCH_SIZE = 4

LR = 1e-3

SUBJECT_NAME = "female-1-casual"


def read_subject():
    smpl_model_data_path_dict = {
        "male": config.SMPL_MALE_MODEL,
        "female": config.SMPL_FEMALE_MODEL,
        "neutral": config.SMPL_NEUTRAL_MODEL,
    }

    smplx_model_data_path_dict = {
        "male": config.SMPLX_MALE_MODEL,
        "female": config.SMPLX_FEMALE_MODEL,
        "neutral": config.SMPLX_NEUTRAL_MODEL,
    }

    model_data_dict: dict[str, smplx_utils.ModelData] = {
        key: smplx_utils.ModelData.from_file(
            model_data_path=value,
            model_config=smplx_utils.smpl_model_config,
            device=DEVICE,
        )
        for key, value in smpl_model_data_path_dict.items()
    }

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME, model_data_dict, DEVICE)

    return subject_data


def main1():
    subject_data = read_subject()

    # ---

    dataset = gom_utils.Dataset(gom_utils.Sample(
        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,
        img=subject_data.video,
        mask=subject_data.mask,
        blending_param=subject_data.blending_param,
    )).to(DEVICE)

    # ---

    smplx_model_builder = smplx_utils.DeformableModelBuilder(
        model_data=subject_data.model_data,
    ).to(DEVICE)

    smplx_model_builder.unfreeze()

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    module = gom_utils.Module(
        avatar_blender=smplx_model_blender,
        color_channels_cnt=3,
    ).to(DEVICE).train()

    # ---

    param_groups = utils.get_param_groups(module, LR)

    optimizer = None

    scheduler = None

    training_core = gom_avatar_training_utils.TrainingCore(
        config=gom_avatar_training_utils.Config(
            proj_dir=PROJ_DIR,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            alpha_rgb=ALPHA_RGB,
            alpha_lap_smoothing=ALPHA_LAP_SMOOTHING,
            alpha_nor_sim=ALPHA_NOR_SIM,
            alpha_color_diff=ALPHA_COLOR_DIFF,
        ),
        module=module,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # ---

    trainer = training_utils.Trainer(
        proj_dir=PROJ_DIR,
        device=DEVICE,
    )

    trainer.training_core = training_core

    # ---

    trainer.load_latest()

    model_data: smplx_utils.ModelData = \
        trainer.training_core.module.avatar_blender.get_avatar_model()

    # ---

    mask_dir = ""

    masks = [
        utils.read_video(mask_dir / "mask<UPPER_GARMENT>.mp4")[0].mean(1),
        utils.read_video(mask_dir / "mask<LOWER_GARMENT>.mp4")[0].mean(1),
        utils.read_video(mask_dir / "mask<HAIR>.mp4")[0].mean(1),
    ]

    # [K][T, H, W]

    K = len(masks)

    F = model_data.mesh_data.faces_cnt

    face_ballot_box = torch.zeros(
        (K, F + 1),  # F + 1 for -1 pixel to face index
        dtype=utils.FLOAT,
        device=DEVICE,
    )

    with torch.no_grad():
        for batch_idxes, sample in dataset_utils.load(dataset):
            assert len(batch_idxes) == 1

            idxes = batch_idxes[0]

            batch_size = idxes.shape[0]

            sample: gom_utils.Sample

            result: gom_utils.Module.ForwardResult = trainer.training_core.module(
                camera_config=sample.camera_config,
                camera_transform=sample.camera_transform,
                img=sample.img,
                mask=sample.mask,
                blending_param=sample.blending_param,
            )

            for i in range(batch_size):
                cur_avatar_model = result.avatar_model[i]

                mesh_ras_result = rendering_utils.rasterize_mesh(
                    vert_pos=cur_avatar_model.vert_pos,
                    faces=cur_avatar_model.mesh_data.f_to_vvv,
                    camera_config=sample.camera_config,
                    camera_transform=sample.camera_transform[i],
                    faces_per_pixel=1,
                )

                pixel_to_faces = mesh_ras_result.pixel_to_faces
                # [H, W, 1]

                pixel_to_faces = pixel_to_faces.squeeze(-1)
                # [H, W]

                for k in range(K):
                    face_seg_utils.vote(
                        pixel_to_faces,  # [H, W]
                        masks[k][idxes[i]],  # [H, W]
                        face_ballot_box[k],
                    )

    pass


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True, True):
        main1()
