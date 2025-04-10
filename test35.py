import dataclasses
import pathlib
import typing

import einops
import numpy as np
import torch
import tqdm
from beartype import beartype

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures

from . import (camera_utils, config, dataset_utils, face_seg_utils,
               gom_avatar_training_utils, gom_utils, people_snapshot_utils,
               rendering_utils, smplx_utils, training_utils, transform_utils,
               utils)


class AlbedoMeshShader(pytorch3d.renderer.mesh.shader.ShaderBase):
    def forward(
        self,
        fragments: pytorch3d.renderer.mesh.rasterizer.Fragments,
        meshes: pytorch3d.structures.Meshes,
        **kwargs,
    ):
        texels = meshes.sample_textures(fragments)

        print(f"{type(texels)=}")
        print(f"{texels.shape=}")

        return texels


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


def read_subject(model_data: typing.Optional[smplx_utils.ModelData] = None):
    if model_data is None:
        if "female" in SUBJECT_NAME:
            model_data_path = config.SMPL_FEMALE_MODEL_PATH
        else:
            model_data_path = config.SMPL_MALE_MODEL_PATH

        model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=model_data_path,
            model_config=smplx_utils.smpl_model_config,
            dtype=utils.FLOAT,
            device=DEVICE,
        )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME, model_data, DEVICE)

    return subject_data


def load_trainer():
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

    return trainer


def main1():
    trainer = load_trainer()

    trainer.load_latest()

    training_core: gom_avatar_training_utils.TrainingCore = \
        trainer.training_core

    dataset = training_core.dataset

    model_data: smplx_utils.ModelData = \
        training_core.module.avatar_blender.get_avatar_model()

    # ---

    obj_list = [
        "UPPER_GARMENT",
        "LOWER_GARMENT",
        "HAIR",
    ]

    mask_dir = DIR / "segment_2025_0406_01"

    masks = [
        utils.read_video(mask_dir / f"mask<{obj}>.mp4")[0].mean(1).to(DEVICE)
        for obj in obj_list
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
        for batch_idxes, sample in tqdm.tqdm(
                dataset_utils.load(dataset, batch_size=8)):
            assert len(batch_idxes) == 1

            idxes = batch_idxes[0]

            batch_size = idxes.shape[0]

            sample: gom_utils.Sample

            result: gom_utils.ModuleForwardResult = trainer.training_core.module(
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

                pixel_to_faces = (pixel_to_faces + (F + 1)) % (F + 1)
                # -1 -> F       [0, F-1] -> [0, F-1]

                for k in range(K):
                    face_seg_utils.vote(
                        pixel_to_faces,  # [H, W]
                        masks[k][idxes[i]].to(utils.FLOAT),  # [H, W]
                        face_ballot_box[k],
                    )

    utils.write_pickle(
        DIR / "face_ballot_box",
        {
            "face_ballot_box": np.array(face_ballot_box.numpy(force=True),
                                        dtype=np.float64, copy=True),
        }
    )


def main2():
    trainer = load_trainer()

    trainer.load_latest()

    training_core: gom_avatar_training_utils.TrainingCore = \
        trainer.training_core

    dataset = training_core.dataset

    smplx_model_blender: smplx_utils.ModelBlender = \
        training_core.module.avatar_blender

    smplx_model_builder: smplx_utils.DeformableModelBuilder = \
        smplx_model_blender.model_builder

    model_data: smplx_utils.ModelData = smplx_model_builder.model_data

    face_ballot_box = torch.from_numpy(
        utils.read_pickle(DIR / "face_ballot_box")["face_ballot_box"])
    # [K, F]

    face_ballot_box = face_ballot_box.to(utils.FLOAT)

    F = model_data.mesh_data.faces_cnt

    obj_list = [
        "UPPER_GARMENT",
        "LOWER_GARMENT",
        "HAIR",
    ]

    K = len(obj_list)

    obj_idx = face_seg_utils.assign(
        face_ballot_box,  # [K, F]
    ).to(utils.CPU_DEVICE)  # [F]

    face_idx_list = [[] for k in range(K)]

    for f in range(F):
        i = obj_idx[f]

        if i == -1:
            continue

        face_idx_list[i].append(f)

    for k in range(K):
        obj = obj_list[k]

        model_data_extraction_result = model_data.extract(face_idx_list[k])

        obj_model_data = model_data_extraction_result.model_data

        obj_model_data.show()

        obj_model_data.save(DIR / f"obj_model_data<{obj}>.pkl")


def main3():
    obj = "UPPER_GARMENT"

    obj_model_data: smplx_utils.ModelData = smplx_utils.ModelData.from_file(
        DIR / f"obj_model_data<{obj}>.pkl", dtype=utils.FLOAT, device=DEVICE)

    model_builder = smplx_utils.StaticModelBuilder(obj_model_data)

    model_blender = smplx_utils.ModelBlender(model_builder)

    subject_data = read_subject(obj_model_data)

    T, C, H, W = subject_data.video.shape

    albedo_map = utils.read_image(
        DIR / "train_2025_0404_2" / "tex_1743769009.png").to(DEVICE, utils.FLOAT)

    albedo_tex = pytorch3d.renderer.TexturesUV(
        maps=[einops.rearrange(albedo_map, "c h w -> h w c")],
        verts_uvs=[obj_model_data.tex_vert_pos],
        faces_uvs=[obj_model_data.tex_mesh_data.f_to_vvv],
        padding_mode="reflection",
        sampling_mode="bilinear",
    )

    subject_data.camera_transform = subject_data.camera_transform.to(
        DEVICE, utils.FLOAT)

    out_images = torch.empty((T, C, H, W), dtype=utils.FLOAT)

    for t in tqdm.tqdm(range(T)):
        camera_transform = subject_data.camera_transform
        blending_param = subject_data.blending_param[t]

        cur_model: smplx_utils.Model = model_blender(blending_param)

        print(f"{blending_param.shape=}")
        print(f"{cur_model.shape=}")
        print(f"{cur_model.vert_pos.shape=}")

        fragments = rendering_utils.rasterize_mesh(
            cur_model.vert_pos,
            cur_model.mesh_data.f_to_vvv,
            subject_data.camera_config,
            camera_transform,
            1,
        )

        # fragments.pixel_to_faces[1, H, W, 1]
        # fragments.bary_coords[1, H, W, 1, 3]

        texels = albedo_tex.sample_textures(fragments)
        # [1, H, W, 1, C]

        out_images[t] = einops.rearrange(
            texels.view(H, W, C), "h w c -> c h w")

    utils.write_video(
        DIR / f"seg_video<{obj}>.mp4",
        out_images,
        subject_data.fps,
    )


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True, True):
        with torch.no_grad():
            main3()
