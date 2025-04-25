import dataclasses
import math
import pathlib
import typing

import einops
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
import tqdm
from beartype import beartype

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures

from . import (avatar_utils, camera_utils, config, dataset_utils,
               face_seg_utils, gom_avatar_training_utils, gom_utils,
               kernel_splatting_utils, mesh_utils, people_snapshot_utils,
               rendering_utils, segment_utils, smplx_utils, training_utils,
               transform_utils, utils, vision_utils)


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

PROJ_DIR = DIR / "train_2025_0422_3"

VERT_GRAD_NORM_THRESHOLD = 1e-3

ALPHA_RGB = 1.0
ALPHA_LAP_SMOOTHNESS = 1000.0
ALPHA_NOR_SIM = 10.0
ALPHA_EDGE_VAR = 1.0
ALPHA_COLOR_DIFF = 1.0
ALPHA_GP_SCALE_DIFF = 1.0

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


def create_optimizer(param_groups):
    return torch.optim.Adam(
        param_groups,
        lr=LR,
        betas=(0.5, 0.5),
    )


def create_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=pow(0.1, 1/4),
        patience=5,
        threshold=0.05,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-7,
    )


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

            lr=LR,

            vert_grad_norm_threshold=VERT_GRAD_NORM_THRESHOLD,

            alpha_rgb=ALPHA_RGB,
            alpha_lap_smoothness=ALPHA_LAP_SMOOTHNESS,
            alpha_nor_sim=ALPHA_NOR_SIM,
            alpha_edge_var=ALPHA_EDGE_VAR,
            alpha_color_diff=ALPHA_COLOR_DIFF,
            alpha_gp_scale_diff=ALPHA_GP_SCALE_DIFF,
        ),
        module=module,
        dataset=dataset,
        optimizer_factory=create_optimizer,
        scheduler_factory=create_scheduler,
    )

    # ---

    trainer = training_utils.Trainer(
        proj_dir=PROJ_DIR,
        device=DEVICE,
    )

    trainer.training_core = training_core

    return subject_data, trainer


def main1():
    subject_data, trainer = load_trainer()

    trainer.load_latest()

    training_core: gom_avatar_training_utils.TrainingCore = \
        trainer.training_core

    dataset = training_core.dataset

    model_data: smplx_utils.ModelData = \
        training_core.module.avatar_blender.get_avatar_model()

    # ---

    obj_list = [
        segment_utils.ObjectType.UPPER_GARMENT,
        segment_utils.ObjectType.LOWER_GARMENT,
        segment_utils.ObjectType.HAIR,
    ]

    mask_dir = DIR / "segment_2025_0412_1"

    masks = [vision_utils.read_video_mask(
        mask_dir / segment_utils.blurred_object_mask_filename(obj),
        dtype=torch.float32,
        device=DEVICE,
    )[0] for obj in obj_list]

    # [K][T, H, W]

    gom_module: gom_utils.Module = trainer.training_core.module

    K = len(masks)

    avatar_model: avatar_utils.AvatarModel = gom_module.avatar_blender.get_avatar_model()

    face_segmentor = face_seg_utils.FaceSegmentor.from_mesh_graph(
        avatar_model.mesh_data.mesh_graph,
        len(obj_list),
        torch.float64,
        DEVICE
    )

    T, C, H, W = dataset.sample.img.shape

    albedo_tex = pytorch3d.renderer.TexturesUV(
        maps=[
            einops.rearrange(subject_data.tex,
                             "c h w -> h w c").to(utils.FLOAT)
        ],
        verts_uvs=[model_data.tex_vert_pos.to(utils.FLOAT)],
        faces_uvs=[model_data.tex_mesh_graph.f_to_vvv],
        padding_mode="reflection",
        sampling_mode="bilinear",
    )

    out_images = torch.empty((T, C, H, W), dtype=torch.uint8)

    with torch.no_grad():
        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(
                dataset, batch_size=8, shuffle=False)):
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

                # cur_avatar_model.mesh_graph.show(cur_avatar_model.vert_pos)

                mesh_ras_result = rendering_utils.rasterize_mesh(
                    vert_pos=cur_avatar_model.vert_pos,
                    faces=cur_avatar_model.mesh_graph.f_to_vvv,
                    camera_config=sample.camera_config,
                    camera_transform=sample.camera_transform[i],
                    faces_per_pixel=1,
                )

                pix_to_face: torch.Tensor = \
                    mesh_ras_result.pix_to_face.reshape(H, W)
                # [H * W]

                bary_coord: torch.Tensor = \
                    mesh_ras_result.bary_coords.reshape(H, W, 3)
                # [H, W, 3]

                texels = albedo_tex.sample_textures(mesh_ras_result)
                # [1, H, W, 1, C]

                out_images[idxes[i]] = einops.rearrange(
                    texels.view(H, W, C), "h w c -> c h w")

                for k in range(K):
                    cur_mask = masks[k][idxes[i]].to(utils.FLOAT)

                    ballot = cur_mask * 2 - 1

                    face_segmentor.vote(
                        k,
                        pix_to_face,  # [H, W]
                        bary_coord,  # [H, W, 3]
                        ballot,  # [H, W]
                    )

    vision_utils.write_video(
        PROJ_DIR / f"output_video_{utils.timestamp_sec()}.avi",
        out_images,
        subject_data.fps,
    )

    utils.write_pickle(
        PROJ_DIR / f"face_voting_result_{utils.timestamp_sec()}.pkl",
        face_segmentor.state_dict(),
    )


def main2():
    subject_data, trainer = load_trainer()

    trainer.load_latest()

    training_core: gom_avatar_training_utils.TrainingCore = \
        trainer.training_core

    smplx_model_blender: smplx_utils.ModelBlender = \
        training_core.module.avatar_blender

    smplx_model_builder: smplx_utils.DeformableModelBuilder = \
        smplx_model_blender.model_builder

    model_data: smplx_utils.ModelData = smplx_model_builder.model_data

    face_segmentor: face_seg_utils.FaceSegmentor = \
        face_seg_utils.FaceSegmentor.from_state_dict(utils.read_pickle(
            PROJ_DIR / "face_voting_result_1745335872.pkl"),
            None,
            DEVICE,
        )

    vert_ballot_cnt = face_segmentor.vert_ballot_cnt
    # [V + 1, K]

    vert_ballot_box = face_segmentor.vert_ballot_box
    # [V + 1, K]

    std = 50 * 1e-3
    # 50 mm = 5 cm

    def kernel(x):
        return (-(x / std)**2 / 2).exp()

    vert_ballot_cnt[:-1] = kernel_splatting_utils.interp(
        model_data.vert_pos,  # [V, 3]
        vert_ballot_cnt[:-1],  # [V, K]
        model_data.vert_pos,  # [V, 3]
        kernel,
    )

    vert_ballot_box[:-1] = kernel_splatting_utils.interp(
        model_data.vert_pos,  # [V, 3]
        vert_ballot_box[:-1],  # [V, K]
        model_data.vert_pos,  # [V, 3]
        kernel,
    )

    """
    for i in range(3):
        vert_ballot_cnt[:-1] = model_data.mesh_graph.vert_lap_trans(
            vert_ballot_cnt[:-1], 0.05)

        vert_ballot_box[:-1] = model_data.mesh_graph.vert_lap_trans(
            vert_ballot_box[:-1], 0.05)
    """

    face_segmentor.vert_ballot_box = vert_ballot_box
    face_segmentor.vert_ballot_cnt = vert_ballot_cnt

    face_segment_result: face_seg_utils.FaceSegmentationResult = \
        face_segmentor.segment(mesh_utils.MeshData(
            model_data.mesh_graph,
            model_data.vert_pos,
        ))

    model_data_subdivision_result = model_data.subdivide(
        mesh_subdivision_result=face_segment_result.mesh_subdivision_result)

    sub_model_data = model_data_subdivision_result.model_data

    obj_list = [
        "UPPER_GARMENT",
        "LOWER_GARMENT",
        "HAIR",
    ]

    K = len(obj_list)

    for k in range(K):
        obj = obj_list[k]

        model_data_extraction_result = sub_model_data.extract(
            target_faces=face_segment_result.target_faces[k])

        obj_model_data = model_data_extraction_result.model_data

        obj_model_data.show()

        utils.write_pickle(
            PROJ_DIR / f"obj_model_data_{obj}_{utils.timestamp_sec()}.pkl",
            obj_model_data.state_dict(),
        )


def main3():
    subject_data, trainer = load_trainer()

    module: gom_utils.Module = trainer.training_core.module

    person_smplx_blender: smplx_utils.ModelBlender = module.avatar_blender

    obj = "UPPER_GARMENT"

    obj_model_data: smplx_utils.ModelData = smplx_utils.ModelData.from_file(
        PROJ_DIR / f"obj_model_data_UPPER_GARMENT_1744638970.pkl", dtype=utils.FLOAT, device=DEVICE)

    person_model_data: smplx_utils.ModelData = \
        person_smplx_blender.get_avatar_model()

    model_builder = smplx_utils.StaticModelBuilder(obj_model_data)

    model_blender = smplx_utils.ModelBlender(model_builder)

    T, C, H, W = subject_data.video.shape

    albedo_map = vision_utils.read_image(
        PROJ_DIR / "tex_1744688649.png").to(DEVICE, utils.FLOAT)

    albedo_tex = pytorch3d.renderer.TexturesUV(
        maps=[einops.rearrange(albedo_map, "c h w -> h w c")],
        verts_uvs=[person_model_data.tex_vert_pos.to(utils.FLOAT)],
        faces_uvs=[person_model_data.tex_mesh_graph.f_to_vvv],
        padding_mode="reflection",
        sampling_mode="bilinear",
    )

    subject_data.camera_transform = subject_data.camera_transform.to(
        DEVICE, utils.FLOAT)

    out_images = torch.empty((T, C, H, W), dtype=torch.uint8)

    for t in tqdm.tqdm(range(T)):
        camera_transform = subject_data.camera_transform
        blending_param = subject_data.blending_param[t]

        # cur_model: smplx_utils.Model = model_blender(blending_param)
        cur_model = smplx_utils.Model = trainer.training_core.module.avatar_blender(
            blending_param)

        fragments = rendering_utils.rasterize_mesh(
            cur_model.vert_pos,
            cur_model.mesh_graph.f_to_vvv,
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

    vision_utils.write_video(
        PROJ_DIR / f"seg_video[{obj}]_{utils.timestamp_sec()}.avi",
        out_images,
        subject_data.fps,
    )


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True, True):
        with torch.no_grad():
            main2()
