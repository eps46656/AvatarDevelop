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

from . import (avatar_utils, camera_utils, config, dataset_utils, gom_avatar_utils,
               kernel_splatting_utils, mesh_seg_utils, mesh_utils,
               people_snapshot_utils, rendering_utils, smplx_utils, training_utils, transform_utils, utils, video_seg_utils,
               vision_utils)


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


DEVICE = utils.CUDA_DEVICE

PROJ_DIR = config.DIR / "train_2025_0530_1"

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

MESH_SEG_PROJ = "mesh_seg_2025_0501_0154_1"


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

    dataset = gom_avatar_utils.Dataset(gom_avatar_utils.Sample(
        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,
        img=subject_data.video,
        mask=subject_data.mask,
        blending_param=subject_data.blending_param,
    )).to(DEVICE)

    # ---

    smplx_model_builder = smplx_utils.DeformableModelBuilder(
        temp_model_data=subject_data.model_data,
        model_data=subject_data.model_data,
    ).to(DEVICE)

    smplx_model_builder.unfreeze()

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    module = gom_avatar_utils.Module(
        avatar_blender=smplx_model_blender,
        color_channels_cnt=3,
    ).to(DEVICE).train()

    # ---

    trainer_core = gom_avatar_utils.TrainerCore(
        config=gom_avatar_utils.TrainerCoreConfig(
            proj_dir=PROJ_DIR,
            device=DEVICE,
            batch_size=BATCH_SIZE,

            lr=LR,
            betas=(0.5, 0.5),
            gamma=0.95,

            vert_grad_norm_threshold=VERT_GRAD_NORM_THRESHOLD,

            alpha_img_diff=ALPHA_RGB,
            alpha_lap_diff=ALPHA_LAP_SMOOTHNESS,
            alpha_nor_sim=ALPHA_NOR_SIM,
            alpha_edge_var=ALPHA_EDGE_VAR,
            alpha_gp_color_diff=ALPHA_COLOR_DIFF,
            alpha_gp_scale_diff=ALPHA_GP_SCALE_DIFF,
        ),
        module=module,
        dataset=dataset,
    )

    # ---

    trainer = training_utils.Trainer(
        proj_dir=PROJ_DIR,
        trainer_core=trainer_core,
    )

    return subject_data, trainer


def main1():
    subject_data, trainer = load_trainer()

    trainer.load_latest()

    trainer_core: gom_avatar_utils.TrainerCore = \
        trainer.trainer_core

    dataset = trainer_core.dataset

    temp_model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_FEMALE_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=utils.FLOAT,
        device=DEVICE,
    )

    model_data: smplx_utils.ModelData = \
        trainer_core.module.avatar_blender.get_avatar_model()

    model_builder = smplx_utils.DeformableModelBuilder(
        temp_model_data=temp_model_data,
        model_data=model_data,
    ).to(DEVICE)

    mesh_data: mesh_utils.MeshData = mesh_utils.MeshData(
        model_data.mesh_graph,
        model_data.vert_pos,
    ).remesh(
        epochs_cnt=100,
        lr=1e-3,
        betas=(0.5, 0.5),
    )

    model_builder.model_data.vert_pos = mesh_data.vert_pos
    model_builder.refresh()

    model_data = model_builder.model_data

    model_data.show()

    # ---

    obj_list = [
        video_seg_utils.ObjectType.UPPER_GARMENT,
        video_seg_utils.ObjectType.LOWER_GARMENT,
        video_seg_utils.ObjectType.HAIR,
    ]

    mask_dir = config.DIR / "segment_2025_0509_1"

    masks = [vision_utils.read_video(
        mask_dir / video_seg_utils.get_refined_obj_mask_filename(obj),
        dtype=torch.float32,
        device=DEVICE,
    )[0] for obj in obj_list]

    # [K][T, H, W]

    K = len(masks)

    gom_module: gom_avatar_utils.Module = trainer.trainer_core.module

    smplx_blender: smplx_utils.ModelBlender = gom_module.avatar_blender

    smplx_model_builder: smplx_utils.DeformableModelBuilder = \
        smplx_blender.model_builder

    print(f"{smplx_model_builder.model_data.vert_pos.shape=}")

    smplx_model_builder.refresh()

    print(f"{smplx_model_builder.model_data.vert_pos.shape=}")

    origin_model_data = smplx_model_builder.model_data

    # origin_model_data.show()

    print(f"{smplx_model_builder.model_data.vert_pos.shape=}")

    utils.write_pickle(
        config.DIR / MESH_SEG_PROJ / "origin_model_data.pkl",
        origin_model_data.state_dict(),
    )

    print(f"{smplx_model_builder.model_data.vert_pos.shape=}")

    """
    smplx_model_builder.remesh(utils.ArgPack(
        target_length=0.01,
        iterations=5,
    ))
    """

    edge_len_threshold = 20 * 1e-3

    while True:
        utils.mem_clear()

        cur_model_data = smplx_model_builder.model_data

        print(f"{cur_model_data.vert_pos.shape=}")

        mesh_data = mesh_utils.MeshData(
            cur_model_data.mesh_graph,
            cur_model_data.vert_pos,
        )

        print(f"{mesh_data.vert_pos.shape=}")

        target_edges = {
            ei
            for ei, edge_len in enumerate(mesh_data.edge_norm)
            if edge_len_threshold < edge_len
        }

        print(f"{len(target_edges)=}")

        if len(target_edges) == 0:
            break

        gom_module.subdivide(target_edges=target_edges)

        utils.torch_cuda_sync()

        utils.mem_clear()

        smplx_model_builder.refresh()

    model_data = smplx_model_builder.model_data

    # model_data.show()

    utils.write_pickle(
        config.DIR / MESH_SEG_PROJ / "model_data.pkl",
        model_data.state_dict(),
    )

    avatar_model: avatar_utils.AvatarModel = gom_module.avatar_blender.get_avatar_model()

    mesh_segmentor = mesh_seg_utils.MeshSegmentor.from_empty(
        avatar_model.mesh_data.mesh_graph,
        len(obj_list),
        torch.float64,
        torch.float64,
        DEVICE,
    )

    T, C, H, W = dataset.sample.img.shape

    with torch.no_grad():
        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(
                dataset, batch_size=8, shuffle=False)):
            assert len(batch_idxes) == 1

            idxes = batch_idxes[0]

            batch_size = idxes.shape[0]

            sample: gom_avatar_utils.Sample

            result: gom_avatar_utils.ModuleForwardResult = trainer.trainer_core.module(
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
                    mesh_ras_result.bary_coord.reshape(H, W, 3)
                # [H, W, 3]

                for k in range(K):
                    cur_mask = masks[k][idxes[i]].to(utils.FLOAT)

                    ballot = cur_mask * 2 - 1

                    mesh_segmentor.vote(
                        k,
                        pix_to_face,  # [H, W]
                        bary_coord,  # [H, W, 3]
                        ballot,  # [H, W]
                    )

    utils.write_pickle(
        PROJ_DIR / f"face_voting_result_{utils.timestamp_sec()}.pkl",
        mesh_segmentor.state_dict(),
    )


def main2():
    model_data = smplx_utils.ModelData.from_state_dict(
        utils.read_pickle(config.DIR / MESH_SEG_PROJ / "model_data.pkl"),
        dtype=utils.FLOAT,
        device=DEVICE,
    )

    model_data.show()

    obj_list = [
        "UPPER_GARMENT",
        "LOWER_GARMENT",
        "HAIR",
    ]

    K = len(obj_list)

    mesh_segmentor: mesh_seg_utils.MeshSegmentor = \
        mesh_seg_utils.MeshSegmentor.from_state_dict(utils.read_pickle(
            PROJ_DIR / "face_voting_result_1746071719.pkl"),
            None,
            DEVICE,
        )

    vert_ballot_box = mesh_segmentor.vert_ballot_box / (
        1e-2 + mesh_segmentor.vert_ballot_cnt)
    # [V + 1, K]

    std = 50 * 1e-3
    # 50 mm = 5 cm

    def kernel(x):
        return torch.exp(-0.5 * (x / std)**2)

    vert_weight = 1 / (1e-2 + kernel_splatting_utils.calc_density(
        mesh_data.vert_pos,  # [V, 3]
        kernel,
    ))
    # [V]

    print(f"{vert_weight.shape=}")

    for i in range(3):
        vert_ballot_box[:-1] = kernel_splatting_utils.interp(
            mesh_data.vert_pos,  # [V, 3]
            vert_ballot_box[:-1],  # [V, K]
            vert_weight,  # [V]
            mesh_data.vert_pos,  # [V, 3]
            kernel,
        )

    """
    for i in range(3):
        vert_ballot_cnt[:-1] = model_data.mesh_graph.vert_lap_trans(
            vert_ballot_cnt[:-1], 0.05)

        vert_ballot_box[:-1] = model_data.mesh_graph.vert_lap_trans(
            vert_ballot_box[:-1], 0.05)
    """

    mesh_segmentor.vert_ballot_cnt.fill_(1)
    mesh_segmentor.vert_ballot_box = vert_ballot_box

    mesh_segment_result: mesh_seg_utils.MeshSegmentationResult = \
        mesh_segmentor.segment(mesh_data.mesh_graph)

    sub_vert_obj_kdx = mesh_segment_result.sub_vert_obj_kdx

    vert_src_table = mesh_segment_result.mesh_subdivision_result.vert_src_table
    # [V_, 2]

    total_sub_vert_pos = model_data.vert_pos[vert_src_table]
    # [V_, 2, 3]

    total_sub_vert_pos_a = total_sub_vert_pos[:, 0]
    total_sub_vert_pos_b = total_sub_vert_pos[:, 1]
    # [V_, 3]

    V_ = vert_src_table.shape[0]

    total_sub_mesh_graph = mesh_segment_result.mesh_subdivision_result.mesh_graph

    raw_vert_t = utils.zeros_like(
        mesh_data.vert_pos, shape=(V_,)
    ).requires_grad_()

    optimizer = torch.optim.Adam(
        [raw_vert_t],
        lr=1e-3,
        betas=(0.5, 0.5),
    )

    sub_mesh_graphs = [
        total_sub_mesh_graph.extract(
            target_faces=mesh_segment_result.target_faces[k],
            remove_orphan_vert=False,
        ).mesh_graph
        for k in range(K)
    ]

    for epoch_i in range(800):
        print(f"{epoch_i=}")
        optimizer.zero_grad()

        total_sub_vert_t = torch.where(
            sub_vert_obj_kdx == 0,
            raw_vert_t.sigmoid() * 0.8 + 0.1,
            0.5,
        )
        # [V_] [0.1, 0.9]

        cur_total_sub_vert_pos = \
            total_sub_vert_pos_a * (1 - total_sub_vert_t)[..., None] + \
            total_sub_vert_pos_b * total_sub_vert_t[..., None]

        print(f"{cur_total_sub_vert_pos.requires_grad=}")

        loss = 0.0

        for k in range(K):
            sub_mesh_data = mesh_utils.MeshData(
                sub_mesh_graphs[k],
                cur_total_sub_vert_pos,
            )

            loss = loss + sub_mesh_data.l2_uni_lap_smoothness

        print(f"{loss=}")

        loss.backward(retain_graph=True)

        optimizer.step()

    total_sub_vert_t = torch.where(
        sub_vert_obj_kdx == 0,
        raw_vert_t.sigmoid() * 0.8 + 0.1,
        0.5,
    )
    # [V_] [0.1, 0.9]

    print(f"{total_sub_vert_t.min()=}")
    print(f"{total_sub_vert_t.max()=}")

    model_data_subdivision_result = model_data.subdivide(
        mesh_subdivision_result=mesh_segment_result.mesh_subdivision_result,
        new_vert_t=raw_vert_t,
    )

    total_sub_model_data = model_data_subdivision_result.model_data

    for k in range(K):
        obj = obj_list[k]

        model_data_extraction_result = total_sub_model_data.extract(
            target_faces=mesh_segment_result.target_faces[k])

        obj_model_data = model_data_extraction_result.model_data

        obj_model_data.show()

        utils.write_pickle(
            PROJ_DIR / f"obj_model_data_{obj}_{utils.timestamp_sec()}.pkl",
            obj_model_data.state_dict(),
        )


def main3():
    model_data = smplx_utils.ModelData.from_state_dict(
        utils.read_pickle(config.DIR / MESH_SEG_PROJ / "model_data.pkl"),
        dtype=utils.FLOAT,
        device=DEVICE,
    )

    model_blender = smplx_utils.ModelBlender(
        smplx_utils.StaticModelBuilder(model_data))

    subject_data = read_subject()

    tex = rendering_utils.bake_texture(
        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,

        img=vision_utils.normalize_image(
            subject_data.video),
        mask=subject_data.mask,
        blending_param=subject_data.blending_param,

        avatar_blender=model_blender,

        tex_h=1024,
        tex_w=1024,

        batch_size=2,

        device=DEVICE,
    )
    # [H, W, C]

    vision_utils.write_image(
        path=PROJ_DIR / f"tex_{utils.timestamp_sec()}.png",
        img=vision_utils.denormalize_image(
            einops.rearrange(tex, "h w c -> c h w")),
    )


def main4():
    model_data = smplx_utils.ModelData.from_state_dict(
        utils.read_pickle(
            PROJ_DIR / f"obj_model_data_UPPER_GARMENT_1746071935.pkl"),
        dtype=utils.FLOAT,
        device=DEVICE,
    )

    model_blender = smplx_utils.ModelBlender(
        smplx_utils.StaticModelBuilder(model_data))

    subject_data = read_subject()

    obj = "UPPER_GARMENT"

    T, C, H, W = subject_data.video.shape

    texture = einops.rearrange(vision_utils.read_image(
        PROJ_DIR / "tex_1746082618.png").to(DEVICE, utils.FLOAT),
        "c h w -> h w c",
    )
    # [H, W, C]

    print(f"{texture.min()=}")
    print(f"{texture.max()=}")

    """
    albedo_tex = pytorch3d.renderer.TexturesUV(
        maps=[einops.rearrange(albedo_map, "c h w -> h w c")],
        verts_uvs=[model_data.tex_vert_pos.to(utils.FLOAT)],
        faces_uvs=[model_data.tex_mesh_graph.f_to_vvv],
        padding_mode="reflection",
        sampling_mode="bilinear",
    )
    """

    subject_data.camera_transform = subject_data.camera_transform.to(
        DEVICE, utils.FLOAT)

    out_images = torch.empty((T, C, H, W), dtype=torch.uint8)

    bg_color = torch.tensor([255], dtype=torch.uint8).to(DEVICE)

    for t in tqdm.tqdm(range(T)):
        camera_transform = subject_data.camera_transform
        blending_param = subject_data.blending_param[t]

        cur_model: smplx_utils.Model = model_blender(blending_param)

        mesh_ras_result = rendering_utils.rasterize_mesh(
            cur_model.vert_pos,
            cur_model.mesh_graph.f_to_vvv,
            subject_data.camera_config,
            camera_transform,
            1,
        )

        # mesh_ras_result.pix_to_face[H, W, 1]
        # mesh_ras_result.bary_coord[H, W, 1, 3]

        # texels = albedo_tex.sample_textures(mesh_ras_result).view(H, W, C)
        # [H, W, C]

        vvv = model_data.tex_mesh_graph.f_to_vvv[
            mesh_ras_result.pix_to_face.view(H, W)]
        # [H, W, 3]

        vvv_pos = model_data.tex_vert_pos[vvv]
        # [H, W, 3, 2]

        tex_coord = (
            mesh_ras_result.bary_coord.view(H, W, 1, 3) * vvv_pos).sum(dim=-2)
        # [H, W, 3]

        texels = rendering_utils.sample_texture(
            texture=texture,
            tex_coord=tex_coord,
            wrap_mode=rendering_utils.WrapMode.MIRROR,
            sampling_mode=rendering_utils.SamplingMode.LINEAR,
        )

        texels = torch.where(
            (0 <= mesh_ras_result.pix_to_face.view(H, W))
            [..., None].expand(H, W, 1),

            texels,

            bg_color.expand(H, W, C),
        )

        out_images[t] = einops.rearrange(texels, "h w c -> c h w")

    vision_utils.write_video(
        PROJ_DIR / f"seg_video[{obj}]_{utils.timestamp_sec()}.avi",
        out_images,
        subject_data.fps,
    )


def main5():
    pass


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True, True):
        main1()
