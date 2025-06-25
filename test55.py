import dataclasses
import typing

import einops
import torch
import torchvision
import tqdm
from beartype import beartype

from . import (avatar_utils, camera_utils, config, kernel_splatting_utils,
               mesh_seg_utils, mesh_utils, people_snapshot_utils,
               rendering_utils, smplx_utils, transform_utils,
               utils, video_seg_utils, vision_utils)

DTYPE = utils.FLOAT
DEVICE = utils.CUDA_DEVICE

SUBJECT_NAME = "female-1-casual"


# GOM_AVATAR_DIR = config.DIR / "gom_avatar_2025_0514_1"
VIDEO_SEG_DIR = config.DIR / "video_seg_2025_0514_1"

MESH_SEG_DIR = config.DIR / "mesh_seg_2025_0514_1"


@beartype
@dataclasses.dataclass
class AvatarPack:
    avatar_blender: smplx_utils.ModelBlender

    camera_config: camera_utils.CameraConfig
    camera_transform: transform_utils.ObjectTransform

    blending_param: smplx_utils.BlendingParam


@beartype
def load_avatar_pack():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_FEMALE_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=DTYPE,
        device=DEVICE,
    )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        model_data=model_data,
        device=utils.CPU_DEVICE,
    )

    model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_utils.StaticModelBuilder(
            model_data=model_data,
        )
    )

    return AvatarPack(
        avatar_blender=model_blender,

        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,

        blending_param=subject_data.blending_param,
    )


def main1():
    avatar_pack = load_avatar_pack()

    avatar_blender = avatar_pack.avatar_blender
    camera_config = avatar_pack.camera_config
    camera_transform = avatar_pack.camera_transform
    blending_param = avatar_pack.blending_param

    obj_type_list = [
        "HAIR",
        "UPPER_GARMENT",
        "LOWER_GARMENT",
    ]

    obj_mask_videos = {
        obj_type: vision_utils.VideoReader(
            VIDEO_SEG_DIR /
            video_seg_utils.get_refined_obj_mask_filename(obj_type),
            "GRAY",
        )
        for obj_type in obj_type_list
    }

    H, W = camera_config.img_h, camera_config.img_w

    mesh_segmentor: mesh_seg_utils.MeshSegmentor = mesh_seg_utils.mesh_mask_vote(
        avatar_blender=avatar_blender,

        camera_config=camera_config,
        camera_transform=camera_transform,

        blending_param=blending_param,

        obj_mask=[
            (frame.reshape(H, W) / 255 for frame in obj_mask_video)
            for obj_mask_video in obj_mask_videos.values()
        ],

        batch_size=8,

        device=DEVICE,
    )

    utils.write_pickle(
        MESH_SEG_DIR / "mesh_segmentor.pkl",
        mesh_segmentor.state_dict(),
    )


def main2():
    avatar_pack = load_avatar_pack()

    avatar_blender = avatar_pack.avatar_blender
    camera_config = avatar_pack.camera_config
    camera_transform = avatar_pack.camera_transform
    blending_param = avatar_pack.blending_param

    smplx_model_blender: smplx_utils.ModelBlender = avatar_blender

    assert isinstance(smplx_model_blender, smplx_utils.ModelBlender)

    smplx_model_builder: smplx_utils.ModelBuilder = \
        smplx_model_blender.model_builder

    smplx_model_builder.refresh()

    model_data: smplx_utils.ModelData = \
        smplx_model_builder.get_model_data()

    avatar_model: avatar_utils.AvatarModel = \
        smplx_model_blender.get_avatar_model()

    mesh_data = avatar_model.mesh_data

    obj_type_list = [
        "HAIR",
        "UPPER_GARMENT",
        "LOWER_GARMENT",
    ]

    O = len(obj_type_list)

    mesh_segmentor:  mesh_seg_utils.MeshSegmentor = \
        mesh_seg_utils.MeshSegmentor.from_state_dict(
            utils.read_pickle(MESH_SEG_DIR / "mesh_segmentor.pkl"),
            dtype=torch.float64,
            device=DEVICE,
        )

    vert_ballot_box = mesh_segmentor.vert_ballot_box / (
        1e-2 + mesh_segmentor.vert_ballot_cnt)
    # [V + 1, K]

    if False:
        std = 10e-3

        def kernel(x):
            return torch.exp(-0.5 * (x / std).square())

        vert_weight = 1 / (1e-2 + kernel_splatting_utils.calc_density(
            mesh_data.vert_pos,  # [V, 3]
            kernel,
        ))

        for i in range(3):
            vert_ballot_box[:-1] = kernel_splatting_utils.interp(
                mesh_data.vert_pos,  # [V, 3]
                vert_ballot_box[:-1],  # [V, K]
                vert_weight,  # [V]
                mesh_data.vert_pos,  # [V, 3]
                kernel,
            )

    mesh_segmentor.vert_ballot_cnt.fill_(1)
    mesh_segmentor.vert_ballot_box = vert_ballot_box

    mesh_segment_result: mesh_seg_utils.MeshSegmentationResult = \
        mesh_segmentor.segment(1 / 3)

    sub_vert_obj_kdx = mesh_segment_result.sub_vert_obj_kdx

    vert_src_table = mesh_segment_result.mesh_subdivision_result.vert_src_table
    # [V_, 2]

    V_ = vert_src_table.shape[0]

    total_sub_vert_pos = mesh_data.vert_pos[vert_src_table]
    # [V_, 2, 3]

    total_sub_vert_pos_a = total_sub_vert_pos[:, 0]
    total_sub_vert_pos_b = total_sub_vert_pos[:, 1]
    # [V_, 3]

    total_sub_mesh_graph = mesh_segment_result.mesh_subdivision_result.mesh_graph

    T_LB = 0.1
    T_RB = 0.9

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
            target_faces=mesh_segment_result.target_faces[o],
            remove_orphan_vert=False,
        ).mesh_graph
        for o in range(O)
    ]

    for epoch_i in tqdm.tqdm(range(800)):
        optimizer.zero_grad()

        total_sub_vert_t = torch.where(
            sub_vert_obj_kdx == 0,
            utils.smooth_clamp(raw_vert_t, T_LB, T_RB),
            0.5,
        )
        # [V_] [0.1, 0.9]

        cur_total_sub_vert_pos = \
            total_sub_vert_pos_a * (1 - total_sub_vert_t)[..., None] + \
            total_sub_vert_pos_b * total_sub_vert_t[..., None]

        loss = 0.0

        for o in range(O):
            sub_mesh_data = mesh_utils.MeshData(
                sub_mesh_graphs[o],
                cur_total_sub_vert_pos,
            )

            loss = loss + sub_mesh_data.l2_uni_lap_smoothness

        print(f"{loss=}")

        loss.backward(retain_graph=True)

        optimizer.step()

    total_sub_vert_t = torch.where(
        vert_src_table[:, 0] == vert_src_table[:, 1],
        0.5,
        utils.smooth_clamp(raw_vert_t, T_LB, T_RB),
    )
    # [V_] [0.1, 0.9]

    total_sub_vert_pos = \
        total_sub_vert_pos_a * (1 - total_sub_vert_t)[..., None] + \
        total_sub_vert_pos_b * total_sub_vert_t[..., None]

    total_sub_model_data = model_data.subdivide(
        mesh_subdivide_result=mesh_segment_result.mesh_subdivision_result,
        new_vert_t=total_sub_vert_t,
    ).model_data

    with utils.Timer():
        temp_model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=DTYPE,
            device=DEVICE,
        )

        blending_coeff_field = smplx_utils.BlendingCoeffField.from_model_data(
            temp_model_data)

    skin_faces = set(range(total_sub_model_data.mesh_graph.faces_cnt))

    for o in range(O):
        obj_type = obj_type_list[o]

        skin_faces.difference_update(mesh_segment_result.target_faces[o])

        sub_model_data = blending_coeff_field.query_model_data(
            total_sub_model_data.extract(
                target_faces=mesh_segment_result.target_faces[o],
                remove_orphan_vert=True,
            ).model_data
        )

        sub_model_data.show()

        utils.write_pickle(
            MESH_SEG_DIR /
            f"obj_model_data_{obj_type}_{utils.timestamp_sec()}.pkl",
            sub_model_data.state_dict(),
        )

    skin_sub_model_data = blending_coeff_field.query_model_data(
        total_sub_model_data.extract(
            target_faces=skin_faces,
            remove_orphan_vert=True,
        ).model_data
    )

    skin_sub_model_data.show()

    utils.write_pickle(
        MESH_SEG_DIR /
        f"obj_model_data_SKIN_{utils.timestamp_sec()}.pkl",
        skin_sub_model_data.state_dict(),
    )


if __name__ == "__main__":
    main2()
