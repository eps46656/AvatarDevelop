import dataclasses
import itertools
import math
import typing

import beartype
import einops
import torch
import torchvision
import tqdm
from beartype import beartype

from . import (camera_utils, config, mesh_seg_utils, mesh_utils,
               people_snapshot_utils, rendering_utils, smplx_utils,
               transform_utils, utils, video_seg_utils, vision_utils)

SUBJECT_NAME = "female-1-casual"

VIDEO_SEG_DIR = config.DIR / "video_seg_2025_0517_1"

MESH_SEG_DIR = config.DIR / "mesh_seg_2025_0518_1"

DTYPE = torch.float64
DEVICE = utils.CUDA_DEVICE


OBJ_LIST = [
    "HAIR",
    "UPPER_GARMENT",
    "LOWER_GARMENT",
]


O = len(OBJ_LIST)


def read_subject():
    return people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        model_data=smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_NEUTRAL_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=torch.float64,
            device=DEVICE,
        ),
        device=DEVICE,
    )


def main1():
    subject_data = read_subject()

    model_data = subject_data.model_data

    V = model_data.verts_cnt

    model_blender = smplx_utils.ModelBlender(
        smplx_utils.StaticModelBuilder(
            model_data=model_data,
        )
    )

    def frame_to_mask(x):
        return x.to(torch.float64) / 255.0

    def mask_to_conf(x):
        x = x * 2 - 1
        # [-1, 1]

        return x.sign() * x.abs().pow(1.5)

    vert_weight, vert_conf = mesh_seg_utils.face_vert_vote(
        vert_weight=None,
        vert_conf=None,

        avatar_blender=model_blender,

        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,

        blending_param=subject_data.blending_param,

        obj_mask=[
            *(
                map(frame_to_mask, video_seg_utils.read_refined_obj_mask(
                    VIDEO_SEG_DIR, obj_name, None))
                for obj_name in OBJ_LIST
            ),

            map(frame_to_mask, video_seg_utils.read_skin_mask(
                VIDEO_SEG_DIR, None))
        ],

        mask_to_conf=mask_to_conf,

        batch_size=8,

        device=utils.CUDA_DEVICE,
    )

    utils.write_pickle(MESH_SEG_DIR / "vert_weight.pkl", vert_weight)
    utils.write_pickle(MESH_SEG_DIR / "vert_conf.pkl", vert_conf)


def main2():
    subject_data = read_subject()

    model_data = subject_data.model_data

    # model_data.show()

    mesh_graph = model_data.mesh_graph

    mesh_data = mesh_utils.MeshData(
        mesh_graph,
        model_data.vert_pos
    )

    vert_weight = utils.read_pickle(MESH_SEG_DIR / "vert_weight.pkl")
    vert_conf = utils.read_pickle(MESH_SEG_DIR / "vert_conf.pkl")

    std = 50 * 1e-3
    # 50 mm = 5 cm

    def kernel(x):
        return torch.exp(-0.5 * (x / std)**2)

    vert_conf = vert_conf / (1e-2 + vert_weight)

    vert_conf[:-1] = mesh_seg_utils.refine_vert_conf(
        vert_conf=vert_conf[:-1],

        vert_pos=mesh_data.vert_pos,

        kernel=kernel,

        iters_cnt=3,
    )

    assign_sub_mesh_faces_result: mesh_seg_utils.AssignSubMeshFacesResult = \
        mesh_seg_utils.assign_sub_mesh_faces(
            mesh_graph=mesh_graph,
            vert_conf=vert_conf,
        )

    extract_sub_mesh_data_result: mesh_seg_utils.ExtractSubMeshDataResult = \
        mesh_seg_utils.extract_sub_mesh_data(
            mesh_data=mesh_data,
            mesh_subdivide_result=assign_sub_mesh_faces_result.mesh_subdivide_result,
            target_faces=assign_sub_mesh_faces_result.target_faces,
        )

    with torch.no_grad():
        blending_coeff_field = smplx_utils.BlendingCoeffField.from_model_data(
            model_data=smplx_utils.ModelData.from_origin_file(
                model_data_path=config.SMPL_FEMALE_MODEL_PATH,
                model_config=smplx_utils.smpl_model_config,
                dtype=utils.FLOAT,
                device=DEVICE,
            ),
        )

        extract_smplx_sub_model_data_result = \
            mesh_seg_utils.extract_smplx_sub_model_data(
                model_data=model_data,

                blending_coeff_field=blending_coeff_field,

                extract_sub_mesh_data_result=extract_sub_mesh_data_result,
            )

        print(
            f"{len(extract_smplx_sub_model_data_result.sub_model_data_extract_result)=}")

        for o in range(O):
            sub_model_data = extract_smplx_sub_model_data_result. \
                sub_model_data_extract_result[o].model_data

            sub_model_data.show()

            utils.write_pickle(
                MESH_SEG_DIR /
                f"obj_model_data_{OBJ_LIST[o]}_{utils.timestamp_sec()}.pkl",

                sub_model_data.state_dict(),
            )

        skin_model_data = extract_smplx_sub_model_data_result. \
            sub_model_data_extract_result[O].model_data

        skin_model_data.show()

        utils.write_pickle(
            MESH_SEG_DIR /
            f"skin_model_data_{utils.timestamp_sec()}.pkl",

            skin_model_data.state_dict(),
        )


if __name__ == "__main__":
    main1()
