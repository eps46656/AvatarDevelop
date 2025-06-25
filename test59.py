import dataclasses
import itertools
import math
import random
import typing

import beartype
import einops
import torch
import torchvision
import tqdm
from beartype import beartype

from . import (camera_utils, config, kernel_splatting_utils, mesh_seg_utils,
               mesh_utils, people_snapshot_utils, pipeline_utils,
               rendering_utils, smplx_utils, transform_utils, utils,
               video_seg_utils, vision_utils)

SUBJECT_NAME = "female-3-casual"
SUBJECT_SHORT_NAME = "f3c"

VIDEO_SEG_DIR = config.DIR / f"video_seg_{SUBJECT_SHORT_NAME}_2025_0609_1"

MESH_SEG_DIR = config.DIR / f"mesh_seg_{SUBJECT_SHORT_NAME}_2025_0624_1"

MODEL_DATA_PATH = config.DIR / \
    f"tex_avatar_{SUBJECT_SHORT_NAME}_2025_0624_1/remeshed_model_data.pkl"

# SECOND_MODEL_DATA_PATH = config.DIR / \
#     f"mesh_layer_{SUBJECT_SHORT_NAME}_2025_0609_1/model_data.pkl"

SECOND_MODEL_DATA_PATH = config.DIR / \
    f"mesh_seg_{SUBJECT_SHORT_NAME}_2025_0624_1/union_sub_model_data.pkl"


DTYPE = torch.float64
DEVICE = utils.CUDA_DEVICE


OBJ_LIST = [
    "hair",
    "upper_garment",
    "lower_garment",
    "footwear",
    "skin",
]


O = len(OBJ_LIST)


def main1():
    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        model_data=smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_NEUTRAL_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=torch.float64,
            device=DEVICE,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    model_data = smplx_utils.ModelData.from_state_dict(
        state_dict=utils.read_pickle(MODEL_DATA_PATH),
        dtype=torch.float64,
        device=DEVICE,
    )

    model_data.show()

    vert_pos_diff = model_data.vert_pos - subject_data.model_data.vert_pos

    print(f"{vert_pos_diff.abs().max()=}")
    print(f"{vert_pos_diff.abs().mean()=}")

    """
    model_data = model_data.remesh(
        utils.ArgPack(
            epochs_cnt=10,
            lr=1e-3,
            betas=(0.5, 0.5),

            alpha_lap_diff=1000.0,
            alpha_edge_var=1.0,
        )
    )

    blending_coeff_field = smplx_utils.BlendingCoeffField.from_model_data(
        smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=DTYPE,
            device=DEVICE,
        )
    )

    model_data = blending_coeff_field.query_model_data(model_data)

    model_data.show()

    utils.write_pickle(
        MESH_SEG_DIR / "base_model_data.pkl", model_data.state_dict())
    """

    V = model_data.verts_cnt

    model_blender = smplx_utils.ModelBlender(
        smplx_utils.StaticModelBuilder(
            model_data=model_data,
        )
    )

    def frame_to_conf(x: torch.Tensor) -> torch.Tensor:
        # x[0, 255]

        x = x.to(torch.float64) / 255.0
        # [0, 1]

        x = x * 2 - 1
        # [-1, 1]

        return x

        return x.sign() * x.abs().pow(1.5)

    vert_weight, vert_conf = mesh_seg_utils.face_vert_vote(
        vert_weight=None,
        vert_conf=None,

        avatar_blender=model_blender,

        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,

        blending_param=subject_data.blending_param,

        obj_conf=[
            map(
                frame_to_conf,
                video_seg_utils.read_obj_mask(VIDEO_SEG_DIR, obj_name, None),
            )
            for obj_name in OBJ_LIST
        ],

        batch_size=8,

        device=utils.CUDA_DEVICE,
    )
    # vert_weight: [V + 1, O]
    # vert_conf: [V + 1, O]

    vert_weight = vert_weight.mean(-1)  # [V + 1]

    utils.write_pickle(MESH_SEG_DIR / "raw_vert_weight.pkl", vert_weight)
    utils.write_pickle(MESH_SEG_DIR / "raw_vert_conf.pkl", vert_conf)


def main2():
    model_data = smplx_utils.ModelData.from_state_dict(
        state_dict=utils.read_pickle(MODEL_DATA_PATH),
        dtype=torch.float64,
        device=DEVICE,
    )

    vert_weight = utils.read_pickle(MESH_SEG_DIR / "raw_vert_weight.pkl")
    vert_conf = utils.read_pickle(MESH_SEG_DIR / "raw_vert_conf.pkl")

    # vert_weight[V + 1]
    # vert_conf[V + 1, O]

    vert_conf = vert_conf / (1e-8 + vert_weight[:, None])

    print(f"{vert_conf.shape=}")

    def kernel(x): return (1e-5 + x).pow(-3)

    for _ in range(4):
        nxt_vert_weight = kernel_splatting_utils.query(
            d_pos=model_data.vert_pos,  # [V, 3]
            d_val=vert_weight[:-1, None],  # [V, 1]
            d_weight=vert_weight[:-1],  # [V]

            q_pos=model_data.vert_pos,  # [V, 3]
            kernel=kernel,
        )[1]  # [V, 1]

        nxt_vert_conf = kernel_splatting_utils.query(
            d_pos=model_data.vert_pos,  # [V, 3]
            d_val=vert_conf[:-1, :],  # [V, O]
            d_weight=vert_weight[:-1],  # [V]

            q_pos=model_data.vert_pos,  # [V, 3]
            kernel=kernel,
        )[1]  # [V, O]

        vert_weight[:-1] = nxt_vert_weight[:, 0]
        vert_conf[:-1, :] = nxt_vert_conf

    utils.write_pickle(MESH_SEG_DIR / "vert_weight.pkl", vert_weight)
    utils.write_pickle(MESH_SEG_DIR / "vert_conf.pkl", vert_conf)

    assign_sub_mesh_faces_result: mesh_seg_utils.AssignSubMeshFacesResult = \
        mesh_seg_utils.assign_sub_mesh_faces(
            mesh_graph=model_data.mesh_graph,
            vert_conf=vert_conf,
            threshold=None,
        )

    utils.write_pickle(
        MESH_SEG_DIR / "assign_sub_mesh_faces_result.pkl",
        assign_sub_mesh_faces_result,
    )

    extract_sub_mesh_data_result: mesh_seg_utils.ExtractSubMeshDataResult = \
        mesh_seg_utils.extract_sub_mesh_data(
            mesh_data=mesh_utils.MeshData(
                mesh_graph=model_data.mesh_graph,
                vert_pos=model_data.vert_pos,
            ),
            mesh_subdivide_result=assign_sub_mesh_faces_result.mesh_subdivide_result,
            target_faces=assign_sub_mesh_faces_result.target_faces,
        )

    utils.write_pickle(
        MESH_SEG_DIR / "extract_sub_mesh_data_result.pkl",
        extract_sub_mesh_data_result,
    )

    model_data_subdivide_result = model_data.subdivide(
        mesh_subdivide_result=assign_sub_mesh_faces_result.mesh_subdivide_result,

        new_vert_t=extract_sub_mesh_data_result.union_sub_vert_t,
    )

    union_sub_model_data = model_data_subdivide_result.model_data

    utils.write_pickle(
        MESH_SEG_DIR / "union_sub_model_data.pkl",
        union_sub_model_data.state_dict(),
    )

    mesh_data = mesh_utils.MeshData(
        union_sub_model_data.mesh_graph,
        union_sub_model_data.vert_pos,
    )

    obj_color = torch.tensor([
        [*color, 255]
        for _, color in zip(range(O), utils.generate_color())
    ], dtype=torch.uint8)
    # [O, 4]

    tm = mesh_data.trimesh

    tm.visual.vertex_colors = obj_color[
        assign_sub_mesh_faces_result.sub_vert_obj_idx.cpu()
    ].numpy()
    # [V, 4]

    tm.show()


def main3():
    model_data = smplx_utils.ModelData.from_state_dict(
        state_dict=utils.read_pickle(SECOND_MODEL_DATA_PATH),
        dtype=torch.float64,
        device=DEVICE,
    )

    # model_data.show()

    mesh_graph = model_data.mesh_graph

    mesh_data = mesh_utils.MeshData(
        mesh_graph,
        model_data.vert_pos
    )

    vert_weight = utils.read_pickle(MESH_SEG_DIR / "vert_weight.pkl")
    vert_conf = utils.read_pickle(MESH_SEG_DIR / "vert_conf.pkl")

    # vert_weight[V + 1]
    # vert_conf[V + 1, O]

    extract_sub_mesh_data_result: mesh_seg_utils.ExtractSubMeshDataResult = \
        utils.read_pickle(MESH_SEG_DIR / "extract_sub_mesh_data_result.pkl")

    blending_coeff_field = smplx_utils.BlendingCoeffField.from_model_data(
        model_data=pipeline_utils.load_smplx_model_data(
            pipeline_utils.smpl_female_model_info,
            dtype=DTYPE,
            device=DEVICE,
        )
    )

    with torch.no_grad():
        sub_model_data_list: list[smplx_utils.ModelData] = list()

        for o in range(O):
            sub_model_data_extract_result = model_data.extract(
                mesh_extract_result=extract_sub_mesh_data_result.sub_mesh_extract_result[o],
            )

            sub_model_data = blending_coeff_field.query_model_data(
                sub_model_data_extract_result.model_data)

            sub_model_data.show()

            sub_model_data_list.append(sub_model_data)

            utils.write_pickle(
                MESH_SEG_DIR /
                f"obj_model_data_{OBJ_LIST[o]}_{utils.timestamp_sec()}.pkl",

                sub_model_data.state_dict(),
            )

        mesh_utils.show_mesh_data(
            [
                mesh_utils.MeshData(
                    m.mesh_graph,
                    m.vert_pos,
                )
                for m in sub_model_data_list
            ]
        )

        """
        skin_model_data = extract_smplx_sub_model_data_result. \
            sub_model_data_extract_result[O].model_data

        skin_model_data.show()

        utils.write_pickle(
            MESH_SEG_DIR /
            f"skin_model_data_{utils.timestamp_sec()}.pkl",

            skin_model_data.state_dict(),
        )
        """


if __name__ == "__main__":
    main3()
