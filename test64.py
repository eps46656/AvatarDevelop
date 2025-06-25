import copy
import itertools
import math
import pathlib
import time
import typing

import einops
import numpy as np
import torch
import tqdm
import trimesh
from beartype import beartype

from . import (avatar_utils, camera_utils, config, gom_avatar_utils,
               mesh_layer_utils, mesh_seg_utils, mesh_utils,
               people_snapshot_utils, pipeline_utils, rendering_utils,
               sdf_utils, smplx_utils, training_utils, transform_utils, utils,
               vision_utils)

DTYPE = torch.float64
DEVICE = torch.device("cuda")


SUBJECT_NAME = "female-3-casual"
SUBJECT_SHORT_NAME = "f3c"

SDF_MODULE_DIR = config.DIR / "sdf_module_2025_0609_1"

MESH_SEG_DIR = config.DIR / "mesh_seg_f3c_2025_0609_1"

MODEL_DATA_PATH = config.DIR / \
    "tex_avatar_{SUBJECT_SHORT_NAME}_2025_0624_1/remeshed_model_data.pkl"

MESH_LAYER_DIR = config.DIR / "mesh_layer_f3c_2025_0609_1"


def read_subject(model_data: typing.Optional[smplx_utils.ModelData] = None):
    if model_data is None:
        model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=config.SMPL_FEMALE_MODEL_PATH,
            model_config=smplx_utils.smpl_model_config,
            dtype=DTYPE,
            device=DEVICE,
        )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME, model_data, DTYPE, DEVICE)

    return subject_data


def main1():
    temp_model_data = pipeline_utils.load_smplx_model_data(
        pipeline_utils.smpl_female_model_info,
        dtype=DTYPE,
        device=DEVICE,
    )

    blending_coeff_field = smplx_utils.BlendingCoeffField.from_model_data(
        model_data=temp_model_data)

    sdf_module = pipeline_utils.load_sdf_module(
        SDF_MODULE_DIR,
        ckpt_id=pipeline_utils.LoadEnum.LOAD_LATEST,  # load latest
        dtype=DTYPE,
        device=DEVICE,
    )[1].eval()

    assign_sub_mesh_faces_result: mesh_seg_utils.AssignSubMeshFacesResult = \
        utils.read_pickle(MESH_SEG_DIR / "assign_sub_mesh_faces_result.pkl")

    extract_sub_mesh_data_result: mesh_seg_utils.ExtractSubMeshDataResult = \
        utils.read_pickle(MESH_SEG_DIR / "extract_sub_mesh_data_result.pkl")

    union_sub_model_data: smplx_utils.ModelData = \
        smplx_utils.ModelData.from_state_dict(
            state_dict=utils.read_pickle(
                MESH_SEG_DIR / "union_sub_model_data.pkl"
            ),
            dtype=DTYPE,
            device=DEVICE,
        )

    union_sub_model_data.show()

    boundary_vert_flag = \
        assign_sub_mesh_faces_result.sub_vert_obj_idx[
            assign_sub_mesh_faces_result.mesh_subdivide_result.vert_src_table[:, 0]
        ] != \
        assign_sub_mesh_faces_result.sub_vert_obj_idx[
            assign_sub_mesh_faces_result.mesh_subdivide_result.vert_src_table[:, 1]
        ]
    # [V]

    print(f"{boundary_vert_flag.sum()=}")

    signed_dist_lb = torch.where(
        boundary_vert_flag,
        2e-3,
        5e-3,
    )

    signed_dist_rb = torch.where(
        boundary_vert_flag,
        8e-3,
        100e-3,
    )

    mesh_layer_trainer = pipeline_utils.load_mesh_layer_trainer(
        MESH_LAYER_DIR,

        sdf_module=sdf_module,

        mesh_graph=union_sub_model_data.mesh_graph,
        vert_pos=union_sub_model_data.vert_pos,

        signed_dist_lb=signed_dist_lb,
        signed_dist_rb=signed_dist_rb,

        dtype=DTYPE,
        device=DEVICE,
    )

    mesh_layer_trainer_core: mesh_layer_utils.TrainerCore = \
        mesh_layer_trainer.trainer_core

    def _show_model(*args, **kwargs):
        print(f"{args=}")
        print(f"{kwargs=}")

        temp_tm = trimesh.Trimesh(
            vertices=temp_model_data.vert_pos.detach().to(utils.CPU_DEVICE),
            faces=temp_model_data.mesh_graph.f_to_vvv_cpu,
            validate=True,
        )

        temp_tm.visual.vertex_colors = np.array(
            [128, 128, 128, 255], dtype=np.uint8)

        object_tm = trimesh.Trimesh(
            vertices=mesh_layer_trainer_core.vert_pos.detach().to(utils.CPU_DEVICE),

            faces=mesh_layer_trainer_core.init_mesh_data.mesh_graph.f_to_vvv_cpu,

            validate=True,
        )

        object_tm.visual.vertex_colors = np.array(
            [255, 0, 0, 255], dtype=np.uint8)

        scene = trimesh.Scene()

        scene.add_geometry(temp_tm)
        scene.add_geometry(object_tm)

        scene.show()

    def _output_model_data(*args, **kwargs):
        o_model_data = copy.copy(union_sub_model_data)

        o_model_data.vert_pos = mesh_layer_trainer_core.vert_pos.detach()

        o_model_data = blending_coeff_field.query_model_data(o_model_data)

        o_model_data.show()

        utils.write_pickle(
            MESH_LAYER_DIR / f"model_data_{utils.timestamp_sec()}.pkl",

            o_model_data.state_dict(),
        )

    mesh_layer_trainer_core.show_model = _show_model

    mesh_layer_trainer_core.output_model_data = _output_model_data

    mesh_layer_trainer.enter_cli()


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(False, False):
        main1()

    print("ok")
