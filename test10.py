import dataclasses
import json
import math
import pathlib
import pickle

import torch

from . import (blending_utils, camera_utils, config, kin_utils, mesh_utils,
               smplx_utils, utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


INT = torch.int32
FLOAT = torch.float32
DEVICE = utils.CUDA_DEVICE


def main1():
    model_data_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    model_data: smplx_utils.Model = smplx_utils.ReadModelData(
        model_data_path=model_data_path,
        model_config=smplx_utils.smplx_model_config,
        device=DEVICE,
    )

    face_vertex_adj_mat = mesh_utils.GetFaceVertexAdjMat(
        model_data.vert_pos.shape[-2],
        model_data.faces
    )  # [F, V]

    F, V = utils.check_shapes(face_vertex_adj_mat, (-1, -2))

    D = 5

    lap_diff_mat = mesh_utils.GetLapDiffMat(face_vertex_adj_mat)

    vv_adj_mat = face_vertex_adj_mat.T @ face_vertex_adj_mat

    vertices = torch.rand((V, D))

    with utils.Timer():
        result_a = lap_diff_mat @ vertices

    with utils.Timer():
        result_b = torch.zeros((V, D))

        for i in range(V):
            degree = 0
            g = torch.zeros((D,))

            for j in range(V):
                if i != j and vv_adj_mat[i, j]:
                    degree += degree
                    g += vertices[j]

            if 0 < degree:
                result_b[i] = g / degree - vertices[i]

    err = utils.get_diff(result_a, result_b).mean()

    print(f"{err=}")


def main2():
    F, V, D = 100, 50, 12

    face_vertex_adj_mat = torch.randint(
        low=0, high=2, size=(F, V), dtype=utils.INT)

    vv_adj_mat = face_vertex_adj_mat.T @ face_vertex_adj_mat

    lap_diff_mat = mesh_utils.GetLapDiffMat(face_vertex_adj_mat)

    vertices = torch.rand((V, D))

    with utils.Timer():
        result_a = lap_diff_mat @ vertices

    with utils.Timer():
        result_b = torch.zeros((V, D))

        for i in range(V):
            degree = 0
            g = torch.zeros((D,))

            for j in range(V):
                if i != j and vv_adj_mat[i, j]:
                    degree += 1
                    g += vertices[j]

            if 0 < degree:
                result_b[i] = g / degree - vertices[i]

    err = utils.get_diff(result_a, result_b).mean()

    print(f"{err=}")


def main3():
    F, V, D = 100, 100, 12

    face_vertex_adj_list = torch.empty((F, 3), dtype=torch.long)

    for f in range(F):
        face_vertex_adj_list[f, :] = torch.randperm(F)[:3]

    result_a = mesh_utils.GetFaceVertexAdjMatNaive(V, face_vertex_adj_list)

    result_b = mesh_utils.GetFaceVertexAdjMat(
        V, face_vertex_adj_list)
    # [F, V]

    # vv_adj_mat = torch.sparse.mm(result_b.T, result_b, reduce="amax")

    vv_adj_mat = (result_b.T.reshape((1, V, F)) &
                  result_b.reshape((V, 1, F))).max(dim=-1)[0]

    print(vv_adj_mat)

    print(vv_adj_mat.shape.numel())

    for f in range(F):
        for v in range(V):
            assert result_a[f, v] == result_b[f,
                                              v], f"{result_a[f, v]}, {result_b[f, v]}"


def main4():
    model_data_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    body_shapes_cnt = 10
    expr_shapes_cnt = 10
    body_joints_cnt = 22
    jaw_joints_cnt = 1
    eye_joints_cnt = 1
    hand_joints_cnt = 15

    model_data: smplx_utils.Model = smplx_utils.ReadModelData(
        model_data_path=model_data_path,
        body_shapes_cnt=body_shapes_cnt,
        expr_shapes_cnt=expr_shapes_cnt,
        body_joints_cnt=body_joints_cnt,
        jaw_joints_cnt=jaw_joints_cnt,
        eye_joints_cnt=eye_joints_cnt,
        hand_joints_cnt=hand_joints_cnt,
        device=DEVICE,
    )

    with utils.Timer():
        ff = mesh_utils.GetFaceFaceAdjMat(model_data.faces)

    print(f"{type(ff)}")
    print(f"{ff.shape}")


def main5():
    model_data_path = config.SMPL_FEMALE_MODEL_PATH
    model_config = smplx_utils.smpl_model_config

    model_data: smplx_utils.Model = smplx_utils.ModelData.from_origin_file(
        model_data_path=model_data_path,
        model_config=model_config,
        device=DEVICE,
    )

    V = model_data.vert_pos.shape[-2]
    F = model_data.faces_cnt

    D = 3

    mesh_graph: mesh_utils.MeshGraph = mesh_utils.MeshGraph.from_faces(
        V,
        model_data.mesh_graph.f_to_vvv,
    )

    vertices_cnt = mesh_graph.verts_cnt
    faces_cnt = mesh_graph.faces_cnt

    vert_pos = torch.rand((vertices_cnt, 3), dtype=torch.float32) * 10.0
    face_normals = torch.rand((faces_cnt, 3), dtype=torch.float32)

    vps_1 = vert_pos.clone()
    vps_2 = vert_pos.clone()

    vps_1.requires_grad = True
    vps_2.requires_grad = True

    print(f"{vps_1.requires_grad=}")

    with utils.Timer():
        loss_1 = mesh_utils.MeshData(mesh_graph, vps_1).l1_uni_lap_smoothness

    with utils.Timer():
        loss_2 = mesh_utils.MeshData(
            mesh_graph, vps_2).calc_uni_lap_smoothness_pytorch3d()

    print(f"{loss_1=}")
    print(f"{loss_2=}")

    loss_1.backward()
    loss_2.backward()

    vert_pos_diff = (vps_1 - vps_2).square().max()
    print(f"{vert_pos_diff=}")

    loss_diff_err = (loss_1 - loss_2).square().max()
    print(f"{loss_diff_err=}")

    assert vps_1 is not vps_2

    grad_diff_err = (vps_1.grad - vps_2.grad).square().max()

    print(f"{grad_diff_err=}")

    return

    with utils.Timer():
        normal_sim = mesh_graph.calc_face_cos_sim(face_normals)

    with utils.Timer():
        normal_sim_naive = mesh_graph.calc_face_cos_sim_naive(face_normals)

    normal_sim_1 = normal_sim.square().mean()
    normal_sim_2 = normal_sim_naive.square().mean()

    print(f"{normal_sim_1=}")
    print(f"{normal_sim_2=}")

    normal_sim_err = (normal_sim - normal_sim_naive).square().mean()

    print(f"{normal_sim_err=}")


if __name__ == "__main__":
    main5()
