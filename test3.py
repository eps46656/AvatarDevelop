import numpy as np
import utils
import pickle

import torch
import sampling_utils

import camera_utils
import rendering_utils

import smplx.smplx

import pathlib

import json

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

INT = torch.int32
FLOAT = torch.float32

ORIGIN = torch.tensor([[0], [0], [0]], dtype=float)
X_AXIS = torch.tensor([[1], [0], [0]], dtype=float)
Y_AXIS = torch.tensor([[0], [1], [0]], dtype=float)
Z_AXIS = torch.tensor([[0], [0], [1]], dtype=float)

SMPLX_FT = +Z_AXIS
SMPLX_BK = -Z_AXIS

SMPLX_LT = +X_AXIS
SMPLX_RT = -X_AXIS

SMPLX_UP = +Y_AXIS
SMPLX_DW = -Y_AXIS

DTYPE = torch.float64


def main1():
    H, W = 960, 960

    raduis = 10
    theta = 60 * utils.DEG
    phi = (180 + 270) / 2 * utils.DEG

    proj_mat = camera_utils.MakeProjMat(
        img_shape=(H, W),
        origin=torch.tensor(utils.Sph2Cart(raduis, theta, phi)),
        # origin=torch.tensor([4, 5, 6, 7]),
        aim=ORIGIN,
        quasi_u_dir=Z_AXIS,
        diag_fov=45 * utils.DEG,
    )[:3, :]

    # model_mat = np.eye(4, dtype=float)
    # model_mat[]

    model_mat: torch.Tensor = \
        utils.AxisAngleToRotMat(torch.tensor([0, 0, 1], dtype=FLOAT), 135*utils.DEG) @ \
        torch.linalg.lstsq(
            # model coordinate
            torch.tensor([
                ORIGIN.flatten().tolist() + [1],  # position trnasform
                SMPLX_FT.flatten().tolist() + [0],  # direction transform
                SMPLX_RT.flatten().tolist() + [0],  # direction transform
                SMPLX_UP.flatten().tolist() + [0],  # direction transform
            ], dtype=float),

            # world coordinate
            torch.tensor([
                [0, 0, 0, 1],  # position trnasform
                X_AXIS.flatten().tolist() + [0],  # direction transform
                Z_AXIS.flatten().tolist() + [0],  # direction transform
                Y_AXIS.flatten().tolist() + [0],  # direction transform
            ], dtype=float),
        )[0].transpose(0, 1)

    print(f"{model_mat=}")

    with utils.Timer() as t:
        smplx_builder = smplx.smplx.SMPLX(
            model_path=DIR / "smplx/models/smplx",
            num_betas=10,
            use_pca=False,
            dtype=float
        )

    with open(DIR / "smplx_param.json") as f:
        pose_params = json.load(f)

    left_hand_pose = torch.tensor(
        pose_params["lhand_pose"]).reshape((1, -1))

    print(f"{left_hand_pose.shape=}")
    print(f"{len(pose_params["shape"])=}")
    print(f"{len(pose_params["body_pose"])=}")
    print(f"{len(pose_params["lhand_pose"])=}")
    print(f"{len(pose_params["rhand_pose"])=}")
    print(f"{len(pose_params["jaw_pose"])=}")
    print(f"{len(pose_params["expr"])=}")

    with utils.Timer() as t:
        smplx_model = smplx_builder.forward(
            betas=torch.tensor(
                pose_params["shape"], dtype=torch.float64).reshape((1, -1)),
            body_pose=torch.tensor(
                pose_params["body_pose"], dtype=torch.float64).reshape((1, -1, 3)),
            left_hand_pose=torch.tensor(
                pose_params["lhand_pose"], dtype=torch.float64).reshape((1, -1)),
            right_hand_pose=torch.tensor(
                pose_params["rhand_pose"], dtype=torch.float64).reshape((1, -1)),
            jaw_pose=torch.tensor(
                pose_params["jaw_pose"], dtype=torch.float64).reshape((1, 3)),
            expression=torch.tensor(
                pose_params["expr"], dtype=torch.float64).reshape((1, -1)),
            return_full_pose=True,
            return_verts=True,
            return_shaped=True,
        )

    model_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    with open(model_path, "rb") as model_file:
        model_data = pickle.load(model_file, encoding="latin1")

    faces = smplx_builder

    vertices_: torch.Tensor = smplx_model.vertices.reshape((-1, 3))
    # [V, 3]

    V = vertices_.shape[0]

    vertices: torch.Tensor = torch.empty((4, V), dtype=float)
    vertices[:3, :] = vertices_.transpose(0, 1)
    vertices[3, :] = 1

    texture_vertices = torch.tensor(model_data["vt"], dtype=float)
    # [V, 2]

    proj_vertices = (proj_mat @ model_mat) @ vertices
    # [3, V]

    img_points = proj_vertices[:2, :] / proj_vertices[2, :]
    # [2, V]

    faces = model_data["f"]
    texture_faces = model_data["ft"]

    print(f"{vertices.shape=}")
    print(f"{texture_vertices.shape=}")
    print(f"{faces.shape=}")
    print(f"{texture_faces.shape=}")

    texture_vertices_min = torch.min(texture_vertices)
    texture_vertices_max = torch.max(texture_vertices)

    print(f"{texture_vertices_min=}")
    print(f"{texture_vertices_max=}")

    F = faces.shape[0]

    texture = torch.tensor(
        utils.ReadImage(
            DIR / "black_male.png", "chw"),
        dtype=float)

    texture_map = torch.zeros((H, W, 2), dtype=float)
    w_map = torch.zeros((H, W), dtype=float)

    with utils.Timer() as timer:
        for fi in range(F):
            vai = int(faces[fi, 0])
            vbi = int(faces[fi, 1])
            vci = int(faces[fi, 2])

            vtai = int(texture_faces[fi, 0])
            vtbi = int(texture_faces[fi, 1])
            vtci = int(texture_faces[fi, 2])

            vta = texture_vertices[vtai, :]
            vtb = texture_vertices[vtbi, :]
            vtc = texture_vertices[vtci, :]

            img_point_a = img_points[:, vai].tolist()
            img_point_b = img_points[:, vbi].tolist()
            img_point_c = img_points[:, vci].tolist()

            # 3, 2

            wa = 1 / proj_vertices[2, vai]
            wb = 1 / proj_vertices[2, vbi]
            wc = 1 / proj_vertices[2, vci]

            for (x, y), (la, lb, lc) in rendering_utils.RasterizeTriangle(
                torch.tensor([img_point_a, img_point_b, img_point_c]
                             ).transpose(0, 1),
                    (0, H-1), (0, W-1)):
                cur_w = wa * la + wb * lb + wc * lc

                if cur_w <= w_map[x, y]:
                    continue

                u, v = vta * la + vtb * lb + vtc * lc

                texture_map[x, y, 0] = 1-v
                texture_map[x, y, 1] = u
                w_map[x, y] = cur_w

    albedo_texture_sampler = sampling_utils.TextureSampler(
        2,
        texture,
        sampling_utils.WrapModeEnum.MIRROR_REPEAT,
        sampling_utils.InterpModeEnum.CUBIC,
    )

    with utils.Timer() as timer:
        albedo_map = albedo_texture_sampler.Sample(texture_map)
        # [c, h, w]

    print(f"{albedo_map.shape=}")

    img = np.zeros((H, W, 3), dtype=np.uint8)

    for h in range(H):
        for w in range(W):
            if 0 < w_map[h, w]:
                img[h, w, :] = albedo_map[:, h, w].to(dtype=int).numpy()

    utils.WriteImage(DIR / "test.png", img, "hwc")


if __name__ == "__main__":
    main1()
