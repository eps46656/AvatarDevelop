import os
import pathlib
import pickle

import numpy as np
import scipy

import chumpy

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]


def read_obj(obj_path: os.PathLike):
    tex_vert_pos = []
    tex_faces = []

    with open(obj_path, "r") as file:
        for line in file:
            parts = line.split()

            if not parts:
                continue

            if parts[0] == 'vt':
                tex_vert_pos.append(list(map(float, parts[1:])))

            if parts[0] == "f":
                tex_faces.append(
                    [int(part.split('/')[1]) - 1 for part in parts[1:]])

    return tex_vert_pos, tex_faces


def F(model_name: str, tex_faces, tex_vert_pos):
    model_data_path = DIR / f"SMPL_python_v.1.1.0/smpl/models/{model_name}.pkl"

    uv_map_path = DIR / f"smpl_uv_20200910/smpl_uv.obj"

    dst_model_data_path = DIR / f"smpl_models/{model_name}.pkl"

    with open(model_data_path, mode="rb") as f:
        model_data = pickle.load(f, encoding="latin")

    dst_model_data = dict()

    for key, value in model_data.items():
        print(f"{key}")
        print(f"{type(value)}")

        if isinstance(value, str):
            print("value: ", value)
            dst_model_data[key] = value
            continue

        if isinstance(value, np.ndarray):
            print("dst_value.shape: ", value.shape)
            dst_model_data[key] = value
            continue

        if isinstance(value, scipy.sparse.csc.csc_matrix):
            dst_value = value.toarray()
            assert isinstance(dst_value, np.ndarray)
            print("dst_value.shape: ", dst_value.shape)
            dst_model_data[key] = dst_value
            continue

        if isinstance(value, chumpy.ch.Ch):
            dst_value = value.r
            assert isinstance(dst_value, np.ndarray)
            print("dst_value.shape: ", dst_value.shape)
            dst_model_data[key] = dst_value
            continue

        assert False

    assert "ft" not in dst_model_data
    assert "vt" not in dst_model_data

    dst_model_data["ft"] = tex_faces
    dst_model_data["vt"] = tex_vert_pos

    with open(dst_model_data_path, mode="wb+") as f:
        pickle.dump(dst_model_data, f)


def G(model_name: str):
    model_data_path = DIR / f"smpl_models/{model_name}.pkl"

    with open(model_data_path, mode="rb") as f:
        model_data = pickle.load(f, encoding="latin")

    J_regressor_prior = model_data["J_regressor_prior"]
    J_regressor = model_data["J_regressor"]

    err = np.sqrt(np.mean(np.square(J_regressor - J_regressor_prior)))

    print(err)


def main1():
    # model_name = "basicmodel_f_lbs_10_207_0_v1.1.0"
    # model_name = "basicmodel_m_lbs_10_207_0_v1.1.0"

    model_names = [
        "basicmodel_neutral_lbs_10_207_0_v1.1.0",
        "basicmodel_f_lbs_10_207_0_v1.1.0",
        "basicmodel_m_lbs_10_207_0_v1.1.0",
    ]

    uv_map_path = DIR / f"smpl_uv_20200910/smpl_uv.obj"

    tex_verts, tex_faces = read_obj(uv_map_path)

    smpl_tex_verts = np.array(tex_verts)
    smpl_tex_faces = np.array(tex_faces)

    print(f"{smpl_tex_verts.shape}")
    print(f"{smpl_tex_faces.shape}")

    print(f"{smpl_tex_faces.min()}")
    print(f"{smpl_tex_faces.max()}")

    return

    for model_name in model_names:
        F(model_name, smpl_tex_verts, smpl_tex_faces)


def main2():
    model_names = [
        "basicmodel_neutral_lbs_10_207_0_v1.1.0",
        "basicmodel_f_lbs_10_207_0_v1.1.0",
        "basicmodel_m_lbs_10_207_0_v1.1.0",
    ]

    for model_name in model_names:
        G(model_name)


def main3():
    model_name = "basicmodel_neutral_lbs_10_207_0_v1.1.0"

    model_data_path = DIR / f"smpl_models/{model_name}.pkl"

    model_data = smplx_utils.ReadModelData(
        model_data_path=model_data_path,
        body_shapes_cnt=smplx_utils.BODY_SHAPES_CNT,
        expr_shapes_cnt=0,
        body_joints_cnt=smplx_utils.BODY_JOINTS_CNT,
        jaw_joints_cnt=0,
        eye_joints_cnt=0,
        hand_joints_cnt=0,
        device=torch.device("cpu")
    )

    print(model_data)


if __name__ == "__main__":
    main1()

    print("ok")
