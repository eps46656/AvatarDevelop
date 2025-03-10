
import enum

import typing
import pathlib
import torch
import pickle

import utils
import blending_utils

import smplx.smplx

from kin_tree import KinTree
import smplx_utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cpu")


def main1():
    model_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    with open(model_path, "rb") as f:
        model_data = pickle.load(f, encoding="latin1")

    kintree_table = model_data["kintree_table"]

    print(kintree_table.shape)

    J = kintree_table.shape[1]

    '''
    for j in range(J):
        a = int(kintree_table[0, j])
        b = int(kintree_table[1, j])

        print(f"{a}, {b}")

        if a < 0 or J <= a:
            a_name = "none"
        else:
            a_name = SMPL_Skeleton.JointIdxToJointName(a)

        if b < 0 or J <= b:
            b_name = "none"
        else:
            b_name = SMPL_Skeleton.JointIdxToJointName(b)

        print(f"{a_name}, {b_name}")
    '''

    kin_tree_links = [(int(kintree_table[0, j]), int(kintree_table[1, j]))
                      for j in range(J)]

    kin_tree = KinTree.FromLinks(kin_tree_links, 2**32-1)
    # joints_cnt = J

    vertices = torch.from_numpy(model_data["v_template"]).to(
        dtype=utils.FLOAT, device=DEVICE)
    # [V, 3]

    lbs_weights = torch.from_numpy(model_data["weights"]).to(
        dtype=utils.FLOAT, device=DEVICE)
    # [V, J]

    joint_regressor = torch.from_numpy(model_data["J_regressor"]).to(
        dtype=utils.FLOAT, device=DEVICE)
    # [J, V]

    shape_dirs = torch.from_numpy(model_data["shapedirs"]).to(
        dtype=utils.FLOAT, device=DEVICE)
    # [V, 3, B]

    ###

    J = kin_tree.joints_cnt
    V = vertices.shape[0]

    ###

    binding_pose_rs = torch.eye(3, dtype=utils.FLOAT, device=DEVICE) \
        .reshape(1, 3, 3).expand((J, 3, 3))

    binding_pose_ts = torch.zeros(
        (J, 3), dtype=utils.FLOAT, device=DEVICE)

    pose_rs = utils.GetRotMat(utils.RandUnit((J, 3)),
                              torch.rand((J,)))
    # [J, 3, 3]

    pose_ts = torch.rand((J, 3))

    print(f"{pose_rs.shape=}")

    v = blending_utils.LBS(
        kin_tree,
        vertices,
        lbs_weights,
        binding_pose_rs,
        binding_pose_ts,
        pose_rs,
        pose_ts
    )

    print(f"{v.shape}")


def main3():
    model_path = DIR / "smplx/models/smplx/SMPLX_NEUTRAL.pkl"

    with utils.Timer():
        my_smplx_builder = smplx_utils.SMPLXBuilder(model_path, DEVICE)

    with utils.Timer():
        smplx_builder = smplx.smplx.SMPLX(
            model_path=DIR / "smplx/models/smplx",
            use_pca=False,
            num_betas=10,
            num_expression_coeffs=10,
            dtype=utils.FLOAT
        )

    J = my_smplx_builder.GetJointsCnt()
    B = my_smplx_builder.GetShapesCnt()

    print(f"{J=}")
    print(f"{B=}")

    with utils.Timer():
        if True:
            body_shapes = torch.rand((1, smplx_utils.BODY_SHAPES_CNT),
                                     dtype=utils.FLOAT, device=DEVICE) * 10
            expr_shapes = torch.rand((1, smplx_utils.EXPR_SHAPES_CNT),
                                     dtype=utils.FLOAT, device=DEVICE) * 10
        else:
            body_shapes = torch.zeros((1, smplx_utils.BODY_SHAPES_CNT),
                                      dtype=utils.FLOAT, device=DEVICE)
            expr_shapes = torch.zeros((1, smplx_utils.EXPR_SHAPES_CNT),
                                      dtype=utils.FLOAT, device=DEVICE)

    with utils.Timer():
        root_ts = torch.rand(
            (1, 3), dtype=utils.FLOAT, device=DEVICE) * 10 - 5

        root_rs = utils.RandRotVec(
            (1, 3), dtype=utils.FLOAT, device=DEVICE)

        print(f"{root_rs=}")

        if True:
            body_poses = utils.RandRotVec((1, smplx_utils.BODY_POSES_CNT, 3),
                                          dtype=utils.FLOAT, device=DEVICE)

            jaw_poses = utils.RandRotVec((1, smplx_utils.JAW_POSES_CNT, 3),
                                         dtype=utils.FLOAT, device=DEVICE)

            leye_poses = utils.RandRotVec((1, smplx_utils.LEYE_POSES_CNT, 3),
                                          dtype=utils.FLOAT, device=DEVICE)

            reye_poses = utils.RandRotVec((1, smplx_utils.REYE_POSES_CNT, 3),
                                          dtype=utils.FLOAT, device=DEVICE)

            lhand_poses = utils.RandRotVec((1, smplx_utils.LHAND_POSES_CNT, 3),
                                           dtype=utils.FLOAT, device=DEVICE)

            rhand_poses = utils.RandRotVec((1, smplx_utils.RHAND_POSES_CNT, 3),
                                           dtype=utils.FLOAT, device=DEVICE)
        else:
            body_poses = torch.zeros((1, smplx_utils.BODY_POSES_CNT, 3),
                                     dtype=utils.FLOAT, device=DEVICE)

            jaw_poses = torch.zeros((1, smplx_utils.JAW_POSES_CNT, 3),
                                    dtype=utils.FLOAT, device=DEVICE)

            leye_poses = torch.zeros((1, smplx_utils.LEYE_POSES_CNT, 3),
                                     dtype=utils.FLOAT, device=DEVICE)

            reye_poses = torch.zeros((1, smplx_utils.REYE_POSES_CNT, 3),
                                     dtype=utils.FLOAT, device=DEVICE)

            lhand_poses = torch.zeros((1, smplx_utils.LHAND_POSES_CNT, 3),
                                      dtype=utils.FLOAT, device=DEVICE)

            rhand_poses = torch.zeros((1, smplx_utils.RHAND_POSES_CNT, 3),
                                      dtype=utils.FLOAT, device=DEVICE)

    with utils.Timer():
        smplx_model = smplx_builder.forward(
            betas=body_shapes,
            expression=expr_shapes,

            global_orient=root_rs,
            transl=root_ts,

            body_pose=body_poses,
            jaw_pose=jaw_poses,
            leye_pose=leye_poses,
            reye_pose=reye_poses,
            left_hand_pose=lhand_poses,
            right_hand_pose=rhand_poses,

            return_full_pose=True,
            return_verts=True,
            return_shaped=True,
        )

    with utils.Timer():
        my_smplx_model = my_smplx_builder.forward(
            body_shapes=body_shapes,
            expr_shapes=expr_shapes,

            root_ts=root_ts,
            root_rs=root_rs,

            body_poses=body_poses,
            jaw_poses=jaw_poses,
            leye_poses=leye_poses,
            reye_poses=reye_poses,
            lhand_poses=lhand_poses,
            rhand_poses=rhand_poses,
        )

    print(f"{my_smplx_model.joints.shape=}")
    print(f"{my_smplx_model.vertices.shape=}")

    print(f"{smplx_model.vertices.shape=}")

    # print(f"{my_smplx_model.joints=}")
    # print(f"{smplx_model.joints=}")

    v_temp_err = utils.GetL2RMS(
        my_smplx_builder.GetVertices().reshape((-1, 3)) -
        smplx_builder.v_template.reshape((-1, 3))
    )

    print(f"{v_temp_err=}")

    joint_err = utils.GetL2RMS(
        my_smplx_model.joints.reshape((-1, 3)) -
        smplx_model.joints.flatten()[:165].reshape((-1, 3))
    )

    print(f"{joint_err=}")

    vertices_err = utils.GetL2RMS(
        my_smplx_model.vertices.reshape((-1, 3)) -
        smplx_model.vertices.reshape((-1, 3))
    )

    print(f"{vertices_err=}")


if __name__ == "__main__":
    main3()
