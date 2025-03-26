
import enum
import pathlib
import pickle
import typing

import torch

import smplx

from . import blending_utils, kin_utils, mesh_utils, smplx_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = utils.CPU


def main1():
    model_path = DIR / "models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.pkl"

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

    pose_rs = utils.AxisAngleToRotMat(utils.RandUnit((J, 3)),
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
    model_path = DIR / "models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.pkl"

    body_shapes_cnt = 10
    expr_shapes_cnt = 10
    body_joints_cnt = 22
    jaw_joints_cnt = 1
    eye_joints_cnt = 1
    hand_joints_cnt = 15

    with utils.Timer():
        my_smplx_builder = smplx_utils.ModelBlender(
            smplx_utils.ReadModelData(
                model_data_path=model_path,
                body_shapes_cnt=body_shapes_cnt,
                expr_shapes_cnt=expr_shapes_cnt,
                body_joints_cnt=body_joints_cnt,
                jaw_joints_cnt=jaw_joints_cnt,
                eye_joints_cnt=eye_joints_cnt,
                hand_joints_cnt=hand_joints_cnt,
                device=DEVICE,
            ),
            device=DEVICE,
        )

    with utils.Timer():
        smplx_builder = smplx.SMPLX(
            model_path=DIR / "models_smplx_v1_1/models/smplx",
            use_pca=False,
            num_betas=10,
            num_expression_coeffs=10,
            dtype=utils.FLOAT
        )

    with utils.Timer():
        if True:
            body_shapes = torch.rand((body_shapes_cnt,),
                                     dtype=utils.FLOAT, device=DEVICE) * 10
            expr_shapes = torch.rand((expr_shapes_cnt,),
                                     dtype=utils.FLOAT, device=DEVICE) * 10
        else:
            body_shapes = torch.zeros((body_shapes_cnt,),
                                      dtype=utils.FLOAT, device=DEVICE)
            expr_shapes = torch.zeros((expr_shapes_cnt,),
                                      dtype=utils.FLOAT, device=DEVICE)

    with utils.Timer():
        global_transl = torch.rand(
            (1, 3), dtype=utils.FLOAT, device=DEVICE) * 10 - 5

        global_rot = utils.RandRotVec(
            (1, 3), dtype=utils.FLOAT, device=DEVICE)

        print(f"{global_rot=}")

        if True:
            body_poses = utils.RandRotVec((1, body_joints_cnt - 1, 3),
                                          dtype=utils.FLOAT, device=DEVICE)

            jaw_poses = utils.RandRotVec((1, jaw_joints_cnt, 3),
                                         dtype=utils.FLOAT, device=DEVICE)

            leye_poses = utils.RandRotVec((1, eye_joints_cnt, 3),
                                          dtype=utils.FLOAT, device=DEVICE)

            reye_poses = utils.RandRotVec((1, eye_joints_cnt, 3),
                                          dtype=utils.FLOAT, device=DEVICE)

            lhand_poses = utils.RandRotVec((1, hand_joints_cnt, 3),
                                           dtype=utils.FLOAT, device=DEVICE)

            rhand_poses = utils.RandRotVec((1, hand_joints_cnt, 3),
                                           dtype=utils.FLOAT, device=DEVICE)
        else:
            body_poses = torch.zeros((1, body_joints_cnt - 1, 3),
                                     dtype=utils.FLOAT, device=DEVICE)

            jaw_poses = torch.zeros((1, jaw_joints_cnt, 3),
                                    dtype=utils.FLOAT, device=DEVICE)

            leye_poses = torch.zeros((1, LEYE_POSES_CNT, 3),
                                     dtype=utils.FLOAT, device=DEVICE)

            reye_poses = torch.zeros((1, eye_joints_cnt, 3),
                                     dtype=utils.FLOAT, device=DEVICE)

            lhand_poses = torch.zeros((1, hand_joints_cnt, 3),
                                      dtype=utils.FLOAT, device=DEVICE)

            rhand_poses = torch.zeros((1, hand_joints_cnt, 3),
                                      dtype=utils.FLOAT, device=DEVICE)

    print(f"{body_poses.shape=}")

    with utils.Timer():
        smplx_model = smplx_builder.forward(
            betas=body_shapes.unsqueeze(0),
            expression=expr_shapes.unsqueeze(0),

            global_orient=global_rot,
            transl=global_transl,

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
            smplx_utils.BlendingParam(
                body_shapes=body_shapes,
                expr_shapes=expr_shapes,

                global_transl=global_transl,
                global_rot=global_rot,

                body_poses=body_poses,
                jaw_poses=jaw_poses,
                leye_poses=leye_poses,
                reye_poses=reye_poses,
                lhand_poses=lhand_poses,
                rhand_poses=rhand_poses,

                blending_vertex_normal=False,
            )
        )

    print(f"{my_smplx_model.joint_Ts.shape=}")
    print(f"{my_smplx_model.vertex_positions.shape=}")

    print(f"{smplx_model.vertices.shape=}")

    # print(f"{my_smplx_model.joints=}")
    # print(f"{smplx_model.joints=}")

    joint_err = utils.GetL2RMS(
        my_smplx_model.joint_Ts[..., :3, 3].reshape((-1, 3)) -
        smplx_model.joints.flatten()[:165].reshape((-1, 3))
    )

    print(f"{joint_err=}")

    vertices_err = utils.GetL2RMS(
        my_smplx_model.vertex_positions.reshape((-1, 3)) -
        smplx_model.vertices.reshape((-1, 3))
    )

    print(f"{vertices_err=}")

    vertex_normals = my_smplx_model.vertex_normals

    with utils.Timer():
        re_vertex_normals = mesh_utils.GetAreaWeightedVertexNormals(
            faces=my_smplx_builder.GetFaces(),
            vertex_positions=my_smplx_model.vertex_positions
        )

    """
    norms = utils.VectorNorm(vertex_normals)

    print(norms)
    print(utils.VectorNorm(re_vertex_normals))

    normal_err = utils.GetAngle(vertex_normals, re_vertex_normals).mean()

    print(f"{normal_err=}")
    print(f"{normal_err * utils.RAD / utils.DEG=}")
    """


if __name__ == "__main__":
    main3()
