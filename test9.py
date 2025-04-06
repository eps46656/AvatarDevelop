
import enum
import pathlib
import pickle
import typing

import torch

import smplx

from . import blending_utils, config, kin_utils, mesh_utils, smplx_utils, utils

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = utils.CPU_DEVICE


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

    kin_tree = kin_utils.KinTree.from_links(kin_tree_links, 2**32-1)
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
        .reshape(1, 3, 3).expand(J, 3, 3)

    binding_pose_ts = torch.zeros(
        (J, 3), dtype=utils.FLOAT, device=DEVICE)

    pose_rs = utils.axis_angle_to_rot_mat(utils.rand_unit((J, 3)),
                                          torch.rand((J,)))
    # [J, 3, 3]

    pose_ts = torch.rand((J, 3))

    print(f"{pose_rs.shape=}")

    v = blending_utils.lbs(
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
    model_data_path = config.SMPLX_NEUTRAL_MODEL

    model_config = smplx_utils.smplx_model_config

    with utils.Timer():
        my_smplx_model_data: smplx_utils.Model = smplx_utils.ModelData.from_origin_file(
            model_data_path=model_data_path,
            model_config=model_config,
            dtype=utils.FLOAT,
            device=DEVICE,
        )

    with utils.Timer():
        my_smplx_model_builder = smplx_utils.StaticModelBuilder(
            model_data=my_smplx_model_data,
        )

    with utils.Timer():
        my_smplx_model_blender = smplx_utils.ModelBlender(
            model_builder=my_smplx_model_builder,
        )

    with utils.Timer():
        smplx_builder = smplx.SMPLX(
            model_path=DIR / "models_smplx_v1_1/models/smplx",
            gender="NEUTRAL",
            use_pca=False,
            num_betas=10,
            num_expression_coeffs=10,
            dtype=utils.FLOAT,
        )

    with utils.Timer():
        if True:
            body_shapes = torch.rand((model_config.body_shapes_cnt,),
                                     dtype=utils.FLOAT, device=DEVICE) * 10
            expr_shapes = torch.rand((model_config.expr_shapes_cnt,),
                                     dtype=utils.FLOAT, device=DEVICE) * 10
        else:
            body_shapes = torch.zeros((model_config.body_shapes_cnt,),
                                      dtype=utils.FLOAT, device=DEVICE)
            expr_shapes = torch.zeros((model_config.expr_shapes_cnt,),
                                      dtype=utils.FLOAT, device=DEVICE)

    with utils.Timer():
        global_transl = torch.rand(
            (1, 3), dtype=utils.FLOAT, device=DEVICE) * 10 - 5

        global_rot = utils.rand_rot_vec(
            (1, 3), dtype=utils.FLOAT, device=DEVICE)

        print(f"{global_rot=}")

        if True:
            body_poses = utils.rand_rot_vec(
                (1, model_config.body_joints_cnt - 1, 3),
                dtype=utils.FLOAT, device=DEVICE)

            jaw_poses = utils.rand_rot_vec(
                (1, model_config.jaw_joints_cnt, 3),
                dtype=utils.FLOAT, device=DEVICE)

            leye_poses = utils.rand_rot_vec(
                (1, model_config.eye_joints_cnt, 3),
                dtype=utils.FLOAT, device=DEVICE)

            reye_poses = utils.rand_rot_vec(
                (1, model_config.eye_joints_cnt, 3),
                dtype=utils.FLOAT, device=DEVICE)

            lhand_poses = utils.rand_rot_vec(
                (1, model_config.hand_joints_cnt, 3),
                dtype=utils.FLOAT, device=DEVICE)

            rhand_poses = utils.rand_rot_vec(
                (1, model_config.hand_joints_cnt, 3),
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
        my_smplx_model: smplx_utils.Model = my_smplx_model_blender(
            smplx_utils.BlendingParam(
                body_shape=body_shapes,
                expr_shape=expr_shapes,

                global_transl=global_transl,
                global_rot=global_rot,

                body_pose=body_poses,
                jaw_pose=jaw_poses,
                leye_pose=leye_poses,
                reye_pose=reye_poses,
                lhand_pose=lhand_poses,
                rhand_pose=rhand_poses,

                device=DEVICE,
                dtype=utils.FLOAT,
            )
        )

    print(f"{my_smplx_model.joint_T.shape=}")
    print(f"{my_smplx_model.vert_pos.shape=}")

    print(f"{smplx_model.vertices.shape=}")

    # print(f"{my_smplx_model.joints=}")
    # print(f"{smplx_model.joints=}")

    joint_err = utils.get_l2_rms(
        my_smplx_model.joint_T[..., :3, 3].reshape((-1, 3)) -
        smplx_model.joints.flatten()[:165].reshape((-1, 3))
    )

    print(f"{joint_err=}")

    vertices_err = utils.get_l2_rms(
        my_smplx_model.vert_pos.reshape((-1, 3)) -
        smplx_model.vertices.reshape((-1, 3))
    )

    print(f"{my_smplx_model.vert_pos=}")
    print(f"{smplx_model.vertices=}")

    print(f"{vertices_err=}")

    vertex_normals = my_smplx_model.vert_nor

    """

    with utils.Timer():
        re_vertex_normals = mesh_utils.get_area_weighted_vertex_normals(
            faces=my_smplx_model_blender.faces,
            vertex_positions=my_smplx_model.vertex_positions
        )

    norms = utils.VectorNorm(vertex_normals)

    print(norms)
    print(utils.VectorNorm(re_vertex_normals))

    normal_err = utils.GetAngle(vertex_normals, re_vertex_normals).mean()

    print(f"{normal_err=}")
    print(f"{normal_err * utils.RAD / utils.DEG=}")
    """


if __name__ == "__main__":
    main3()
