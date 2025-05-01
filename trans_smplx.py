import dataclasses
import os

import torch
from beartype import beartype

from . import kin_utils, mesh_utils, utils


@dataclasses.dataclass
class ModelConfig:
    body_shapes_space_dim: int

    body_shapes_cnt: int
    expr_shapes_cnt: int

    body_joints_cnt: int
    jaw_joints_cnt: int
    eye_joints_cnt: int
    hand_joints_cnt: int


@beartype
def trans(
    model_data_path: os.PathLike,
    model_config: ModelConfig,
):
    model_data = utils.read_pickle(model_data_path)

    kin_tree_table = model_data["kintree_table"]

    kin_tree_links = [
        (int(kin_tree_table[0, j]), int(kin_tree_table[1, j]))
        for j in range(kin_tree_table.shape[1])]

    kin_tree = kin_utils.KinTree.from_links(kin_tree_links, 2**32-1)
    # joints_cnt = J

    J = kin_tree.joints_cnt

    # ---

    BS = model_config.body_shapes_cnt
    ES = model_config.expr_shapes_cnt

    BODYJ = model_config.body_joints_cnt
    JAWJ = model_config.jaw_joints_cnt
    EYEJ = model_config.eye_joints_cnt
    HANDJ = model_config.hand_joints_cnt

    assert BODYJ + JAWJ + EYEJ + HANDJ * 2 == J

    # ---

    def fetch_float(field_name: str):
        return torch.from_numpy(model_data[field_name]).to(torch.float64)

    vert_pos = fetch_float("v_template")
    pose_vert_dir = fetch_float("posedirs")
    lbs_weight = fetch_float("weight")
    joint_regressor = fetch_float("J_regressor")
    shape_dirs = fetch_float("shapedirs")

    tex_vert_pos = fetch_float("vt")

    faces = fetch_float("f")
    tex_faces = fetch_float("ft")

    lhand_poses_mean = fetch_float("hands_meanl").reshape((-1, 3))[-HANDJ:, :]
    rhand_poses_mean = fetch_float("hands_meanr").reshape((-1, 3))[-HANDJ:, :]

    def get_shape_dirs(shape_dirs: torch.Tensor, shape_dirs_cnt: int) \
            -> torch.Tensor:
        K = utils.check_shapes(shape_dirs, (V, 3, -1))

        assert 0 <= shape_dirs_cnt

        if shape_dirs_cnt == 0:
            return None

        if shape_dirs_cnt <= K:
            ret = shape_dirs[:, :, :shape_dirs_cnt]
        else:
            ret = torch.nn.functional.pad(
                shape_dirs,
                (0, shape_dirs_cnt - K),
                "constant",
                0
            )

        return ret.to(utils.FLOAT)

    body_shape_dirs = get_shape_dirs(
        shape_dirs[:, :, :model_config.body_shapes_space_dim], BS)

    expr_shape_dirs = get_shape_dirs(
        shape_dirs[:, :, model_config.body_shapes_space_dim:], ES)

    # ---

    V, TV, F = -1, -2, -3, 4

    V, TV, F = utils.check_shapes(
        vert_pos, (V, 3),
        pose_vert_dir, (V, 3, (J - 1) * 3 * 3),
        lbs_weight, (V, J),
        joint_regressor, (J, V),
        tex_vert_pos, (TV, 2),
        faces, (F, 3),
        tex_faces, (F, 3),

        lhand_poses_mean, (HANDJ, 3),
        rhand_poses_mean, (HANDJ, 3),

        body_shape_dirs, (V, 3, BS),
        expr_shape_dirs, (V, 3, ES),
    )

    # ---

    joint_t_mean = torch.einsum(
        "...jv, ...vx -> ...jx",
        joint_regressor,
        vert_pos,
    )

    # ---

    mesh_graph = mesh_utils.MeshGraph.from_faces(V, faces)

    tex_mesh_graph = mesh_utils.MeshGraph.from_faces(TV, tex_faces)

    # ---
