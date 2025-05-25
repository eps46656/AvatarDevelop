
import dataclasses
import os
import pathlib
import typing

import matplotlib.pyplot as plt
import torch
import torchrbf
import trimesh
from beartype import beartype

from . import (config, mesh_utils, people_snapshot_utils, smplx_utils, utils, video_seg_utils, vision_utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")

SUBJECT_NAME = "female-1-casual"


def print_tensor(x: torch.Tensor):
    print(f"[")

    for val in x.flatten().tolist():
        print(f"{val:+.6e}f", end=", ")

    print(f"]")


def get_subdirs(path: os.PathLike) -> list[os.PathLike]:
    return [name for name in utils.to_pathlib_path(path).glob("*") if name.is_dir()]


def get_train_dir(train_base_dir: os.PathLike, train_name: str) -> os.PathLike:
    dirs = get_subdirs(train_base_dir / train_name)

    assert len(dirs) == 1

    return dirs[0]


@beartype
@dataclasses.dataclass
class GARTResult:
    state_dict: typing.Mapping[str, typing.Any]

    body_shape: torch.Tensor  # [10] smpl has 10 body shapes
    canonical_pose: torch.Tensor  # [24, 3, 3] smpl has 24 joints

    gp_mean: torch.Tensor  # [N, 3]
    gp_rot_q: torch.Tensor  # [N, 4]
    gp_scale: torch.Tensor  # [N, 1]
    gp_opacity: torch.Tensor  # [N, 1]
    gp_color: torch.Tensor  # [N, 3]


@beartype
def read_gart_result(
    gart_log_dir: os.PathLike,
    train_name: str,
    device: torch.device = utils.CPU_DEVICE,
):
    gart_log_dir = utils.to_pathlib_path(gart_log_dir)

    train_dir = gart_log_dir / train_name

    dirs = get_subdirs(train_dir)

    assert len(dirs) == 1

    train_data_dir = dirs[0]

    state_dict = torch.load(
        train_dir / "model.pth",
        map_location=utils.CPU_DEVICE,
    )

    body_shape: torch.Tensor = state_dict["body_shape"].cpu()
    # [10]

    gp_mean: torch.Tensor = state_dict["body_shape"]  # [N, 3]
    gp_rot_q: torch.Tensor = state_dict["body_shape"]  # [N, 4]
    gp_scale: torch.Tensor = state_dict["body_shape"]  # [N, 1]
    gp_opacity: torch.Tensor = state_dict["body_shape"]  # [N, 1]
    gp_color: torch.Tensor = state_dict["body_shape"]  # [N, 3]

    N = -1

    N = utils.check_shapes(
        body_shape, (10,),
        gp_mean, (N, 3),
        gp_rot_q, (N, 4),
        gp_scale, (N, 1),
        gp_opacity, (N, 1),
        gp_color, (N, 3),
    )

    return GARTResult(
        state_dict=state_dict,

        body_shape=body_shape.to(device),

        gp_mean=gp_mean.to(device),
        gp_rot_q=gp_rot_q.to(device),
        gp_scale=gp_scale.to(device),
        gp_opacity=gp_opacity.to(device),
        gp_color=gp_color.to(device),
    )


def main1():
    k_dir = "seq=female-3-casual_prof=people_2m_data=people_snapshot"

    train_name = "train_2025_0426_1"

    train_base_dir = DIR / f"GART/logs"

    print(f"{train_base_dir=}")

    train_dir = get_train_dir(train_base_dir, train_name)

    print(f"{train_dir=}")

    state_dict = torch.load(train_dir / "model.pth")

    for key, val in state_dict.items():
        print(f"{key}: {val.shape}    {val.dtype}")

    vert_pos = state_dict["_xyz"].cpu().reshape(-1, 3)
    V = vert_pos.shape[0]

    scene = trimesh.Scene()

    cloud = trimesh.points.PointCloud(vert_pos)

    scene.add_geometry(cloud)

    scene.show()


def main2():
    gard_result = read_gart_result(
        gart_log_dir=config.GART_DIR / "logs",
        train_name="train_2025_0426_1",
        device=DEVICE,
    )

    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPLX_NEUTRAL_MODEL_PATH,
        model_config=smplx_utils.smplx_model_config,
        dtype=torch.float64,
        device=DEVICE,
    )

    J = model_data.kin_tree.joints_cnt
    V = gard_result.gp_mean.shape[0]

    model_builder = smplx_utils.StaticModelBuilder(model_data=model_data)

    model_blender = smplx_utils.ModelBlender(model_builder=model_builder)

    shaped_model: smplx_utils.Model = model_blender(
        smplx_utils.BlendingParam(body_shape=gard_result.body_shape))

    lbs_weight_field = torchrbf.RBFInterpolator(
        y=shaped_model.vert_pos.cpu(),  # [V, 3]
        d=model_data.lbs_weight.detach().cpu(),  # [V, J]
        smoothing=1.0,
        kernel="thin_plate_spline",
    ).to(DEVICE)

    lbs_weight = lbs_weight_field(gard_result.gp_mean.cpu())  # [N, J]

    gard_model_data = smplx_utils.ModelData(
        kin_tree=model_data.kin_tree,

        body_joints_cnt=model_data.body_joints_cnt,
        jaw_joints_cnt=model_data.jaw_joints_cnt,
        eye_joints_cnt=model_data.eye_joints_cnt,

        mesh_graph=mesh_utils.MeshGraph.empty(),
        tex_mesh_graph=mesh_utils.MeshGraph.empty(),

        joint_t_mean=shaped_model.joint_T[..., :3, 3],

        vert_pos=shaped_model.vert_pos,

        tex_vert_pos=utils.zeros_like(shaped_model.tex_vert_pos, shape=(0, 2)),

        lbs_weight=lbs_weight,

        body_shape_joint_dir=utils.zeros_like(
            shaped_model.tex_vert_pos, shape=(J, 3, 0)),
        expr_shape_joint_dir=utils.zeros_like(
            shaped_model.tex_vert_pos, shape=(J, 3, 0)),

        body_shape_vert_dir=utils.zeros_like(
            shaped_model.tex_vert_pos, shape=(V, 3, 0)),
        expr_shape_vert_dir=utils.zeros_like(
            shaped_model.tex_vert_pos, shape=(V, 3, 0)),

        lhand_pose_mean=model_data.lhand_pose_mean,
        rhand_pose_mean=model_data.rhand_pose_mean,

        pose_vert_dir=utils.zeros_like(
            shaped_model.tex_vert_pos, shape=(J, 3, (J - 1) * 3 * 3)),
    )

    gard_model_builder = smplx_utils.StaticModelBuilder(
        model_data=gard_model_data)

    gard_model_blender = smplx_utils.ModelBlender(
        model_builder=gard_model_builder)

    pass


if __name__ == "__main__":
    main1()
