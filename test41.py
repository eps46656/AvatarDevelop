
import dataclasses
import math
import os
import pathlib
import typing

import matplotlib.pyplot as plt
import pyvista as pv
import torch
import torchrbf
import tqdm
from beartype import beartype

from . import (config, gart_utils, gaussian_utils, mesh_utils,
               people_snapshot_utils, rbf_utils, segment_utils, smplx_utils,
               utils, vision_utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")

SUBJECT_NAME = "female-3-casual"


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


def main1():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPLX_NEUTRAL_MODEL_PATH,
        model_config=smplx_utils.smplx_model_config,
        dtype=torch.float64,
        device=DEVICE,
    )

    J = model_data.kin_tree.joints_cnt

    model_builder = smplx_utils.DeformableModelBuilder(model_data=model_data)

    x, y, z = torch.meshgrid(
        torch.linspace(-2, 2, 50, dtype=utils.FLOAT, device=DEVICE),
        torch.linspace(-2, 2, 50, dtype=utils.FLOAT, device=DEVICE),
        torch.linspace(-2, 2, 50, dtype=utils.FLOAT, device=DEVICE),
        indexing="ij",
    )
    # [N]

    print(f"{x.shape=}")

    v = model_builder.lbs_weight_interp(
        torch.stack([x, y, z], dim=-1).reshape(-1, 3))[:, :3]
    # [N, 3]

    print(f"{v.shape=}")

    grid = pv.StructuredGrid(
        x.numpy(force=True),
        y.numpy(force=True),
        z.numpy(force=True),
    )

    grid["vectors"] = v.numpy(force=True)
    grid["magnitude"] = utils.vec_norm(v).numpy(force=True)

    grid.active_vectors_name = "vectors"

    # 建立 plotter
    plotter = pv.Plotter()
    # 加箭頭 (可以選擇加或不加)

    # --- 重點 --- #
    slice_actor = [None]  # 用 list 包起來，方便在 callback 裡修改
    current_z = [0.0]

    def update_slice(z_value):

        if slice_actor[0] is not None:
            plotter.remove_actor(slice_actor[0])  # 移除舊的 slice

        slice_plane = grid.slice(normal="z", origin=(0, 0, z_value))

        slice_actor[0] = plotter.add_mesh(
            slice_plane, scalars="magnitude", cmap="plasma", show_edges=False)

    # 初始化
    update_slice(current_z[0])

    # 加 slider
    slider_widget = plotter.add_slider_widget(
        callback=lambda z: (current_z.__setitem__(0, z), update_slice(z)),
        rng=[-1, 1],
        value=current_z[0],
        title="Slice Z",
        style="modern",
    )

    # --- 加快捷鍵 --- #
    step = 0.05  # 每次按一下移動多少

    def move_left():
        current_z[0] -= step
        current_z[0] = max(current_z[0], -1)
        slider_widget.GetRepresentation().SetValue(current_z[0])
        update_slice(current_z[0])

    def move_right():
        current_z[0] += step
        current_z[0] = min(current_z[0], 1)
        slider_widget.GetRepresentation().SetValue(current_z[0])
        update_slice(current_z[0])

    # 綁定鍵盤事件
    plotter.add_key_event("Left", move_left)
    plotter.add_key_event("Right", move_right)

    plotter.show()


def main2():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPLX_NEUTRAL_MODEL_PATH,
        model_config=smplx_utils.smplx_model_config,
        dtype=torch.float64,
        device=DEVICE,
    )

    J = model_data.kin_tree.joints_cnt

    model_builder = smplx_utils.DeformableModelBuilder(model_data=model_data)

    x, y, z = torch.meshgrid(
        torch.linspace(-2, 2, 50, dtype=utils.FLOAT, device=DEVICE),
        torch.linspace(-2, 2, 50, dtype=utils.FLOAT, device=DEVICE),
        torch.linspace(-2, 2, 50, dtype=utils.FLOAT, device=DEVICE),
        indexing="ij",
    )
    # [N]

    print(f"{x.shape=}")

    v = model_builder.lbs_weight_interp(
        torch.stack([x, y, z], dim=-1).reshape(-1, 3))[:, :9]
    # [N, C]

    print(f"{v.shape=}")

    C = v.shape[-1]

    grid = pv.StructuredGrid(
        x.numpy(force=True),
        y.numpy(force=True),
        z.numpy(force=True),
    )

    for i in range(C):
        grid[f"component_{i}"] = v[..., i].numpy(force=True)

    subplot_shape = (3, 3)

    # 更新切片函數
    plotter = pv.Plotter(shape=subplot_shape)

    for i in range(C):
        plotter.subplot(
            i // subplot_shape[1],
            i % subplot_shape[1],
        )

    slice_z_value = 0.0

    # 更新切片的函數
    def update_slice():
        nonlocal slice_z_value

        for i in range(C):
            component = f"component_{i}"

            slice_plane = grid.slice(normal="z", origin=(0, 0, slice_z_value))

            plotter.subplot(
                i // subplot_shape[1],
                i % subplot_shape[1],
            )

            plotter.add_mesh(
                slice_plane,
                scalars=component,
                cmap="plasma",
                show_edges=False
            )

            plotter.view_xy()

        plotter.render()

    # 初始切片顯示
    update_slice()

    # 添加控制切片的滑動條
    slider_widget = plotter.add_slider_widget(
        callback=update_slice,
        rng=[-1, 1],
        value=0.0,
        title="Slice Z",
        style="modern"
    )

    # 定義鍵盤事件來改變切片位置
    def on_left_event():
        nonlocal slice_z_value

        step = 0.05  # 每次調整的步長

        slice_z_value -= step

        # 更新切片
        update_slice()

    def on_right_event():
        nonlocal slice_z_value

        step = 0.05  # 每次調整的步長

        slice_z_value += step

        # 更新切片
        update_slice()

    # 註冊鍵盤事件
    plotter.add_key_event("Left", on_left_event)
    plotter.add_key_event("Right", on_right_event)

    # 顯示
    plotter.show()


def main3():
    gart_result = gart_utils.read_gart_result(
        gart_log_dir=config.GART_DIR / "logs",
        train_name="train_2025_0426_2",
        dtype=torch.float,
        device=DEVICE,
    )

    temp_model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_NEUTRAL_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=torch.float64,
        device=DEVICE,
    )

    J = temp_model_data.kin_tree.joints_cnt
    V = temp_model_data.verts_cnt
    BS = temp_model_data.body_shapes_cnt

    print(f"{gart_result.gp_mean.shape=}")

    canonical_gp_mean, err = smplx_utils.shape_canon(
        temp_model_data,

        gart_result.body_shape,

        utils.zeros_like(
            gart_result.body_shape, shape=(temp_model_data.expr_shapes_cnt,)),

        gart_result.gp_mean,
    )

    temp_model_builder = smplx_utils.DeformableModelBuilder(
        temp_model_data, temp_model_data)

    blending_coeff = temp_model_builder.query_blending_coeff(
        vert_pos=canonical_gp_mean)

    print(f"{gart_result.gp_mean.shape=}")
    print(f"{canonical_gp_mean.shape=}")
    print(f"{err=}")

    GP_V = canonical_gp_mean.shape[0]

    print(f"{GP_V=}")

    gart_model_data = smplx_utils.ModelData(
        kin_tree=temp_model_data.kin_tree,

        body_joints_cnt=temp_model_data.body_joints_cnt,
        jaw_joints_cnt=temp_model_data.jaw_joints_cnt,
        eye_joints_cnt=temp_model_data.eye_joints_cnt,

        mesh_graph=mesh_utils.MeshGraph.empty(
            GP_V, temp_model_data.mesh_graph.device),
        tex_mesh_graph=mesh_utils.MeshGraph.empty(
            0, temp_model_data.mesh_graph.device),

        joint_t_mean=temp_model_data.joint_t_mean,

        vert_pos=canonical_gp_mean,
        tex_vert_pos=utils.zeros_like(
            temp_model_data.tex_vert_pos, shape=(0, 2)),

        lbs_weight=blending_coeff.lbs_weight,

        body_shape_joint_dir=temp_model_data.body_shape_joint_dir,
        expr_shape_joint_dir=temp_model_data.expr_shape_joint_dir,

        body_shape_vert_dir=blending_coeff.body_shape_vert_dir,
        expr_shape_vert_dir=blending_coeff.expr_shape_vert_dir,

        lhand_pose_mean=temp_model_data.lhand_pose_mean,
        rhand_pose_mean=temp_model_data.rhand_pose_mean,

        pose_vert_dir=blending_coeff.pose_vert_dir,
    )

    # gart_model_data.show()

    gart_model_builder = smplx_utils.StaticModelBuilder(gart_model_data)

    gart_model_blender: smplx_utils.ModelBlender = smplx_utils.ModelBlender(
        model_builder=gart_model_builder)

    subject_data: people_snapshot_utils.SubjectData = \
        people_snapshot_utils.read_subject(
            subject_dir=config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
            model_data=temp_model_data,
            device=DEVICE,
        )

    print(f"{subject_data.video.shape=}")

    T, C, H, W = subject_data.video.shape

    out_video_path = DIR / f"rgb_{utils.timestamp_sec()}.avi"

    out_video = vision_utils.VideoWriter(
        path=out_video_path,
        height=H,
        width=W,
        color_type=vision_utils.ColorType.RGB,
        fps=subject_data.fps,
    )

    bg_color = torch.ones((3, ), dtype=utils.FLOAT, device=DEVICE)

    print(f"{gart_result.gp_rot_q.shape=}")

    gp_rot_mat = utils.quaternion_to_rot_mat(
        gart_result.gp_rot_q, order="WXYZ", out_shape=(3, 3))
    # [GP_V, 3, 3]

    print(f"{gp_rot_mat.shape=}")

    print(f"{gart_result.gp_scale.shape=}")

    gp_scale_mat = utils.make_diag(gart_result.gp_scale)
    # [GP_V, 3, 3]

    print(f"{gart_result.gp_scale.min()=}")
    print(f"{gart_result.gp_scale.max()=}")

    print(f"{gp_scale_mat.shape=}")
    print(f"{gp_scale_mat.max()=}")
    print(f"{gp_scale_mat.min()=}")

    print(f"{gp_scale_mat[0]=}")
    print(f"{gp_scale_mat[1]=}")

    print(f"{gart_result.gp_color.min()=}")
    print(f"{gart_result.gp_color.max()=}")

    with torch.no_grad():
        for t in tqdm.tqdm(range(T)):
            print(f"{t=}")

            model: smplx_utils.Model = gart_model_blender(
                subject_data.blending_param[t])

            print(f"{model.vert_trans.shape=}")

            vert_trans_r = model.vert_trans[..., :3, :3]

            re_vert_pos = utils.do_rt(
                model.vert_trans[..., :3, :3],
                model.vert_trans[..., :3, 3],
                gart_model_data.vert_pos,
            )

            diff = (re_vert_pos - model.vert_pos).square().mean().sqrt()
            print(f"{diff=}")

            cur_gp_trans_r = utils.mat_mul(
                vert_trans_r, gp_rot_mat, gp_scale_mat)

            print(f"{cur_gp_trans_r.shape}")

            gp_cov3d = utils.mat_mul(
                cur_gp_trans_r, cur_gp_trans_r.transpose(-2, -1))

            print(f"{gp_cov3d.det().min()=}")

            gp_render_result = gaussian_utils.render_gaussian(
                camera_config=subject_data.camera_config,
                camera_transform=subject_data.camera_transform,

                sh_degree=0,
                bg_color=bg_color,

                gp_mean=model.vert_pos,

                gp_rot_q=None,
                gp_scale=None,
                gp_cov3d=gp_cov3d,
                gp_cov3d_u=None,

                gp_sh=None,
                gp_color=gart_result.gp_color,

                gp_opacity=gart_result.gp_opacity,

                device=DEVICE,
            )

            print(f"{gp_render_result.colors.shape=}")

            assert gp_render_result.colors.isfinite().all()

            print(f"{gp_render_result.colors.min()=}")
            print(f"{gp_render_result.colors.max()=}")

            out_video.write(vision_utils.denormalize_image(
                gp_render_result.colors))

    out_video.close()

    print(f"{out_video_path=}")


def main4():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_NEUTRAL_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=torch.float64,
        device=DEVICE,
    )

    J = model_data.kin_tree.joints_cnt
    V = model_data.verts_cnt
    BS = model_data.body_shapes_cnt

    body_shape_vert_dir_interp = torchrbf.RBFInterpolator(
        y=model_data.vert_pos.cpu().to(torch.float64),  # [V, 3]
        d=model_data.body_shape_vert_dir.cpu().reshape(V, 3 * BS).to(torch.float64),
        smoothing=1.0,
        kernel="cubic",
        degree=2,
    ).to(DEVICE, torch.float64)

    my_body_shape_vert_dir_interp = \
        rbf_utils.interp_utils.RBFInterpolator.from_data_point(
            data_pos=model_data.vert_pos.to(torch.float32),  # [V, 3]
            data_val=model_data.body_shape_vert_dir.reshape(
                V, 3 * BS).to(torch.float32),
            kernel=rbf_utils.radial_func.CubicRadialFunc(),
            degree=2,
            smoothness=1.0,
        ).to(DEVICE, torch.float64)

    data_x = model_data.vert_pos
    data_y = model_data.body_shape_vert_dir.reshape(V, 3 * BS)

    re_data_y = body_shape_vert_dir_interp(data_x)
    my_re_data_y = my_body_shape_vert_dir_interp(data_x)

    print(f"m = {(re_data_y - data_y).square().mean().sqrt():.6e}")
    print(f"my_m = {(my_re_data_y - data_y).square().mean().sqrt():.6e}")

    point_pos = torch.normal(
        mean=0.0, std=1.0, size=(1000, 3), dtype=torch.float64, device=DEVICE)

    result_a = body_shape_vert_dir_interp(point_pos)
    result_b = my_body_shape_vert_dir_interp(point_pos)

    print(f"{result_a.shape=}")
    print(f"{result_b.shape=}")

    diff = result_a - result_b

    rel_diff = (diff.abs() / (result_a.abs() + 1e-2))

    print(f"max abs diff = {diff.abs().max():.6e}")
    print(f"mean abs diff = {diff.abs().mean():.6e}")
    print(f"rel_diff = {rel_diff.max():.6e}")


if __name__ == "__main__":
    main4()
