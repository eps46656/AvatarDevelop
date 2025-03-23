import pathlib

import torch
import einops

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures

from . import (config, people_snapshot_utils, smplx_utils, transform_utils,
               utils, camera_utils)

FILE = pathlib.Path(__file__)
DIR = FILE.parents[0]

DEVICE = torch.device("cuda")


class AlbedoMeshShader(pytorch3d.renderer.mesh.shader.ShaderBase):
    def forward(
        self,
        fragments: pytorch3d.renderer.mesh.rasterizer.Fragments,
        meshes: pytorch3d.structures.Meshes,
        **kwargs,
    ):
        texels = meshes.sample_textures(fragments)

        print(f"{type(texels)=}")
        print(f"{texels.shape=}")

        return texels


def main1():
    smpl_model_data_path_dict = {
        "male": config.SMPL_MALE_MODEL,
        "female": config.SMPL_FEMALE_MODEL,
        "neutral": config.SMPL_NEUTRAL_MODEL,
    }

    smplx_model_data_path_dict = {
        "male": config.SMPLX_MALE_MODEL,
        "female": config.SMPLX_FEMALE_MODEL,
        "neutral": config.SMPLX_NEUTRAL_MODEL,
    }

    body_shapes_cnt = 10
    expr_shapes_cnt = 0
    body_joints_cnt = 24
    jaw_joints_cnt = 0
    eye_joints_cnt = 0
    hand_joints_cnt = 0

    model_data_dict = {
        key: smplx_utils.ReadSMPLXModelData(
            model_data_path=value,
            body_shapes_cnt=body_shapes_cnt,
            expr_shapes_cnt=expr_shapes_cnt,
            body_joints_cnt=body_joints_cnt,
            jaw_joints_cnt=jaw_joints_cnt,
            eye_joints_cnt=eye_joints_cnt,
            hand_joints_cnt=hand_joints_cnt,
            device=DEVICE,
        )
        for key, value in smpl_model_data_path_dict.items()
    }

    people_snapshot_dir = DIR / "people_snapshot_public"

    subject_name = "female-1-casual"

    subject_dir = people_snapshot_dir / subject_name

    subject_data = people_snapshot_utils.ReadSubject(
        subject_dir=subject_dir,
        model_data_dict=model_data_dict,
        device=DEVICE,
    )

    camera_config = subject_data.camera_config

    # ---

    smplx_builder = smplx_utils.SMPLXModelBuilder(
        model_data=subject_data.model_data,
        device=DEVICE,
    )

    # ---

    camera_view_transform = transform_utils.ObjectTransform.FromMatching(
        "LUF")
    # camera <-> view

    # subject_data.camera_transform
    # camera <-> world

    world_to_view_mat = subject_data.camera_transform.GetTransTo(
        camera_view_transform)
    # world -> view

    # subject_data.blending_param.global_transl
    # view -> model

    # ---

    print(f"{camera_config=}")

    print(f"{camera_config.foc_u=}")
    print(f"{camera_config.foc_d=}")
    print(f"{camera_config.foc_l=}")
    print(f"{camera_config.foc_r=}")
    print(f"{camera_config.depth_near=}")
    print(f"{camera_config.depth_far=}")
    print(f"{camera_config.img_h=}")
    print(f"{camera_config.img_w=}")

    camera_proj_mat = camera_utils.MakeProjMat(
        camera_config=camera_config,
        camera_view_transform=camera_view_transform,
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
        dtype=utils.FLOAT,
    )

    cameras = pytorch3d.renderer.PerspectiveCameras(
        R=world_to_view_mat[:3, :3].transpose(0, 1).unsqueeze(0),
        T=world_to_view_mat[:3, 3].unsqueeze(0),
        K=camera_proj_mat.transpose(0, 1).unsqueeze(0),
        in_ndc=True,
        device=DEVICE,
    )

    print(f"{world_to_view_mat=}")
    print(f"{camera_proj_mat=}")

    # ---

    smplx_model: smplx_utils.SMPLXModel = smplx_builder.forward(
        subject_data.blending_param)

    print(f"{smplx_model.vertex_positions.shape}")

    # smplx_model.joint_Ts[T, J, 4, 4]

    T = smplx_model.vertex_positions.shape[0]

    out_frames = torch.empty(
        (T, 3,
         subject_data.camera_config.img_h, subject_data.camera_config.img_w)
    )

    for frame_i in range(T):
        print(f"{frame_i=}")

        # smplx_model.vertex_positions[T, V, 3]
        # smplx_model.vertex_positions[F, 3]

        # smplx_model.vertex_positions[T, V, 3]
        V = smplx_model.vertex_positions.shape[1]

        if False:
            mesh = pytorch3d.structures.Meshes(
                verts=[smplx_model.vertex_positions[frame_i, :, :]],
                faces=[smplx_model.faces],
            )

            lights = pytorch3d.renderer.lighting.PointLights(
                ambient_color=[[1.0, 1.0, 1.0]],
                location=[[0.0, 0.0, 3.0]],
                device=DEVICE,
            )

            raster_settings = pytorch3d.renderer.RasterizationSettings(
                image_size=(camera_config.img_h, camera_config.img_w),
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            renderer = pytorch3d.renderer.MeshRenderer(
                rasterizer=pytorch3d.renderer.MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings,
                ),
                shader=pytorch3d.renderer.mesh.shader.SoftPhongShader(
                    cameras=cameras,
                    lights=lights,
                    device=DEVICE,
                ),
            )

            img = renderer(
                mesh,
                image_size=(camera_config.img_h, camera_config.img_w),
            )
        else:
            point_cloud = pytorch3d.structures.Pointclouds(
                points=[
                    smplx_model.vertex_positions[frame_i, :, :]
                ],

                features=[
                    torch.tensor(
                        [1.0, 0.0, 0.0],
                        dtype=utils.FLOAT, device=DEVICE,
                    ).expand((V, 3))
                ],
            )

            raster_settings = pytorch3d.renderer.PointsRasterizationSettings(
                image_size=(camera_config.img_h, camera_config.img_w),
                radius=0.01,
                points_per_pixel=10,
            )

            rasterizer = pytorch3d.renderer.PointsRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
            )

            renderer = pytorch3d.renderer.PointsRenderer(
                rasterizer=rasterizer,
                compositor=pytorch3d.renderer.points.compositor.AlphaCompositor(
                    background_color=(1.0, 1.0, 1.0),
                ),
            )

            img = renderer(point_cloud).squeeze(0)
            # 1 h w c

        # ---

        print(f"{img.shape=}")

        out_frames[frame_i, :, :, :] = einops.rearrange(
            img, "h w c -> c h w")

    utils.WriteVideo(
        path=DIR / "output.mp4",
        video=(out_frames * 255).round().to(dtype=torch.uint8).clamp(0, 255),
        order="t c h w",
        fps=30,
    )


if __name__ == "__main__":
    main1()

    print("ok")
