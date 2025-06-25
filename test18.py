import pathlib

import einops
import torch
import tqdm

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures

from . import (camera_utils, config, people_snapshot_utils, rendering_utils,
               smplx_utils, transform_utils, utils, vision_utils)

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
        "male": config.SMPL_MALE_MODEL_PATH,
        "female": config.SMPL_FEMALE_MODEL_PATH,
        "neutral": config.SMPL_NEUTRAL_MODEL_PATH,
    }

    smplx_model_data_path_dict = {
        "male": config.SMPLX_MALE_MODEL_PATH,
        "female": config.SMPLX_FEMALE_MODEL_PATH,
        "neutral": config.SMPLX_NEUTRAL_MODEL_PATH,
    }

    model_data_dict = {
        key: smplx_utils.Core.from_file(
            model_data_path=value,
            model_config=smplx_utils.smpl_model_config,
            device=DEVICE,
        )
        for key, value in smpl_model_data_path_dict.items()
    }

    people_snapshot_dir = DIR / "people_snapshot_public"

    subject_name = "female-1-casual"

    subject_dir = people_snapshot_dir / subject_name

    subject_data: people_snapshot_utils.SubjectData = \
        people_snapshot_utils.read_subject(
            subject_dir=subject_dir,
            model_data_dict=model_data_dict,
            device=DEVICE,
        )

    camera_config = subject_data.camera_config

    # ---

    smplx_model_builder = smplx_utils.StaticModelBuilder(
        model_data=subject_data.model_data,
    )

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_model_builder,
    )

    # ---

    camera_view_transform = transform_utils.ObjectTransform.from_matching(
        "LUF")
    # camera <-> view

    # subject_data.camera_transform
    # camera <-> world

    world_to_view_mat = subject_data.camera_transform.get_trans_to(
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

    camera_proj_mat = camera_utils.make_proj_mat(
        camera_config=camera_config,
        camera_transform=camera_view_transform,
        convention=camera_utils.Convention.PyTorch3D,
        target_coord=camera_utils.Coord.NDC,
        dtype=utils.FLOAT,
    )

    print(f"{camera_proj_mat=}")

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

    smplx_model: smplx_utils.Model = smplx_model_blender.forward(
        subject_data.blending_param)

    print(f"{smplx_model.vert_pos.shape}")

    # smplx_model.joint_Ts[T, J, 4, 4]

    T = smplx_model.vert_pos.shape[0]

    out_frames = torch.empty(
        (T, 3,
         subject_data.camera_config.img_h, subject_data.camera_config.img_w)
    )

    zeros = torch.zeros(
        (1, camera_config.img_h, camera_config.img_w), dtype=utils.FLOAT)
    ones = torch.ones(
        (1, camera_config.img_h, camera_config.img_w), dtype=utils.FLOAT)

    for frame_i in tqdm.tqdm(range(T)):
        # smplx_model.vertex_positions[T, V, 3]
        # smplx_model.vertex_positions[F, 3]

        # smplx_model.vertex_positions[T, V, 3]
        V = smplx_model.vert_pos.shape[1]

        if True:
            mesh_ras_result = rendering_utils.rasterize_mesh(
                vert_pos=smplx_model.vert_pos[frame_i],
                faces=smplx_model.mesh_graph.f_to_vvv,
                camera_config=camera_config,
                camera_transform=subject_data.camera_transform.to(
                    utils.CUDA_DEVICE),
                faces_per_pixel=1,
            )

            out_frames[frame_i] = torch.where(
                (mesh_ras_result["pixel_to_faces"][:, :, 0] == -1)
                .to(out_frames.device),

                zeros,
                ones,
            )
        else:
            point_cloud = pytorch3d.structures.Pointclouds(
                points=[
                    smplx_model.vert_pos[frame_i, :, :]
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

            out_frames[frame_i, :, :, :] = einops.rearrange(
                img, "h w c -> c h w")

    vision_utils.write_video(
        path=DIR / "output.mp4",
        video=out_frames,
        fps=25,
    )


if __name__ == "__main__":
    main1()

    print("ok")
