import typing

import einops
import torch
import tqdm
from beartype import beartype

from . import (avatar_utils, camera_utils, config, mesh_utils,
               people_snapshot_utils, rendering_utils, smplx_utils,
               transform_utils, utils, vision_utils)

SUBJECT_NAME = "female-1-casual"


DEVICE = utils.CUDA_DEVICE


def read_subject(model_data: typing.Optional[smplx_utils.ModelData] = None):
    if model_data is None:
        if "female" in SUBJECT_NAME:
            model_data_path = config.SMPL_FEMALE_MODEL_PATH
        else:
            model_data_path = config.SMPL_MALE_MODEL_PATH

        model_data = smplx_utils.ModelData.from_origin_file(
            model_data_path=model_data_path,
            model_config=smplx_utils.smpl_model_config,
            dtype=utils.FLOAT,
            device=DEVICE,
        )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME, model_data, DEVICE)

    return subject_data


def main1():
    subject_data = read_subject()

    smplx_model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_utils.DeformableModelBuilder(
            temp_model_data=subject_data.model_data,
            model_data=subject_data.model_data,
        )
    )

    img = subject_data.video
    # [T, C, H, W]

    T, C, H, W = img.shape

    video_writer = vision_utils.VideoWriter(
        config.DIR / f"nor_{utils.timestamp_sec()}.avi",
        height=subject_data.camera_config.img_h,
        width=subject_data.camera_config.img_w,
        color_type=vision_utils.ColorType.RGB,
        fps=subject_data.fps,
    )

    # subject_data.camera_transform
    # camera <-> frame

    opengl_proj_config: camera_utils.ProjConfig = \
        camera_utils.make_proj_config_OpenGL(
            camera_config=subject_data.camera_config,
            target_coord=camera_utils.Coord.NDC,
        )

    opengl_camera_proj_transform = opengl_proj_config.camera_proj_transform.to(
        DEVICE)
    # camera <-> view

    for t in tqdm.tqdm(range(T)):
        avatar_model: avatar_utils.AvatarModel = smplx_model_blender(
            subject_data.blending_param[t])

        normal_map = rendering_utils.make_normal_map(
            camera_config=subject_data.camera_config,
            camera_transform=subject_data.camera_transform,
            mesh_data=avatar_model.mesh_data,

            pix_to_face=None,
            bary_coord=None,
        )
        # [H, W, 3]

        # normal_map = (frame_to_view[:3, :3] @ normal_map[..., None])[..., 0]

        video_writer.write(vision_utils.denormalize_image(einops.rearrange(
            normal_map, "h w c -> c h w")))

    video_writer.close()


if __name__ == "__main__":
    main1()
