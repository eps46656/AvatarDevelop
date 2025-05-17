import dataclasses
import math
import typing

import beartype
import einops
import torch
import torchvision
import tqdm
from beartype import beartype

from . import (camera_utils, config, mesh_utils, people_snapshot_utils,
               smplx_utils, tex_avatar_utils, training_utils,
               transform_utils, utils, video_seg_utils, vision_utils)

SEG_DIR = config.DIR / "video_seg_2025_0514_1"

MESH_SEG_DIR = config.DIR / "mesh_seg_2025_0514_1"

DTYPE = torch.float64
DEVICE = utils.CUDA_DEVICE

SUBJECT_NAME = "female-1-casual"

VIDEO_SEG_DIR = config.DIR / "video_seg_2025_0514_1"

MESH_SEG_DIR = config.DIR / "mesh_seg_2025_0514_1"

TEX_AVATAR_DIR = config.DIR / "tex_avatar_2025_0516_1"


@beartype
@dataclasses.dataclass
class AvatarPack:
    avatar_blender: smplx_utils.ModelBlender

    camera_config: camera_utils.CameraConfig
    camera_transform: transform_utils.ObjectTransform

    blending_param: smplx_utils.BlendingParam

    img: torch.Tensor  # [..., C, H, W]
    fps: float


@beartype
def load_avatar_pack():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_FEMALE_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=DTYPE,
        device=DEVICE,
    )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        model_data=model_data,
        device=DEVICE,
    )

    model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_utils.StaticModelBuilder(
            model_data=model_data,
        )
    )

    return AvatarPack(
        avatar_blender=model_blender,

        camera_config=subject_data.camera_config,
        camera_transform=subject_data.camera_transform,

        blending_param=subject_data.blending_param,

        img=subject_data.video,
        fps=subject_data.fps,
    )


def main1():
    avatar_pack = load_avatar_pack()

    avatar_blender = avatar_pack.avatar_blender
    camera_config = avatar_pack.camera_config
    camera_transform = avatar_pack.camera_transform
    blending_param = avatar_pack.blending_param
    img = avatar_pack.img
    fps = avatar_pack.fps

    init_tex = torch.rand(
        (3, 1000, 1000), dtype=DTYPE, device=DEVICE)

    trainer = training_utils.Trainer(
        TEX_AVATAR_DIR,
        tex_avatar_utils.TrainerCore(
            config=tex_avatar_utils.TrainerCoreConfig(
                proj_dir=TEX_AVATAR_DIR,
                device=DEVICE,
                batch_size=8,

                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                num_inference_steps=20,

                init_ref_imgs_cnt=3,
                ref_img_gamma=0.1**(1 / 12),

                lr=1e-4,
                betas=(0.5, 0.5),
                gamma=0.95,
            ),

            avatar_blender=avatar_blender,

            dataset=tex_avatar_utils.Dataset(
                tex_avatar_utils.Sample(
                    camera_config=camera_config,
                    camera_transform=camera_transform,

                    img=img,

                    person_mask=vision_utils.read_video_mask(
                        VIDEO_SEG_DIR / video_seg_utils.REFINED_PERSON_MASK_FILENAME,
                        dtype=DTYPE,
                        device=utils.CPU_DEVICE,
                        disk_mem=True,
                    )[0],

                    skin_mask=vision_utils.read_video_mask(
                        VIDEO_SEG_DIR / "refined_mask_SKIN.avi",
                        dtype=DTYPE,
                        device=utils.CPU_DEVICE,
                        disk_mem=True,
                    )[0],

                    blending_param=blending_param,
                )
            ),

            text_prompt="a female with underwear, realistic, photorealistic, natural skin texture, high detail",
            negative_text_prompt="anime, cartoon, illustration, drawing, painting, sketch, unrealistic, 3d render, cgi, low quality, low resolution, blurry, extra limbs, mutated hands, deformed, bad anatomy",

            init_tex=init_tex,
        )
    )

    trainer.enter_cli()


if __name__ == "__main__":
    main1()
