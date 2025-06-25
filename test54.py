import typing

import einops
import torch
import torchvision
import tqdm
from beartype import beartype

from . import (avatar_utils, camera_utils, config, controlnet_utils,
               mesh_utils, people_snapshot_utils, rendering_utils,
               sds_texture_avatar_trainer, smplx_utils, training_utils,
               transform_utils, utils, vision_utils, bare_avatar_utils)

SUBJECT_NAME = "female-1-casual"

PROJ_DIR = config.DIR / "sds_2025_0506_1"


DTYPE = utils.FLOAT
DEVICE = utils.CUDA_DEVICE


def main1():
    model_data = smplx_utils.ModelData.from_origin_file(
        model_data_path=config.SMPL_FEMALE_MODEL_PATH,
        model_config=smplx_utils.smpl_model_config,
        dtype=DTYPE,
        device=DEVICE,
    )

    subject_data = people_snapshot_utils.read_subject(
        config.PEOPLE_SNAPSHOT_DIR / SUBJECT_NAME,
        model_data=model_data,
        device=utils.CPU_DEVICE,
    )

    model_blender = smplx_utils.ModelBlender(
        model_builder=smplx_utils.StaticModelBuilder(
            model_data=model_data,
        )
    )

    T, C, H, W = subject_data.video.shape

    tex_h = 512
    tex_w = 512

    normal_map = vision_utils.normalize_image(
        vision_utils.read_video(
            config.DIR / "nor_1746349999.avi",
            "RGB",
        )[0],
        dtype=DTYPE,
        device=utils.CPU_DEVICE,
    )

    init_tex = (torch.randn(
        (C, tex_h, tex_w), device=DEVICE
    ) * 0.2 + 0.5).clamp(0, 1).to(DEVICE, torch.float16)

    """
    init_tex = torchvision.transforms.Resize(
        (tex_h, tex_w),
        torchvision.transforms.InterpolationMode.BILINEAR,
    )(subject_data.tex / 255.0).to(DEVICE, torch.float16)
    """

    trainer = training_utils.Trainer(
        PROJ_DIR,
        sds_texture_avatar_trainer.TrainerCore(
            config=sds_texture_avatar_trainer.TrainerCoreConfig(
                proj_dir=PROJ_DIR,
                device=DEVICE,
                batch_size=1,
                noise_samples=8,
                sds_loss_config={
                    controlnet_utils.ControlNetModel.NORMAL: sds_texture_avatar_trainer.SDSLossConfig(
                        alpha=1.0,
                        guidance_scale=20.0,
                        controlnet_conditioning_scale=1.0,
                    ),
                },
            ),

            init_timestep=999,

            avatar_blender=model_blender,

            camera_config=subject_data.camera_config,
            camera_transform=subject_data.camera_transform.to(DEVICE),

            blending_param=subject_data.blending_param.to(DEVICE),

            init_tex=init_tex,

            val_lr=torch.ones((C, tex_h, tex_w), device=DEVICE) * 2e0,

            text_prompt="a naked person",

            control_img={
                controlnet_utils.ControlNetModel.NORMAL: normal_map,
            },
        ),
    )

    trainer.enter_cli()


if __name__ == "__main__":
    main1()
