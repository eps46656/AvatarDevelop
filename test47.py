import typing

import einops
import torch

from . import (avatar_utils, camera_utils, config, controlnet_utils,
               dataset_utils, training_utils, transform_utils, utils,
               vision_utils)

DEVICE = utils.CUDA_DEVICE


def main1():
    pipe = controlnet_utils.load_controlnet_pipe(
        controlnet_utils.ControlNetModel.NORMAL,
        dtype=torch.float16,
        device=DEVICE,
    )

    control_image = vision_utils.read_image(
        config.DIR / "bunny_depth_map.png")

    C, H, W = control_image.shape[-3:]

    pred_noise = controlnet_utils.calc_sds_loss(
        pipe=pipe,
        text_prompt="a cyberpunk bunny",

        img=torch.randn(1, C, H, W).to(DEVICE),

        control_img=control_image,

        timestep=999,
    )

    print(f"{pred_noise.shape=}")


if __name__ == "__main__":
    main1()
