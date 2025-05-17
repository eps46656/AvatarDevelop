import collections
import typing

import torch
import einops
import torchvision
from beartype import beartype

from . import config, controlnet_utils, utils, vision_utils


def main1():
    pipe = controlnet_utils.load_controlnet_pipe(
        controlnet_model=controlnet_utils.ControlNetModel.INPAINT,
        dtype=torch.float16,
        device=utils.CUDA_DEVICE,
    )

    mask = vision_utils.read_image(
        config.DIR / "mask.png") / 255

    print(f"{mask.shape=}")
    print(f"{mask.min()=}")
    print(f"{mask.max()=}")

    mask = mask.mean(dim=0)
    # [H, W]

    # imgs = controlnet_utils.get_inpaint_mask(

    img = img.detach().clone().requires_grad_()

    print(f"{id(img)=}")

    print(f"{img.shape=}")
    print(f"{img.grad_fn=}")
    print(f"{img.min()=}")
    print(f"{img.max()=}")

    text_prompt = "a photo of a women"

    images = pipe(
        prompt=text_prompt,
        negative_prompt=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        image=normal_map[None, ...]
    ).images

    images[0].save(config.DIR / f"test_img_{utils.timestamp_sec()}.png")


if __name__ == "__main__":
    main1()
