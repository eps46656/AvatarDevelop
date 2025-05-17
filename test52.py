import collections
import typing

import torch
import einops
import torchvision
from beartype import beartype

from . import config, controlnet_utils, utils, vision_utils


def main1():
    pipe = controlnet_utils.load_controlnet_pipe(
        controlnet_model=controlnet_utils.ControlNetModel.NORMAL,
        dtype=torch.float16,
        device=utils.CUDA_DEVICE,
    )

    normal_map = vision_utils.read_image(
        config.DIR / "test_normal_map.png") / 255

    print(f"{normal_map.shape=}")
    print(f"{normal_map.min()=}")
    print(f"{normal_map.max()=}")

    normal_map = torchvision.transforms.Resize(
        (512, 512),
        torchvision.transforms.InterpolationMode.BILINEAR,
    )(normal_map).detach().requires_grad_()

    C, H, W = normal_map.shape

    print(f"{C=}")
    print(f"{H=}")
    print(f"{W=}")

    img = torch.rand(
        (C, H, W),
        dtype=torch.float16,
        device=utils.CUDA_DEVICE,
        requires_grad=True,
    )

    """
    img = (vision_utils.read_image(
        config.DIR / "test_img_1746877885.png"
    ) / 255).to(utils.CUDA_DEVICE, torch.float16)
    """

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
