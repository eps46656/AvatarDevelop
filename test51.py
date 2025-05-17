import collections
import typing

import diffusers
import math
import einops
import torch
import torchvision
import tqdm
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

    prompt = "a photo of a women"

    optimizer = torch.optim.Adam(
        [img],
        lr=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.99,
        last_epoch=-1,
    )

    @torch.no_grad
    def _step():
        nonlocal img

        print(f"{id(img)=}")

        print(f"{img.requires_grad=}")

        assert img.grad is not None

        if img.grad is None:
            print(f"img.grad is None")
            return

        step_tensor = (img.grad * 1e2).clamp(-5e-3, 5e-3)

        print(f"step norm: {step_tensor.norm()}")
        print(f"avg step norm: {step_tensor.abs().mean()}")

        img -= step_tensor

        img.clamp_(0, 1)

        img.grad = None

    batch_size = 8

    for i in range(1000):
        for b in range(batch_size):
            timestep = torch.randint(10, 100, (1,), device=utils.CUDA_DEVICE)

            cur_sds_result: controlnet_utils.SDSLossResult = \
                controlnet_utils.calc_sds_loss(
                    pipe=pipe,

                    prompt=prompt,
                    negative_prompt=None,

                    img=img,
                    img_latent=None,

                    control_img=normal_map,

                    timestep=timestep,

                    guidance_scale=7.5,

                    controlnet_conditioning_scale=1.0,
                )

            # assert cur_sds_result.noise_pred.isfinite().all()
            # assert cur_sds_result.noise_sample.isfinite().all()

            img_latent = cur_sds_result.img_latent
            grad = cur_sds_result.grad.nan_to_num()
            grad = grad.clamp(-5e-3, 5e-3)

            print(f"grad norm: {grad.norm()}")

            cur_sds_loss = 0.5 * torch.nn.functional.mse_loss(
                img_latent,
                (img_latent - grad).detach(),
                reduction="sum",
            ) / batch_size

            print(f"{cur_sds_loss=}")
            print(f"{cur_sds_loss.grad_fn=}")

            optimizer.zero_grad()
            cur_sds_loss.backward(retain_graph=True)

            img_grad = img.grad

            print(f"{img_grad.norm()=}")
            print(f"{img_grad.abs().mean()=}")

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            img.nan_to_num_().clamp_(0, 1)

    vision_utils.write_image(
        config.DIR / f"test_img_{utils.timestamp_sec()}.png",
        utils.rct(img * 255, dtype=torch.uint8),
    )


def main2():
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
    )(normal_map).detach().to(utils.CUDA_DEVICE, torch.float16).requires_grad_()

    C, H, W = normal_map.shape

    print(f"{C=}")
    print(f"{H=}")
    print(f"{W=}")

    """
    img_latent = controlnet_utils.encode_img(
        pipe=pipe,
        img=normal_map,
    ).to(utils.CUDA_DEVICE, torch.float16)


    """

    img = (vision_utils.read_image(
        config.DIR / "test_img_1746975112.png"
    ) / 255).to(utils.CUDA_DEVICE, torch.float16)

    """
    img = torch.rand(
        (C, H, W),
        dtype=torch.float16,
        device=utils.CUDA_DEVICE,
        requires_grad=True,
    )
    """

    img = img.detach().clone().requires_grad_()

    prompt = "a photo of a women"

    lr = 2e0

    tiemstep_beg = 400
    tiemstep_end = 200

    batch_size = 8

    epochs_cnt = 1000

    for i in tqdm.tqdm(range(epochs_cnt)):
        assert img.isfinite().all()

        progress = i / epochs_cnt

        cur_timestep_base = torch.tensor(
            max(1, round(
                (tiemstep_end - tiemstep_beg) * (
                    math.cos(math.pi * progress) * -0.5 + 0.5
                ) + tiemstep_beg
            )),

            dtype=torch.int32, device=utils.CUDA_DEVICE,
        )

        for b in range(batch_size):
            cur_timestep = torch.randint(
                1, 200, (1,), dtype=torch.int32, device=utils.CUDA_DEVICE)

            cur_sds_result: controlnet_utils.SDSLossResult = \
                controlnet_utils.calc_sds_loss(
                    pipe=pipe,

                    prompt=prompt,
                    negative_prompt=None,

                    img=img,
                    img_latent=None,

                    control_img=normal_map,

                    timestep=cur_timestep,

                    guidance_scale=20.0,

                    controlnet_conditioning_scale=1.0,
                )

            # assert cur_sds_result.noise_pred.isfinite().all()
            # assert cur_sds_result.noise_sample.isfinite().all()

            img_latent = cur_sds_result.img_latent

            cur_sds_loss = 0.5 * torch.nn.functional.mse_loss(
                img_latent,
                (img_latent - cur_sds_result.grad).detach(),
                reduction="sum",
            )

            cur_sds_loss.backward(retain_graph=True)

            with torch.no_grad():
                img.grad.nan_to_num_()

        with torch.no_grad():
            img = (img - img.grad * (lr / batch_size))
            img.nan_to_num_().clamp_(0, 1)
            img.grad = None

            img.requires_grad_()

        if i % 100 == 0:
            vision_utils.write_image(
                config.DIR / f"test_img_{utils.timestamp_sec()}.png",
                utils.rct(img * 255, dtype=torch.uint8),
            )

    vision_utils.write_image(
        config.DIR / f"test_img_{utils.timestamp_sec()}.png",
        utils.rct(img * 255, dtype=torch.uint8),
    )


def main3():
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
    )(normal_map).detach().to(utils.CUDA_DEVICE, torch.float16).requires_grad_()

    C, H, W = normal_map.shape

    print(f"{C=}")
    print(f"{H=}")
    print(f"{W=}")

    print(f"{pipe.vae_scale_factor=}")

    img_latent = torch.randn(
        (1, 4, H // pipe.vae_scale_factor, W // pipe.vae_scale_factor),
        dtype=torch.float16,
        device=utils.CUDA_DEVICE,
    ) * pipe.scheduler.init_noise_sigma

    """
    img = (vision_utils.read_image(
        config.DIR / "test_img_1746877885.png"
    ) / 255).to(utils.CUDA_DEVICE, torch.float16)
    """

    print(f"{id(img_latent)=}")

    print(f"{img_latent.shape=}")
    print(f"{img_latent.grad_fn=}")
    print(f"{img_latent.min()=}")
    print(f"{img_latent.max()=}")

    prompt = "a photo of a women"

    pipe.scheduler.set_timesteps(50, device=utils.CUDA_DEVICE)
    timesteps = pipe.scheduler.timesteps

    pipe._num_timesteps = len(timesteps)

    for timestep in timesteps:
        nxt_img_latent = controlnet_utils.calc_nxt_img_latent(
            pipe=pipe,

            prompt=prompt,
            negative_prompt=None,

            img_latent=img_latent,

            control_img=normal_map,

            timestep=timestep,

            guidance_scale=7.5,

            controlnet_conditioning_scale=1.0,
        )

        img_latent = nxt_img_latent

    vision_utils.write_image(
        config.DIR / f"test_img_{utils.timestamp_sec()}.png",
        utils.rct(controlnet_utils.decode_img(
            pipe, img_latent)[0, ...] * 255, dtype=torch.uint8),
    )


if __name__ == "__main__":
    main2()
