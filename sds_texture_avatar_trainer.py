import collections
import dataclasses
import os
import typing
import torchvision

import diffusers
import einops
import torch
import tqdm
from beartype import beartype

import pytorch3d
import pytorch3d.renderer

from . import (avatar_utils, camera_utils, controlnet_utils, dataset_utils,
               rendering_utils, training_utils, transform_utils, utils,
               vision_utils)


@dataclasses.dataclass
class SDSLossConfig:
    alpha: float
    guidance_scale: float
    controlnet_conditioning_scale: float


@dataclasses.dataclass
class TrainerCoreConfig:
    proj_dir: os.PathLike

    device: torch.device

    batch_size: int

    noise_samples: int

    sds_loss_config: dict[
        controlnet_utils.ControlNetModel,
        SDSLossConfig,
    ]


@beartype
class TrainerCore(training_utils.TrainerCore):
    def __init__(
        self,
        *,

        config: TrainerCoreConfig,

        init_timestep: int,

        avatar_blender: avatar_utils.AvatarBlender,

        camera_config: camera_utils.CameraConfig,
        camera_transform: transform_utils.ObjectTransform,

        blending_param: typing.Any,

        init_tex: torch.Tensor,  # [C, tex_h, tex_w]
        val_lr: torch.Tensor,  # [C, tex_h, tex_w]

        text_prompt: str = None,
        negative_text_prompt: str = None,

        control_img: dict[
            controlnet_utils.ControlNetModel,
            torch.Tensor,  # [..., C, H, W]
        ],
    ):
        H, W = camera_config.img_h, camera_config.img_w

        C, tex_h, tex_w = -1, -2, -3

        C, tex_h, tex_w = utils.check_shapes(
            init_tex, (C, tex_h, tex_w),
            val_lr, (C, tex_h, tex_w),
        )

        assert config.sds_loss_config.keys() == control_img.keys()

        assert 0 < len(config.sds_loss_config)

        for img in control_img.values():
            utils.check_shapes(img, (..., C, H, W),)

        self.shape = utils.broadcast_shapes(
            blending_param,
            camera_transform,
            *(img.shape[:-3] for img in control_img.values())
        )

        self.config = dataclasses.replace(config)
        self.config.proj_dir = utils.to_pathlib_path(self.config.proj_dir)

        self.epoch = 0
        self.timestep = init_timestep

        self.avatar_blender = avatar_blender

        self.camera_config = camera_config

        self.camera_transform = camera_transform.expand(self.shape)

        self.blending_param = blending_param.expand(self.shape)

        self.tex = init_tex.detach().clone().requires_grad_()
        self.val_lr = val_lr

        self.text_prompt = text_prompt
        self.negative_text_prompt = negative_text_prompt

        self.control_img = {
            model_name: utils.batch_expand(val, self.shape, 3)
            for model_name, val in control_img.items()
        }

        self.pipe = {
            model_name: self._make_pipe(controlnet_model=model_name)
            for model_name in config.sds_loss_config.keys()
        }

        self.canonical_avatar_model: avatar_utils.AvatarModel = \
            self.avatar_blender.get_avatar_model()

        self.cache_pix_to_face = None
        self.cache_bary_coord = None
        self.cache_tex_coord = None

    # --

    def get_epoch(self) -> int:
        return self.epoch

    # --

    def _make_pipe(self,
                   controlnet_model: controlnet_utils.ControlNetModel) \
            -> diffusers.StableDiffusionControlNetPipeline:
        return controlnet_utils.load_controlnet_pipe(
            controlnet_model=controlnet_model,
            dtype=torch.float16,
            device=self.config.device,
        )

    # --

    def state_dict(self, full: bool) \
            -> collections.OrderedDict[str, typing.Any]:
        l = [
            ("full", full),
            ("epoch", self.epoch),
            ("timestep", self.timestep),
        ]

        if full:
            l.append(("avatar_blender", self.avatar_blender))
        else:
            l.append(("avatar_blender", self.avatar_blender.state_dict()))

        l.append(("tex", self.tex))
        l.append(("val_lr", self.val_lr))
        l.append(("text_prompt", self.text_prompt))
        l.append(("negative_text_prompt", self.negative_text_prompt))

        return collections.OrderedDict(l)

    def load_state_dict(self, state_dict: typing.Mapping[str, typing.Any]) \
            -> None:
        full = state_dict["full"]

        self.epoch = state_dict["epoch"]

        if full:
            self.avatar_blender = state_dict["avatar_blender"]
        else:
            self.avatar_blender.load_state_dict(
                state_dict["avatar_blender"])

        self.tex = state_dict["tex"].detach().requires_grad_()
        self.val_lr = state_dict["val_lr"]
        self.text_prompt = state_dict["text_prompt"]
        self.negative_text_prompt = state_dict["negative_text_prompt"]

    # ---

    @utils.mem_clear
    @torch.no_grad()
    def _prepare_mesh_rasterization_cache(self):
        if self.cache_pix_to_face is not None and \
                self.cache_bary_coord is not None and \
                self.cache_tex_coord is not None:
            return

        H, W = self.camera_config.img_h, self.camera_config.img_w

        self.cache_pix_to_face = utils.disk_empty(
            (*self.shape, H, W), torch.int64)

        self.cache_bary_coord = utils.disk_empty(
            (*self.shape, H, W, 3), torch.float64)

        self.cache_tex_coord = utils.disk_empty(
            (*self.shape, H, W, 2), torch.float64)

        for B, batch_idxes in tqdm.tqdm(dataset_utils.BatchIdxIterator(
            self.shape,
            batch_size=self.config.batch_size,
            shuffle=False,
        )):
            cur_camera_transform = utils.batch_indexing(
                self.camera_transform, self.shape, 0, batch_idxes
            ).to(self.config.device)

            cur_blending_param = utils.batch_indexing(
                self.blending_param, self.shape, 0, batch_idxes
            ).to(self.config.device)

            cur_avatar_model: avatar_utils.AvatarModel = self.avatar_blender(
                cur_blending_param)

            mesh_ras_result = rendering_utils.rasterize_mesh(
                vert_pos=cur_avatar_model.vert_pos.to(self.config.device),

                faces=cur_avatar_model.mesh_graph.f_to_vvv.to(
                    self.config.device),

                camera_config=self.camera_config,
                camera_transform=cur_camera_transform,
                faces_per_pixel=1,
            )

            # mesh_ras_result.pix_to_face[..., H, W, 1]
            # mesh_ras_result.bary_coord[..., H, W, 1, 3]

            pix_to_face = mesh_ras_result.pix_to_face[..., 0]
            # [..., H, W]

            bary_coord = mesh_ras_result.bary_coord[..., 0, :]
            # [..., H, W, 3]

            tex_f_to_vvv = cur_avatar_model.tex_mesh_graph.f_to_vvv.to(
                self.config.device)
            tex_vert_pos = cur_avatar_model.tex_vert_pos.to(
                self.config.device)

            tex_coord = rendering_utils.calc_tex_coord(
                pix_to_face=pix_to_face,
                bary_coord=bary_coord,
                tex_f_to_vvv=tex_f_to_vvv,
                tex_vert_pos=tex_vert_pos,
            )
            # [..., H, W, 2]

            self.cache_pix_to_face[batch_idxes] = pix_to_face.to(
                utils.CPU_DEVICE, torch.int64)
            # [..., H, W]

            self.cache_bary_coord[batch_idxes] = bary_coord.to(
                utils.CPU_DEVICE, torch.float64)
            # [..., H, W, 3]

            self.cache_tex_coord[batch_idxes] = tex_coord.to(
                utils.CPU_DEVICE, torch.float64)
            # [..., H, W, 2]

        # return rasterizer

    # ---

    @utils.mem_clear
    def train(self) -> training_utils.TrainingResult:
        H, W = self.camera_config.img_h, self.camera_config.img_w

        C = self.tex.shape[0]

        bg_color = torch.tensor(
            [1], dtype=torch.float16, device=self.config.device)

        self.tex.grad = None

        @torch.no_grad
        def _step():
            if self.tex.grad is None:
                print(f"tex.grad is None")
                return

            print(f"{self.tex.grad.norm()=}")
            print(f"{self.tex.grad.abs().mean()=}")
            print(f"{self.tex.grad.abs().min()=}")
            print(f"{self.tex.grad.abs().max()=}")

            step_tensor = self.tex.grad * (
                self.val_lr / self.config.noise_samples)

            print(f"{step_tensor.norm()=}")
            print(f"{step_tensor.abs().mean()=}")

            self.tex -= step_tensor

            self.tex.clamp_(0, 1)

            self.tex.grad = None

        self._prepare_mesh_rasterization_cache()

        for B, batch_idxes in tqdm.tqdm(dataset_utils.BatchIdxIterator(
            self.shape,
            batch_size=self.config.batch_size,
            shuffle=False,
        )):
            batch_idxes: tuple[torch.Tensor, ...]

            cur_control_img = {
                model_name: utils.batch_indexing(
                    control_img, self.shape, 3, batch_idxes
                ).to(self.config.device)
                for model_name, control_img in self.control_img.items()
            }

            print(
                f"{cur_control_img[controlnet_utils.ControlNetModel.NORMAL].min()=}")
            print(
                f"{cur_control_img[controlnet_utils.ControlNetModel.NORMAL].max()=}")

            cur_pix_to_face = self.cache_pix_to_face[batch_idxes].to(
                self.config.device)
            # [..., H, W]

            cur_tex_coord = self.cache_tex_coord[batch_idxes].to(
                self.config.device)
            # [..., H, W, 2]

            print(f"{self.tex.shape=}")
            print(f"{self.tex.dtype=}")
            print(f"{self.tex.min()=}")
            print(f"{self.tex.max()=}")

            img = rendering_utils.sample_texture(
                texture=einops.rearrange(self.tex, "c h w -> h w c"),
                tex_coord=cur_tex_coord,
                wrap_mode=rendering_utils.WrapMode.MIRROR,
                sampling_mode=rendering_utils.SamplingMode.LINEAR,
            ).to(torch.float16)
            # [..., tex_h, tex_w, C]

            print(f"{img.dtype=}")
            print(f"{img.grad_fn=}")

            img = torch.where(
                (0 <= cur_pix_to_face)[..., None].expand(
                    *cur_pix_to_face.shape, C),
                # [..., H, W, C]

                img,

                bg_color.expand(*img.shape),
            )
            # [..., H, W, C]

            print(f"second {img.dtype=}")
            print(f"second {img.shape=}")
            print(f"second {img.min()=}")
            print(f"second {img.max()=}")

            """
            vision_utils.write_image(
                self.config.proj_dir / f"img_{utils.timestamp_sec()}.png",

                vision_utils.denormalize_image(
                    einops.rearrange(img.view(H, W, C), "h w c -> c h w")))
            """

            print(f"{img.grad_fn=}")

            for i in tqdm.tqdm(range(self.config.noise_samples)):
                cur_timestep = torch.randint(
                    1, 200, (1,), dtype=torch.int32, device=utils.CUDA_DEVICE)

                for model_name in tqdm.tqdm(self.config.sds_loss_config.keys()):
                    cur_sds_loss_config = \
                        self.config.sds_loss_config[model_name]

                    cur_sds_result: controlnet_utils.SDSLossResult = \
                        controlnet_utils.calc_sds_loss(
                            pipe=self.pipe[model_name],

                            prompt=self.text_prompt,
                            negative_prompt=self.negative_text_prompt,

                            img=einops.rearrange(
                                img, "... h w c -> ... c h w"),
                            control_img=cur_control_img[model_name],

                            timestep=cur_timestep,

                            guidance_scale=cur_sds_loss_config.guidance_scale,

                            controlnet_conditioning_scale=cur_sds_loss_config.controlnet_conditioning_scale,
                        )

                    img_latent = cur_sds_result.img_latent

                    cur_sds_loss = 0.5 * torch.nn.functional.mse_loss(
                        img_latent,
                        (img_latent - cur_sds_result.grad).detach(),
                        reduction="sum",
                    )

                    cur_sds_loss.backward(retain_graph=True)

                    with torch.no_grad():
                        self.tex.grad.nan_to_num_()

            _step()

        self.epoch += 1
        self.timestep -= 1

        self.output_tex()

        return training_utils.TrainingResult(message="")

    @utils.mem_clear
    def show_params(self):
        for name, param in self.module.named_parameters():
            print(f"{name}: {param}")

        print(self.optimizer.param_groups)

    @utils.mem_clear
    def output_tex(self):
        img_path = self.config.proj_dir / f"tex_{utils.timestamp_sec()}.png"

        vision_utils.write_image(
            img_path, vision_utils.denormalize_image(self.tex))

    @utils.mem_clear
    def output_rgb_video(self) -> None:
        H, W = self.camera_config.img_h, self.camera_config.img_w

        C, tex_h, tex_w = self.tex.shape

        bg_color = torch.tensor(
            [255], dtype=torch.uint8, device=self.config.device)

        video_writer = vision_utils.VideoWriter(
            path=self.config.proj_dir / f"rgb_{utils.timestamp_sec()}.avi",
            height=H,
            width=W,
            color_type=vision_utils.ColorType.RGB,
            fps=25.0,
        )

        self._prepare_mesh_rasterization_cache()

        for B, batch_idx in tqdm.tqdm(dataset_utils.BatchIdxIterator(
            self.shape,
            batch_size=self.config.batch_size,
            shuffle=False,
        )):
            batch_idx: tuple[torch.Tensor, ...]

            # mesh_ras_result.pix_to_face[..., H, W, 1]
            # mesh_ras_result.bary_coord[..., H, W, 1, 3]

            pix_to_face = self.cache_pix_to_face[batch_idx].to(
                self.config.device)
            # [..., H, W]

            tex_coord = self.cache_tex_coord[batch_idx].to(
                self.config.device)
            # [..., H, W, 2]

            img = rendering_utils.sample_texture(
                texture=einops.rearrange(self.tex, "c h w -> h w c"),
                tex_coord=tex_coord,
                wrap_mode=rendering_utils.WrapMode.MIRROR,
                sampling_mode=rendering_utils.SamplingMode.LINEAR,
            )
            # [..., tex_h, tex_w, C]

            print(f"{img.grad_fn=}")

            img = torch.where(
                (0 <= pix_to_face)[..., None].expand(*pix_to_face.shape, C),
                # [..., H, W, C]

                img,

                bg_color.expand(*img.shape),
            )
            # [..., H, W, C]

            video_writer.write(vision_utils.denormalize_image(einops.rearrange(
                img.view(H, W, C), "h w c -> c h w")))

        video_writer.close()

    @utils.mem_clear
    def resize_tex(self, tex_h: int, tex_w: int):
        assert 0 < tex_h
        assert 0 < tex_w

        self.tex = torchvision.transforms.Resize(
            (tex_h, tex_w),
            torchvision.transforms.InterpolationMode.BILINEAR,
        )(self.tex.detach().requires_grad_(False)).requires_grad_()

    @utils.mem_clear
    def save_callback(self):
        self.output_tex()
