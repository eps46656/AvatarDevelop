import collections
import dataclasses
import typing

import diffusers
import einops
import torch
import torchvision
import tqdm
from beartype import beartype

from .. import (avatar_utils, controlnet_utils, dataset_utils, rendering_utils,
                training_utils, utils, vision_utils)
from .Dataset import Dataset, Sample


@dataclasses.dataclass
class TrainerCoreConfig:
    proj_dir: utils.PathLike

    device: torch.device

    batch_size: int

    guidance_scale: float
    controlnet_conditioning_scale: float
    num_inference_steps: int

    init_ref_imgs_cnt: int
    ref_img_gamma: float

    lr: typing.Callable[[int], float]
    betas: tuple[float, float]


@beartype
class TrainerCore(training_utils.TrainerCore):
    def __init__(
        self,
        *,

        config: TrainerCoreConfig,

        avatar_blender: avatar_utils.AvatarBlender,

        dataset: Dataset,

        text_prompt: typing.Optional[str],
        negative_text_prompt: typing.Optional[str],

        init_tex: torch.Tensor,  # [C, tex_h, tex_w]
    ):
        self.config = dataclasses.replace(config)
        self.config.proj_dir = utils.to_pathlib_path(self.config.proj_dir)

        self.epoch = 0

        self.avatar_blender = avatar_blender

        self.dataset = dataset

        self.text_prompt = text_prompt
        self.negative_text_prompt = negative_text_prompt

        self.tex = init_tex.detach().clone().requires_grad_()

        self.pipe = controlnet_utils.load_controlnet_pipe(
            controlnet_model=controlnet_utils.ControlNetModel.INPAINT,
            dtype=torch.float16,
            device=self.config.device,
        )

        self.canonical_avatar_model: avatar_utils.AvatarModel = \
            self.avatar_blender.get_avatar_model()

        self.ref_img: list[typing.Optional[torch.Tensor]] = \
            [None] * self.dataset.shape.numel()

        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()

        self.cache_pix_to_face = None
        self.cache_bary_coord = None
        self.cache_tex_coord = None

    # --

    def get_epoch(self) -> int:
        return self.epoch

    # --

    def state_dict(self, full: bool) \
            -> collections.OrderedDict[str, typing.Any]:
        return collections.OrderedDict([
            ("epoch", self.epoch),
            ("tex", utils.serialize_tensor(self.tex)),
            ("ref_img", list(map(utils.serialize_tensor, self.ref_img))),
        ])

    def load_state_dict(self, state_dict: typing.Mapping[str, typing.Any]) \
            -> None:
        self.epoch = state_dict["epoch"]

        self.tex = utils.deserialize_tensor(state_dict["tex"]).detach().to(
            self.tex, copy=True).requires_grad_()

        self.ref_img = map(utils.deserialize_tensor, state_dict["ref_img"])

        self.optimizer = self._make_optimizer()
        self.optimizer.load_state_dict(state_dict["optimizer"])

        self.scheduler = self._make_scheduler()
        self.scheduler.load_state_dict(state_dict["scheduler"])

    # ---

    def _make_optimizer(self) -> torch.optim.Optimizer:
        base_lr = self.config.lr(0)

        return torch.optim.Adam(
            [self.tex],
            lr=base_lr,
            betas=self.config.betas,
        )

    def _make_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: self.config.lr(epoch) / self.config.lr(0),
            last_epoch=self.epoch - 1,
        )

    # ---

    @utils.mem_clear
    @torch.no_grad()
    def _prepare_mesh_rasterization_cache(self):
        if self.cache_pix_to_face is not None and \
                self.cache_bary_coord is not None and \
                self.cache_tex_coord is not None:
            return

        camera_config = self.dataset.sample.camera_config

        shape = self.dataset.shape

        H, W = camera_config.img_h, camera_config.img_w

        self.cache_pix_to_face = utils.disk_empty(
            (*shape, H, W), torch.int64)

        self.cache_bary_coord = utils.disk_empty(
            (*shape, H, W, 3), torch.float64)

        self.cache_tex_coord = utils.disk_empty(
            (*shape, H, W, 2), torch.float64)

        for B, batch_idx, sample in tqdm.tqdm(dataset_utils.load(
            self.dataset,
            batch_size=8,
            shuffle=False,
        )):
            sample: Sample

            cur_camera_transform = sample.camera_transform

            cur_blending_param = sample.blending_param

            cur_avatar_model: avatar_utils.AvatarModel = self.avatar_blender(
                cur_blending_param)

            mesh_ras_result = rendering_utils.rasterize_mesh(
                vert_pos=cur_avatar_model.vert_pos.to(self.config.device),

                faces=cur_avatar_model.mesh_graph.f_to_vvv.to(
                    self.config.device),

                camera_config=camera_config,
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

            self.cache_pix_to_face[batch_idx] = pix_to_face.detach().to(
                self.cache_pix_to_face)
            # [..., H, W]

            self.cache_bary_coord[batch_idx] = bary_coord.detach().to(
                self.cache_bary_coord)
            # [..., H, W, 3]

            self.cache_tex_coord[batch_idx] = tex_coord.detach().to(
                self.cache_tex_coord)
            # [..., H, W, 2]

    # ---

    @utils.mem_clear
    def train(self) -> training_utils.TrainingResult:
        camera_config = self.dataset.sample.camera_config

        H, W = camera_config.img_h, camera_config.img_w

        C = self.tex.shape[0]

        self._prepare_mesh_rasterization_cache()

        for B, batch_idx, sample in tqdm.tqdm(dataset_utils.load(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )):
            self.optimizer.zero_grad()

            sample: Sample

            flatten_batch_idx = utils.ravel_idxes(
                batch_idx, self.dataset.shape)
            # [B]

            cur_img = sample.img.to(
                self.config.device).reshape(B, C, H, W)
            # [B, C, H, W]

            print(f"{cur_img.min()=}")
            print(f"{cur_img.max()=}")

            cur_person_mask = sample.person_mask.to(
                self.config.device, torch.float32).reshape(B, 1, H, W)

            print(f"{cur_person_mask.min()=}")
            print(f"{cur_person_mask.max()=}")

            cur_pix_to_face = self.cache_pix_to_face[batch_idx].to(
                self.config.device).reshape(B, H, W)
            # [B, H, W]

            cur_tex_coord = self.cache_tex_coord[batch_idx].to(
                self.config.device).reshape(B, H, W, 2)
            # [B, H, W, 2]

            rendered_img = rendering_utils.sample_texture(
                texture=einops.rearrange(self.tex, "c h w -> h w c"),
                tex_coord=cur_tex_coord,
                wrap_mode=rendering_utils.WrapMode.MIRROR,
                sampling_mode=rendering_utils.SamplingMode.LINEAR,
            ).to(torch.float64)
            # [B, H, W, C]

            rendered_img = torch.where(
                (0 <= cur_pix_to_face)[..., None],
                # [..., H, W, C]

                rendered_img,

                1,
            )
            # [B, H, W, C]

            print(f"{rendered_img.dtype=}")
            print(f"{rendered_img.shape=}")
            print(f"{rendered_img.min()=}")
            print(f"{rendered_img.max()=}")

            cur_control_image = torch.where(
                cur_person_mask < 0.5, cur_img, -1)
            # [B, C, H, W]

            loss = 0.0

            for b in range(B):
                utils.mem_clear()

                vision_utils.show_image(
                    "rendered_img",

                    utils.rct(
                        einops.rearrange(
                            rendered_img[b], "h w c -> c h w") * 255,
                        dtype=torch.uint8,
                        device=utils.CPU_DEVICE,
                    ),
                )

                with torch.no_grad():
                    cur_ref_img = self.ref_img[flatten_batch_idx[b]]

                    for _ in range(
                        self.config.init_ref_imgs_cnt if cur_ref_img is None
                        else 1
                    ):
                        pipe_result = self.pipe(
                            prompt=self.text_prompt,
                            negative_prompt=self.negative_text_prompt,

                            image=cur_img[b]
                            .expand(1, C, H, W),
                            # [1, C, H, W]

                            mask_image=cur_person_mask[b]
                            .expand(1, 1, H, W),
                            # [1, H, W]

                            control_image=cur_control_image[b]
                            .expand(1, C, H, W),
                            # [1, C, H, W]

                            num_images_per_prompt=1,

                            num_inference_steps=self.config.num_inference_steps,
                            guidance_scale=self.config.guidance_scale,
                        )

                        new_ref_img = vision_utils.from_pillow_image(
                            pipe_result.images[0],
                            color_type="RGB",
                        ).image.to(utils.CPU_DEVICE, torch.float64) / 255

                        vision_utils.show_image(
                            f"new_ref_img",

                            utils.rct(
                                new_ref_img * 255,
                                dtype=torch.uint8,
                                device=utils.CPU_DEVICE
                            ),
                        )

                        cur_ref_img = new_ref_img if cur_ref_img is None else \
                            cur_ref_img * self.config.ref_img_gamma + \
                            new_ref_img * (1 - self.config.ref_img_gamma)

                        utils.mem_clear()

                    self.ref_img[flatten_batch_idx[b]] = cur_ref_img

                    vision_utils.show_image(
                        f"cur_ref_img",

                        utils.rct(
                            cur_ref_img * 255,
                            dtype=torch.uint8,
                            device=utils.CPU_DEVICE
                        ),
                    )

                print(f"{rendered_img[b].shape=}")
                print(f"{cur_person_mask[b].shape=}")

                img_diff = (
                    einops.rearrange(rendered_img[b], "h w c -> c h w") -
                    cur_ref_img.to(rendered_img)
                ) * cur_person_mask[b]
                # [C, H, W]

                print(f"{img_diff.square().mean()=}")
                print(f"{cur_person_mask[b].sum()=}")

                loss = loss + img_diff.square().sum() / \
                    cur_person_mask[b].sum()

            loss = loss / B

            loss.backward()

            print(f"{self.tex.grad.abs().min()=}")
            print(f"{self.tex.grad.abs().max()=}")
            print(f"{self.tex.grad.abs().mean()=}")

            self.optimizer.step()

            vision_utils.show_image(
                "tex",

                utils.rct(
                    self.tex * 255,
                    dtype=torch.uint8,
                    device=utils.CPU_DEVICE,
                ),
            )

        self.epoch += 1
        self.scheduler.step()

        self.output_tex()

        return training_utils.TrainingResult(message="")

    @utils.mem_clear
    def output_tex(self):
        img_path = self.config.proj_dir / f"tex_{utils.timestamp_sec()}.png"

        vision_utils.write_image(
            img_path, utils.rct(self.tex * 255, dtype=torch.uint8))

    @utils.mem_clear
    def output_rgb_video(self) -> None:
        camera_config = self.dataset.sample.camera_config

        H, W = camera_config.img_h, camera_config.img_w

        C = self.tex.shape[0]

        video_writer = vision_utils.VideoWriter(
            path=self.config.proj_dir / f"rgb_{utils.timestamp_sec()}.avi",
            height=H,
            width=W,
            color_type="RGB",
            fps=25.0,
        )

        self._prepare_mesh_rasterization_cache()

        for B, batch_idx in tqdm.tqdm(dataset_utils.BatchIdxIterator(
            self.dataset.shape,
            batch_size=self.config.batch_size,
            shuffle=False,
        )):
            cur_pix_to_face = self.cache_pix_to_face[batch_idx].to(
                self.config.device).reshape(B, H, W)
            # [B, H, W]

            cur_tex_coord = self.cache_tex_coord[batch_idx].to(
                self.config.device).reshape(B, H, W, 2)
            # [B, H, W, 2]

            rendered_img = rendering_utils.sample_texture(
                texture=einops.rearrange(self.tex, "c h w -> h w c"),
                tex_coord=cur_tex_coord,
                wrap_mode=rendering_utils.WrapMode.MIRROR,
                sampling_mode=rendering_utils.SamplingMode.LINEAR,
            ).to(torch.float64)
            # [B, H, W, C]

            rendered_img = torch.where(
                (0 <= cur_pix_to_face)[..., None].expand(
                    *cur_pix_to_face.shape, C),
                # [..., H, W, C]

                rendered_img,

                1,
            )
            # [B, H, W, C]

            for b in range(B):
                video_writer.write(utils.rct(einops.rearrange(
                    rendered_img[b], "h w c -> c h w") * 255,
                    dtype=torch.uint8,
                    device=utils.CPU_DEVICE,
                ))

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
