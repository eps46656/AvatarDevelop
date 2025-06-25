import collections
import dataclasses
import typing

import einops
import torch
import torchvision
import tqdm
from beartype import beartype

from .. import (avatar_utils, dataset_utils, rendering_utils, training_utils,
                utils, vision_utils)
from .Dataset import Dataset, Sample


@dataclasses.dataclass
class TrainerCoreConfig:
    proj_dir: utils.PathLike

    device: torch.device

    batch_size: int

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

        init_tex: torch.Tensor,  # [C, tex_h, tex_w]
    ):
        self.config = dataclasses.replace(config)
        self.config.proj_dir = utils.to_pathlib_path(self.config.proj_dir)

        self.epoch = 0

        self.avatar_blender = avatar_blender

        self.dataset = dataset

        self.tex = init_tex.detach().clone().requires_grad_()

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
            ("optimizer", self.optimizer.state_dict()),
            ("scheduler", self.scheduler.state_dict()),
        ])

    def load_state_dict(self, state_dict: typing.Mapping[str, typing.Any]) \
            -> None:
        self.epoch = state_dict["epoch"]

        self.tex = utils.deserialize_tensor(state_dict["tex"]).detach().to(
            self.tex, copy=True).requires_grad_()

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

            cur_camera_transform = sample.camera_transform \
                .to(self.config.device)

            cur_blending_param = sample.blending_param \
                .to(self.config.device)

            cur_avatar_model: avatar_utils.AvatarModel = \
                self.avatar_blender(cur_blending_param)

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
    def bake_texture(self, tex_h: int, tex_w: int):
        baked_tex = rendering_utils.bake_texture(
            camera_config=self.dataset.sample.camera_config,
            camera_transform=self.dataset.sample.camera_transform,

            img=self.dataset.sample.img,

            mask=self.dataset.sample.mask,
            blending_param=self.dataset.sample.blending_param,

            avatar_blender=self.avatar_blender,

            tex_h=self.tex.shape[1],
            tex_w=self.tex.shape[2],

            batch_size=self.config.batch_size,

            device=self.config.device,
        )

        self.tex = baked_tex.detach().clone().requires_grad_()

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

            cur_img = sample.img.to(self.config.device).view(B, C, H, W)
            # [B, C, H, W]

            cur_mask = sample.mask \
                .to(self.config.device, torch.float32).view(B, 1, H, W)

            cur_pix_to_face = self.cache_pix_to_face[batch_idx] \
                .to(self.config.device).view(B, H, W)
            # [B, H, W]

            cur_tex_coord = self.cache_tex_coord[batch_idx] \
                .to(self.config.device).view(B, H, W, 2)
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

            rendered_img = einops.rearrange(rendered_img, "b h w c -> b c h w")

            cur_masked_img = cur_img * cur_mask + (1 - cur_mask)

            vision_utils.show_image(
                "cur_masked_img",

                utils.rct(
                    cur_masked_img[0] * 255,
                    dtype=torch.uint8,
                    device=utils.CPU_DEVICE,
                ),
            )

            vision_utils.show_image(
                "rendered_img",

                utils.rct(
                    rendered_img[0] * 255,
                    dtype=torch.uint8,
                    device=utils.CPU_DEVICE,
                ),
            )

            img_diff = (
                rendered_img - cur_masked_img.to(rendered_img)  # [B, C, H, W]
            ) * cur_mask
            # [B, C, H, W]

            loss = img_diff.square().sum() / cur_mask.sum()

            loss.backward()

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

    @utils.mem_clear
    @torch.no_grad()
    def output_figure(self):
        frame_idx = 30

        camera_config = self.dataset.sample.camera_config

        H, W = camera_config.img_h, camera_config.img_w

        C = self.tex.shape[0]

        self._prepare_mesh_rasterization_cache()

        cur_pix_to_face = self.cache_pix_to_face[frame_idx].to(
            self.config.device).reshape(H, W)
        # [H, W]

        cur_tex_coord = self.cache_tex_coord[frame_idx].to(
            self.config.device).reshape(H, W, 2)
        # [H, W, 2]

        rendered_img = rendering_utils.sample_texture(
            texture=einops.rearrange(self.tex, "c h w -> h w c"),
            tex_coord=cur_tex_coord,
            wrap_mode=rendering_utils.WrapMode.MIRROR,
            sampling_mode=rendering_utils.SamplingMode.LINEAR,
        ).to(torch.float64)
        # [H, W, C]

        rendered_img = torch.where(
            (0 <= cur_pix_to_face)[..., None].expand(
                *cur_pix_to_face.shape, C),
            # [H, W, C]

            rendered_img,

            1,
        )
        # [H, W, C]

        vision_utils.write_image(
            self.config.proj_dir /
            f"tex_{utils.timestamp_sec()}.png",
            utils.rct(einops.rearrange(
                rendered_img, "h w c -> c h w") * 255, dtype=torch.uint8),
        )
