import dataclasses
import os
import time
import typing

import einops
import torch
import tqdm
from beartype import beartype

from . import (avatar_utils, dataset_utils, dw_interp_utils, gaussian_utils,
               gom_utils, pca_utils, rendering_utils, smplx_utils,
               texture_utils, training_utils, utils)


@dataclasses.dataclass
class Config:
    proj_dir: os.PathLike

    device: torch.device

    batch_size: int

    alpha_rgb: float
    alpha_lap_smoothing: float
    alpha_nor_sim: float
    alpha_color_diff: float


@beartype
class TrainingCore(training_utils.TrainingCore):
    def __init__(
        self,
        config: Config,
        module: gom_utils.Module,
        dataset: gom_utils.Dataset,
        optimizer: typing.Optional[torch.optim.Optimizer],
        scheduler: object,
    ):
        self.__config = dataclasses.replace(config)
        self.__config.proj_dir = utils.to_pathlib_path(self.__config.proj_dir)

        self.__module = module
        self.__dataset = dataset
        self.__optimizer = optimizer
        self.__scheduler = scheduler

    # --

    @property
    def module(self) -> gom_utils.Module:
        return self.__module

    @module.setter
    def module(self, module: gom_utils.Module) -> None:
        self.__module = module

    # --

    @property
    def dataset(self) -> typing.Optional[gom_utils.Dataset]:
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset: gom_utils.Dataset) -> None:
        self.__dataset = dataset

    # --

    @property
    def optimizer(self) -> typing.Optional[torch.optim.Optimizer]:
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.__optimizer = optimizer

    # --

    @property
    def scheduler(self) -> object:
        return self.__scheduler

    @scheduler.setter
    def scheduler(self, scheduler: object) -> None:
        self.__scheduler = scheduler

    # --

    def calc_loss(self, forward_result: gom_utils.ModuleForwardResult) \
            -> torch.Tensor:
        weighted_rgb_loss = \
            self.__config.alpha_rgb * forward_result.rgb_loss

        weighted_lap_smoothing_loss = \
            self.__config.alpha_lap_smoothing * forward_result.lap_smoothing_loss

        weighted_nor_sim_loss = \
            self.__config.alpha_nor_sim * forward_result.nor_sim_loss

        weighted_color_diff_loss = \
            self.__config.alpha_color_diff * forward_result.color_diff_loss

        return weighted_rgb_loss + weighted_lap_smoothing_loss + weighted_nor_sim_loss + weighted_color_diff_loss

    # --

    @utils.mem_clear
    def train(self) -> training_utils.TrainingResult:
        assert self.scheduler is None or isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        sum_loss = 0.0

        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(
                self.dataset, batch_size=self.__config.batch_size)):
            batch_size = batch_idxes[0].shape[0]

            sample: gom_utils.Sample

            result: gom_utils.ModuleForwardResult = self.module(
                camera_transform=sample.camera_transform,
                camera_config=sample.camera_config,
                img=sample.img,
                mask=sample.mask,
                blending_param=sample.blending_param,
            )

            loss: torch.Tensor = self.calc_loss(result)

            sum_loss += float(loss) * batch_size

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        avg_loss = sum_loss / self.dataset.shape.numel()

        print(f"{avg_loss=}")

        if self.scheduler is not None:
            self.scheduler.step(avg_loss)

        return training_utils.TrainingResult(
            avg_loss=avg_loss
        )

    @utils.mem_clear
    def show_params(self):
        for name, param in self.module.named_parameters():
            print(f"{name}: {param}")

        print(self.optimizer.param_groups)

    @utils.mem_clear
    @torch.no_grad()
    def output_rgb_video(self):
        # call output_rgb_video
        self.dataset: gom_utils.Dataset

        rgb_frames = torch.empty_like(
            self.dataset.sample.img,
            dtype=torch.float16,
            device=self.__config.device,
        )
        # [T, C, H, W]

        T, C, H, W = self.dataset.sample.img.shape

        avatar_model: smplx_utils.Model = \
            self.module.avatar_blender.get_avatar_model()

        avatar_model.mesh_data.show(avatar_model.vert_pos)

        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(
                self.dataset, batch_size=self.__config.batch_size)):
            batch_size = batch_idxes[0].shape[0]

            idxes = utils.ravel_idxes(batch_idxes, self.dataset.shape)
            # [K]

            sample: gom_utils.Sample

            result: gom_utils.ModuleForwardResult = self.module(
                camera_config=sample.camera_config,
                camera_transform=sample.camera_transform,
                img=sample.img,
                mask=sample.mask,
                blending_param=sample.blending_param,
            )

            rendered_img = result.rendered_img.reshape(-1, C, H, W)
            # [K, C, H, W]

            utils.ein_scatter(
                dst=rgb_frames,  # [T, C, H, W]
                dst_expr=(None, *"chw"),

                idx=idxes.to(rgb_frames.device),  # [K]
                idx_expr="k",

                src=rendered_img.to(rgb_frames),  # [K, C, H, W]
                src_expr="kchw",

                inplace=True,
                mode=utils.ScatterMode.SET,
            )
            # rgb_frames[idesx[k], c, h, w] = rendered_img[k, c, h, w]

            """

            ... syms None
            ... syms 1
            ... syms 1

            dst[idx[k], c, h, w] = src[k, c, h, w]

            dst[c, h, w, ?]
            idx[c, h, w, k]
            src[c, h, w, k]


            dst[c, h, w, 1, 1, ?]
            idx[c, 1, w, a, b, 1]
            src[1, h, w, a, b, 1]




            dst[idx[a, b], c] = src[a, b]

            """

            avatar_model: smplx_utils.Model = result.avatar_model

        utils.write_video(
            path=self.__config.proj_dir / f"rgb_{int(time.time())}.mp4",
            video=rgb_frames,
            fps=25,
        )

    @utils.mem_clear
    @torch.no_grad()
    def bake_texture_face(self, tex_h: int, tex_w: int):
        # call bake_texture_face tex_h=1000 tex_w=1000

        self.dataset: gom_utils.Dataset

        T, C, H, W = self.dataset.sample.img.shape

        self.module: gom_utils.Module

        avatar_model: avatar_utils.AvatarModel = \
            self.module.avatar_blender.get_avatar_model()

        F = avatar_model.faces_cnt

        face_color_sum = torch.zeros(
            (F + 1, 3), dtype=torch.float64, device=self.__config.batch_size)

        face_weight_sum = torch.zeros(
            (F + 1,), dtype=torch.int, device=self.__config.batch_size)

        face_ones = torch.ones(
            (tex_h * tex_w,), dtype=torch.int, device=self.__config.batch_size)

        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(
                self.dataset, batch_size=self.__config.batch_size)):
            utils.mem_clear()

            batch_size = batch_idxes[0].shape[0]

            sample: gom_utils.Sample

            result: gom_utils.ModuleForwardResult = self.module(
                camera_config=sample.camera_config,
                camera_transform=sample.camera_transform,
                img=sample.img,
                mask=sample.mask,
                blending_param=sample.blending_param,
            )

            rendered_img = result.rendered_img.reshape(-1, C, H, W)
            # [K, C, H, W]

            for k in range(batch_size):
                cur_avatar_model = result.avatar_model[k]

                rendered_img = result.rendered_img[k]
                # [C, H, W]

                mesh_ras_result = rendering_utils.rasterize_mesh(
                    vert_pos=cur_avatar_model.vert_pos,
                    faces=cur_avatar_model.mesh_data.f_to_vvv,
                    camera_config=sample.camera_config,
                    camera_transform=sample.camera_transform[k],
                    faces_per_pixel=1,
                )

                pixel_to_faces = mesh_ras_result.pixel_to_faces
                # [img_h, img_w, 1]

                rendered_img = einops.rearrange(
                    rendered_img, "c h w -> h w c").reshape(-1, 3)
                # [H * W, 3]

                pixel_to_faces = pixel_to_faces.reshape(-1)
                # [H * W]

                idx = (pixel_to_faces != -1).nonzero().reshape(-1)
                # [z]

                if idx.numel() == 0:
                    continue

                rendered_img = rendered_img[idx, :]
                # [z, 3]

                pixel_to_faces = pixel_to_faces[idx]
                # [z]

                face_color_sum.index_add_(
                    0, pixel_to_faces, rendered_img.to(face_color_sum.dtype))
                face_weight_sum.index_add_(
                    0, pixel_to_faces, face_ones[:pixel_to_faces.shape[0]])

                """

                face_color_sum[pixel_to_faces[i], :] += rendered_img[i, :]
                face_weight_sum[pixel_to_faces[i]] += face_ones[i]

                """

        face_idx_map = texture_utils.calc_face_idx(
            tex_vert_pos=avatar_model.tex_vert_pos,
            tex_faces=avatar_model.tex_mesh_data.f_to_vvv,

            tex_h=tex_h,
            tex_w=tex_w,
        )
        # [H, W, 1]

        print(f"{face_idx_map.shape=}")
        print(f"{face_idx_map.dtype=}")
        print(f"{face_idx_map.min()=}")
        print(f"{face_idx_map.max()=}")

        face_color = face_color_sum / (1e-2 + face_weight_sum).unsqueeze(-1)
        # [F + 1, 3]

        face_color[F, :] = 1

        face_idx_map = face_idx_map.reshape(-1, 1).expand(tex_h * tex_w, 3)

        print(f"{face_color_sum=}")
        print(f"{face_weight_sum=}")

        # [H * W]

        tex = torch.gather(face_color, 0, face_idx_map)
        # [H * W, 3]

        """
        tex[i, j] = face_color[face_idx_map[i, j], j]
        """

        tex = tex.reshape(tex_h, tex_w, 3)

        utils.write_image(
            path=self.__config.proj_dir / f"tex_{int(time.time())}.png",
            img=einops.rearrange(tex, "h w c -> c h w"),
        )

    @utils.mem_clear
    @torch.no_grad()
    def bake_texture_face_2(self, tex_h: int, tex_w: int):
        """
        load_latest
        call bake_texture_face_2 tex_h=1000 tex_w=1000
        """

        self.dataset: gom_utils.Dataset

        T, C, H, W = self.dataset.sample.img.shape

        self.module: gom_utils.Module

        avatar_model: avatar_utils.AvatarModel = \
            self.module.avatar_blender.get_avatar_model()

        F = avatar_model.faces_cnt

        face_color_all_cnt = torch.zeros(
            (F + 1,), dtype=torch.int, device=self.__config.device)

        face_color_all_sum_x = torch.zeros(
            (F + 1, C), dtype=torch.float64, device=self.__config.device)

        face_color_all_sum_xxt = torch.zeros(
            (F + 1, C, C), dtype=torch.float64, device=self.__config.device)

        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(self.dataset, batch_size=8)):
            utils.mem_clear()

            B = batch_idxes[0].shape[0]

            sample: gom_utils.Sample

            cur_avatar_model: smplx_utils.Model = \
                self.module.avatar_blender(sample.blending_param)

            fragments = rendering_utils.rasterize_mesh(
                vert_pos=cur_avatar_model.vert_pos,
                faces=cur_avatar_model.mesh_data.f_to_vvv,
                camera_config=sample.camera_config,
                camera_transform=sample.camera_transform,
                faces_per_pixel=1,
            )

            pix_to_face = fragments.pix_to_face.reshape(-1)
            # [B * H * W]

            pix_to_face = (pix_to_face + (F + 1)) % (F + 1)
            # [B * H * W]

            ref_img = einops.rearrange(
                sample.img, "b c h w -> b h w c").reshape(B * H * W, C)
            # [B * H * W, C]

            pca_utils.scatter_feed(
                idx=pix_to_face,  # [B * H * W]
                x=ref_img,  # [B * H * W, C]
                inplace=True,

                dst_cnts=face_color_all_cnt,  # [F]
                dst_sum_x=face_color_all_sum_x,  # [F, C]
                dst_sum_xxt=face_color_all_sum_xxt,  # [F, C, C]
            )

        face_color_means, face_color_pcas, face_color_stds = pca_utils.get_pca(
            cnt=face_color_all_cnt,
            sum_x=face_color_all_sum_x,
            sum_xxt=face_color_all_sum_xxt,
        )

        # face_color_means[F + 1, C]
        # face_color_pcas[F + 1, C, C]
        # face_color_stds[F + 1, C]

        face_ell_means = face_color_means

        inv_face_ell_axis = torch.where(
            (2 <= face_color_all_cnt)[..., None, None].expand(F + 1, C, C),
            face_color_pcas * face_color_stds.unsqueeze(-1),
            10 * torch.eye(
                C, dtype=face_color_pcas.dtype, device=face_color_pcas.device)
            .expand(F + 1, C, C)
        ).inverse()
        # [F, C, C]

        face_color_inlier_cnt = torch.zeros(
            (F + 1,), dtype=torch.int, device=self.__config.device)

        face_color_inlier_sum_x = torch.zeros(
            (F + 1, C), dtype=torch.float64, device=self.__config.device)

        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(self.dataset, batch_size=8)):
            utils.mem_clear()

            B = batch_idxes[0].shape[0]

            sample: gom_utils.Sample

            cur_avatar_model: smplx_utils.Model = \
                self.module.avatar_blender(sample.blending_param)

            fragments = rendering_utils.rasterize_mesh(
                vert_pos=cur_avatar_model.vert_pos,
                faces=cur_avatar_model.mesh_data.f_to_vvv,
                camera_config=sample.camera_config,
                camera_transform=sample.camera_transform,
                faces_per_pixel=1,
            )

            pix_to_face = fragments.pix_to_face.reshape(-1)
            # [B * H * W]

            pix_to_face = (pix_to_face + (F + 1)) % (F + 1)
            # [B * H * W]

            ref_img = einops.rearrange(
                sample.img, "b c h w -> b h w c").reshape(B * H * W, C)
            # [B * H * W, C]

            cur_face_ell_means = face_ell_means[pix_to_face]
            # [B * H * W, C]

            cur_inv_face_ell_axis = inv_face_ell_axis[pix_to_face]
            # [B * H * W, C, C]

            ell_coord = (
                cur_inv_face_ell_axis @
                (ref_img - cur_face_ell_means).unsqueeze(-1)).squeeze(-1)
            # [B * H * W, C]

            ell_dist = utils.vec_norm(ell_coord)
            # [B * H * W]

            pix_to_face = torch.where(
                ell_dist <= 1.5,
                pix_to_face,
                F,
            )
            # [B * H * W]

            face_color_inlier_cnt += pix_to_face.bincount(minlength=F + 1)

            face_color_inlier_sum_x.index_add_(
                0, pix_to_face,
                ref_img.to(face_color_inlier_sum_x.device,
                           face_color_inlier_sum_x.dtype))

        face_idx_map = texture_utils.calc_face_idx(
            tex_vert_pos=avatar_model.tex_vert_pos,
            tex_faces=avatar_model.tex_mesh_data.f_to_vvv,

            tex_h=tex_h,
            tex_w=tex_w,
        )
        # [H, W]

        face_idx_map = (face_idx_map + (F + 1)) % (F + 1)

        print(f"{face_idx_map.shape=}")
        print(f"{face_idx_map.dtype=}")
        print(f"{face_idx_map.min()=}")
        print(f"{face_idx_map.max()=}")

        face_color = face_color_inlier_sum_x / \
            (1e-2 + face_color_inlier_cnt).unsqueeze(-1)
        # [F + 1, C]

        face_color[F, :] = 1

        face_idx_map = face_idx_map.reshape(-1, 1).expand(tex_h * tex_w, 3)

        print(f"{face_idx_map.min()=}")
        print(f"{face_idx_map.max()=}")

        tex = torch.gather(face_color, 0, face_idx_map)
        # [H * W, 3]

        """
        tex[i, j] = face_color[face_idx_map[i, j], j]
        """

        tex = tex.reshape(tex_h, tex_w, 3)

        utils.write_image(
            path=self.__config.proj_dir / f"tex_{int(time.time())}.png",
            img=einops.rearrange(tex, "h w c -> c h w"),
        )
