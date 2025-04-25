import dataclasses
import math
import os
import shutil
import time
import typing

import einops
import numpy as np
import torch
import torchrbf
import tqdm
from beartype import beartype

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures

from . import (avatar_utils, dataset_utils, gaussian_utils, gom_utils,
               kernel_splatting_utils, mesh_utils, pca_utils, rendering_utils,
               smplx_utils, texture_utils, training_utils, utils, vision_utils)


@dataclasses.dataclass
class Config:
    proj_dir: os.PathLike

    device: torch.device

    batch_size: int

    lr: float

    vert_grad_norm_threshold: float

    alpha_rgb: float
    alpha_lap_smoothness: float
    alpha_nor_sim: float
    alpha_edge_var: float
    alpha_color_diff: float
    alpha_gp_scale_diff: float


@beartype
class TrainingCore(training_utils.TrainingCore):
    def __init__(
        self,
        config: Config,
        module: gom_utils.Module,
        dataset: gom_utils.Dataset,

        optimizer_factory:
        typing.Callable[[dict], torch.optim.Optimizer],

        scheduler_factory:
        typing.Optional[typing.Callable[[torch.optim.Optimizer], object]],
    ):
        self.__config = dataclasses.replace(config)
        self.__config.proj_dir = utils.to_pathlib_path(self.__config.proj_dir)

        self.__module = module
        self.__dataset = dataset

        self.__optimizer_factory = optimizer_factory
        self.__scheduler_factory = scheduler_factory

        self.__optimizer = None
        self.__scheduler = None

        self.set_optimizer()
        self.set_scheduler()

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
        print(f"{forward_result.rgb_loss=}")
        print(f"{forward_result.lap_smoothness_loss=}")
        print(f"{forward_result.nor_sim_loss=}")
        print(f"{forward_result.edge_var_loss=}")
        print(f"{forward_result.color_diff_loss=}")
        print(f"{forward_result.gp_scale_diff_loss=}")

        weighted_rgb_loss = self.__config.alpha_rgb * \
            forward_result.rgb_loss.mean()

        weighted_lap_smoothness_loss = self.__config.alpha_lap_smoothness * \
            forward_result.lap_smoothness_loss.mean()

        weighted_nor_sim_loss = self.__config.alpha_nor_sim * \
            forward_result.nor_sim_loss.mean()

        weighted_edge_var_loss = self.__config.alpha_edge_var * \
            forward_result.edge_var_loss.mean()

        weighted_color_diff_loss = self.__config.alpha_color_diff * \
            forward_result.color_diff_loss.mean()

        weighted_gp_scale_diff_loss = self.__config.alpha_gp_scale_diff * \
            forward_result.gp_scale_diff_loss.mean()

        return \
            weighted_rgb_loss + \
            weighted_lap_smoothness_loss + \
            weighted_nor_sim_loss + \
            weighted_edge_var_loss + \
            weighted_color_diff_loss + \
            weighted_gp_scale_diff_loss

    # --

    @utils.mem_clear
    def set_optimizer(self):
        self.__optimizer = self.__optimizer_factory(
            utils.get_param_groups(self.module, self.__config.lr))

    @utils.mem_clear
    def set_scheduler(self):
        if self.__scheduler_factory is None:
            return

        self.__scheduler = self.__scheduler_factory(self.optimizer)

    @utils.mem_clear
    def train(self) -> training_utils.TrainingResult:
        assert self.scheduler is None or isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        sum_loss = 0.0

        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(
                self.dataset, batch_size=self.__config.batch_size)):
            batch_size = batch_idxes[0].shape[0]

            sample: gom_utils.Sample

            self.module.refresh()

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

            avatar_blender = self.module.avatar_blender

            if isinstance(avatar_blender, smplx_utils.ModelBlender) and isinstance(avatar_blender.model_builder, smplx_utils.DeformableModelBuilder):
                model_builder: smplx_utils.DeformableModelBuilder = avatar_blender.model_builder

                vert_pos = model_builder.model_data.vert_pos

                vert_grad_norm_threshold = self.__config.vert_grad_norm_threshold

                if vert_pos.grad is not None and vert_grad_norm_threshold is not None:
                    grad_norm = utils.vec_norm(vert_pos.grad, -1, True)

                    vert_pos.grad *= vert_grad_norm_threshold / \
                        grad_norm.clamp(vert_grad_norm_threshold)

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

        rgb_frames = utils.empty_like(
            self.dataset.sample.img,
            dtype=torch.uint8,
            device=self.__config.device,
        )
        # [T, C, H, W]

        T, C, H, W = self.dataset.sample.img.shape

        avatar_model: smplx_utils.Model = \
            self.module.avatar_blender.get_avatar_model()

        avatar_model.mesh_graph.show(avatar_model.vert_pos)

        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(
            self.dataset,
            batch_size=self.__config.batch_size,
            shuffle=False,
        )):
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

            rendered_img = result.gp_render_img.reshape(-1, C, H, W)
            # [K, C, H, W]

            rgb_frames.index_put_(
                (idxes,),
                vision_utils.denormalize_image(
                    rendered_img,
                    dtype=rgb_frames.dtype,
                    device=rgb_frames.device
                )
            )

            avatar_model: smplx_utils.Model = result.avatar_model

        with vision_utils.VideoWriter(
            path=self.__config.proj_dir / f"rgb_{utils.timestamp_sec()}.avi",
            height=H,
            width=W,
            color_type=vision_utils.ColorType.RGB,
            fps=25.0,
        ) as video_writer:
            for i in range(T):
                video_writer.write(rgb_frames[i])

    @utils.mem_clear
    @torch.no_grad()
    def face_cos_nor_sim_subdivide(self):
        self.module.refresh()

        model_blender: smplx_utils.ModelBlender = \
            self.module.avatar_blender

        model_builder: smplx_utils.ModelBuilder = model_blender.model_builder

        model_builder.show()

        model_data = model_builder.get_model_data()

        mesh_data = mesh_utils.MeshData(
            model_data.mesh_graph,
            model_data.vert_pos,
        )

        target_faces: set[int] = set()

        norm_cos_sim_threshold = math.cos(45 * utils.DEG)

        for fpi, norm_cos_sim in enumerate(mesh_data.face_norm_cos_sim):
            if norm_cos_sim < norm_cos_sim_threshold:
                target_faces.update(map(int, mesh_data.mesh_graph.ff[fpi]))

        print(f"{len(target_faces)=}")

        self.module.subdivide(target_faces=target_faces)

        model_builder.show()

        self.set_optimizer()
        self.set_scheduler()

    @utils.mem_clear
    @torch.no_grad()
    def edge_len_subdivide(self):
        self.module.refresh()

        model_blender: smplx_utils.ModelBlender = \
            self.module.avatar_blender

        model_builder: smplx_utils.ModelBuilder = model_blender.model_builder

        model_builder.show()

        model_data = model_builder.get_model_data()

        mesh_data = mesh_utils.MeshData(
            model_data.mesh_graph,
            model_data.vert_pos,
        )

        edge_len_threshold = 5 * 1e-3

        target_edges = {
            ei
            for ei, edge_len in enumerate(mesh_data.edge_norm)
            if edge_len_threshold < edge_len
        }

        print(f"{len(target_edges)=}")

        self.module.subdivide(target_edges=target_edges)

        model_builder.show()

        self.set_optimizer()
        self.set_scheduler()

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

            rendered_img = result.gp_render_img.reshape(-1, C, H, W)
            # [K, C, H, W]

            for k in range(batch_size):
                cur_avatar_model = result.avatar_model[k]

                rendered_img = result.gp_render_img[k]
                # [C, H, W]

                mesh_ras_result = rendering_utils.rasterize_mesh(
                    vert_pos=cur_avatar_model.vert_pos,
                    faces=cur_avatar_model.mesh_graph.f_to_vvv,
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
            tex_faces=avatar_model.tex_mesh_graph.f_to_vvv,

            tex_h=tex_h,
            tex_w=tex_w,
        )
        # [H, W, 1]

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

        vision_utils.write_image(
            path=self.__config.proj_dir / f"tex_{utils.timestamp_sec()}.png",
            img=vision_utils.denormalize_image(
                einops.rearrange(tex, "h w c -> c h w")),
        )

    @utils.mem_clear
    @torch.no_grad()
    def bake_texture_face_2(
        self,
        tex_h: int,
        tex_w: int,
    ):
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

        mesh_ras_result_tmp_path = utils.allocate_tmp_dir() / "mesh_ras_result.pkl"

        mesh_ras_result_tmp = utils.PickleWriter(
            mesh_ras_result_tmp_path)

        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(self.dataset, batch_size=8, shuffle=False)):
            utils.mem_clear()

            B = batch_idxes[0].shape[0]

            sample: gom_utils.Sample

            cur_avatar_model: smplx_utils.Model = \
                self.module.avatar_blender(sample.blending_param)

            fragments = rendering_utils.rasterize_mesh(
                vert_pos=cur_avatar_model.vert_pos,
                faces=cur_avatar_model.mesh_graph.f_to_vvv,
                camera_config=sample.camera_config,
                camera_transform=sample.camera_transform,
                faces_per_pixel=1,
            )

            pix_to_face = fragments.pix_to_face.reshape(B, H, W)

            for b in range(B):
                mesh_ras_result_tmp.write(
                    utils.tensor_serialize(pix_to_face[b]))

            pix_to_face = torch.where(
                0.5 <= sample.mask.reshape(B, H, W),
                pix_to_face,
                -1,
            )
            # discard pixels not on person

            pix_to_face = (pix_to_face + (F + 1)) % (F + 1)

            pix_to_face = pix_to_face.reshape(-1)
            # [B * H * W]

            ref_img = einops.rearrange(
                sample.img, "b c h w -> b h w c").reshape(B * H * W, C)
            # [B * H * W, C]

            pca_utils.scatter_feed(
                idx=pix_to_face,  # [B * H * W]
                x=ref_img,  # [B * H * W, C]
                inplace=True,

                dst_sum_w=face_color_all_cnt,  # [F]
                dst_sum_w_x=face_color_all_sum_x,  # [F, C]
                dst_sum_w_xxt=face_color_all_sum_xxt,  # [F, C, C]
            )

        mesh_ras_result_tmp.close()

        mesh_ras_result_tmp = utils.PickleReader(mesh_ras_result_tmp_path)

        face_color_mean, face_color_pca, face_color_std = pca_utils.get_pca(
            cnt=face_color_all_cnt,
            sum_w_x=face_color_all_sum_x,
            sum_w_xxt=face_color_all_sum_xxt,
        )

        # face_color_mean[F + 1, C]
        # face_color_pca[F + 1, C, C]
        # face_color_std[F + 1, C]

        use_pca_threshold = 10

        face_color_ell_mean = face_color_mean

        inv_face_color_ell_axis = torch.where(
            (use_pca_threshold <= face_color_all_cnt)[..., None, None]
            .expand(F + 1, C, C),

            (face_color_pca * face_color_std.unsqueeze(-1)).transpose(-1, -2),

            torch.eye(
                C, dtype=face_color_pca.dtype, device=face_color_pca.device)
            .expand(F + 1, C, C)
        ).inverse()
        # [F, C, C]

        face_color_inlier_sum_weight = torch.zeros(
            (F + 1,), dtype=torch.float64, device=self.__config.device)

        face_color_inlier_sum_x = torch.zeros(
            (F + 1, C), dtype=torch.float64, device=self.__config.device)

        for t in tqdm.tqdm(range(T)):
            utils.mem_clear()

            sample: gom_utils.Sample = self.dataset[t]

            pix_to_face = utils.tensor_deserialize(
                mesh_ras_result_tmp.read(),
                device=self.__config.device,
            )
            # [H, W]

            pix_to_face = pix_to_face.reshape(H * W)

            pix_to_face = (pix_to_face + (F + 1)) % (F + 1)
            # [H * W]

            ref_img = einops.rearrange(
                sample.img, "c h w -> h w c").reshape(H * W, C)
            # [H * W, C]

            cur_face_color_ell_mean = face_color_ell_mean[pix_to_face]
            # [H * W, C]

            cur_inv_face_color_ell_axi = inv_face_color_ell_axis[pix_to_face]
            # [H * W, C, C]

            ell_coord = (
                cur_inv_face_color_ell_axi @
                (ref_img - cur_face_color_ell_mean).unsqueeze(-1)).squeeze(-1)
            # [H * W, C]

            weight = 1 / (0.1 + utils.vec_norm(ell_coord).square())
            # [H * W]

            face_color_inlier_sum_weight.index_add_(
                0, pix_to_face, weight)

            face_color_inlier_sum_x.index_add_(
                0,

                pix_to_face,

                ref_img.to(face_color_inlier_sum_x.device,
                           face_color_inlier_sum_x.dtype) *
                weight.unsqueeze(-1)
            )

        face_idx_map = texture_utils.calc_face_idx(
            tex_vert_pos=avatar_model.tex_vert_pos,
            tex_faces=avatar_model.tex_mesh_graph.f_to_vvv,

            tex_h=tex_h,
            tex_w=tex_w,
        )
        # [H, W]

        face_idx_map = (face_idx_map + (F + 1)) % (F + 1)

        face_color = torch.where(
            (use_pca_threshold <= face_color_all_cnt)
            .unsqueeze(-1).expand(F + 1, 3),

            face_color_inlier_sum_x /
            (1e-2 + face_color_inlier_sum_weight).unsqueeze(-1),

            face_color_all_sum_x /
            (1e-2 + face_color_all_cnt).unsqueeze(-1),
        )

        face_color[F, :] = 1

        face_idx_map = face_idx_map.reshape(-1, 1).expand(tex_h * tex_w, 3)

        tex = torch.gather(face_color, 0, face_idx_map)
        # [H * W, 3]

        tex = tex.reshape(tex_h, tex_w, 3)

        vision_utils.write_image(
            path=self.__config.proj_dir / f"tex_{utils.timestamp_sec()}.png",
            img=vision_utils.denormalize_image(
                einops.rearrange(tex, "h w c -> c h w")),
        )

    @utils.mem_clear
    @torch.no_grad()
    @utils.deferable
    def bake_texture_face_3(
        self,
        tex_h: int,
        tex_w: int,
    ):
        """
        load_latest
        call bake_texture_face_3 tex_h=1000 tex_w=1000
        """

        self.dataset: gom_utils.Dataset

        T, C, H, W = self.dataset.sample.img.shape

        self.module: gom_utils.Module

        avatar_model: avatar_utils.AvatarModel = \
            self.module.avatar_blender.get_avatar_model()

        V = avatar_model.verts_cnt
        TV = avatar_model.tex_verts_cnt
        F = avatar_model.faces_cnt

        tex_vert_color_pca_calculator = pca_utils.PCACalculator(
            n=TV + 1,
            dim=C,
            dtype=torch.float64,
            device=self.__config.device,
        )

        mesh_ras_result_tmp_path = utils.allocate_tmp_dir() / "mesh_ras_result.pkl"

        mesh_ras_result_tmp = utils.PickleWriter(
            mesh_ras_result_tmp_path)

        def exit():
            mesh_ras_result_tmp.close()
            mesh_ras_result_tmp_path.unlink(missing_ok=True)

        utils.defer(exit)

        tex_f_to_vvv = torch.cat([
            avatar_model.tex_mesh_graph.f_to_vvv,  # [F, 3]

            torch.tensor(
                [[TV, TV, TV]],
                dtype=avatar_model.tex_mesh_graph.f_to_vvv.dtype,
                device=avatar_model.tex_mesh_graph.f_to_vvv.device
            ),  # [1, 3]
        ], dim=0)
        # [F + 1, 3]

        tex_vert_pos = torch.cat([
            avatar_model.tex_vert_pos,  # [TV, 2]

            torch.tensor(
                [[0, 0]],
                dtype=avatar_model.tex_vert_pos.dtype,
                device=avatar_model.tex_vert_pos.device
            ),  # [1, 2]
        ], dim=0)
        # [TV + 1, 2]

        for batch_idxes, sample in tqdm.tqdm(dataset_utils.load(
                self.dataset, batch_size=4, shuffle=False)):
            utils.mem_clear()

            B = batch_idxes[0].shape[0]

            sample: gom_utils.Sample

            cur_avatar_model: smplx_utils.Model = \
                self.module.avatar_blender(sample.blending_param)

            frag = rendering_utils.rasterize_mesh(
                vert_pos=cur_avatar_model.vert_pos,
                faces=cur_avatar_model.mesh_graph.f_to_vvv,
                camera_config=sample.camera_config,
                camera_transform=sample.camera_transform,
                faces_per_pixel=1,
            )

            pix_to_face: torch.Tensor = frag.pix_to_face.reshape(B, H, W)
            bary_coord: torch.Tensor = frag.bary_coords.reshape(B, H, W, 3)

            pix_to_face = torch.where(
                0.5 <= sample.mask.reshape(B, H, W),
                pix_to_face,
                -1,
            )
            # discard pixels not on person

            pix_to_face = (pix_to_face + (F + 1)) % (F + 1)

            for b in range(B):
                mesh_ras_result_tmp.write(
                    utils.tensor_serialize(pix_to_face[b], np.int32))

                mesh_ras_result_tmp.write(
                    utils.tensor_serialize(bary_coord[b]))

            ref_img = einops.rearrange(
                sample.img, "b c h w -> b h w c")
            # [B, H, W, C]

            tex_vert_color_pca_calculator.scatter_feed(
                idx=tex_f_to_vvv[pix_to_face],  # [B, H, W, 3]
                w=bary_coord,  # [B, H, W, 3]
                x=ref_img[:, :, :, None, :],  # [B, H, W, 1, C]
            )

        mesh_ras_result_tmp.close()

        mesh_ras_result_tmp = utils.PickleReader(mesh_ras_result_tmp_path)

        tex_vert_color_mean, tex_vert_color_pca, tex_vert_color_std = \
            tex_vert_color_pca_calculator.get_pca(True)

        # tex_vert_color_mean[TV + 1, C]
        # tex_vert_color_pca[TV + 1, C, C]
        # tex_vert_color_std[TV + 1, C]

        use_pca_threshold = 10.0

        tex_vert_color_ell_mean = tex_vert_color_mean
        # [TV + 1, C]

        inv_tex_vert_color_ell_axis = torch.where(
            (use_pca_threshold <=
             tex_vert_color_pca_calculator.sum_w)[..., None, None]
            .expand(TV + 1, C, C),

            (tex_vert_color_pca *
             tex_vert_color_std[:, :, None]).transpose(-1, -2),

            torch.eye(
                C,
                dtype=tex_vert_color_pca.dtype,
                device=tex_vert_color_pca.device
            ).expand(TV + 1, C, C)
        ).inverse()
        # [F, C, C]

        tex_vert_inlier_color_pca_calculator = pca_utils.PCACalculator(
            n=TV + 1,
            dim=C,
            dtype=torch.float64,
            device=self.__config.device,
        )

        for t in tqdm.tqdm(range(T)):
            utils.mem_clear()

            sample: gom_utils.Sample = self.dataset[t]

            pix_to_face = utils.tensor_deserialize(
                mesh_ras_result_tmp.read(),
                device=self.__config.device,
            )
            # [H, W]

            bary_coord = utils.tensor_deserialize(
                mesh_ras_result_tmp.read(),
                device=self.__config.device,
            )
            # [H, W, 3]

            ref_img = einops.rearrange(
                sample.img, "c h w -> h w c")
            # [H, W, C]

            tex_vert = tex_f_to_vvv[pix_to_face]
            # [H, W, 3]

            cur_tex_vert_color_ell_mean = tex_vert_color_ell_mean[tex_vert]
            # [H, W, 3, C]

            cur_inv_tex_vert_color_ell_axis = \
                inv_tex_vert_color_ell_axis[tex_vert]
            # [H, W, 3, C, C]

            ell_coord = (
                cur_inv_tex_vert_color_ell_axis  # [H, W, 3, C, C]
                @
                (ref_img[:, :, None, :] - cur_tex_vert_color_ell_mean)
                [..., None]  # [H, W, 3, C, 1]
            )[..., 0]
            # [H, W, 3, C]

            weight = bary_coord / (0.1 + utils.vec_sq_norm(ell_coord))
            # [H, W, 3]

            tex_vert_inlier_color_pca_calculator.scatter_feed(
                idx=tex_f_to_vvv[pix_to_face],  # [H, W, 3]
                w=weight,  # [H, W, 3]
                x=ref_img[:, :, None, :],  # [H, W, 1, C]
            )

        tex_vert_inlier_color_mean, tex_vert_inlier_color_pca, tex_vert_inlier_color_std = \
            tex_vert_inlier_color_pca_calculator.get_pca(True)

        # tex_vert_inlier_color_mean[TV + 1, C]

        utils.write_tensor_to_file(
            self.__config.proj_dir /
            f"tex_vert_inlier_color_mean_{utils.timestamp_sec()}.txt",
            tex_vert_inlier_color_mean,
        )

        tex_frags = texture_utils.rasterize_texture_map(
            tex_vert_pos=tex_vert_pos,
            tex_faces=tex_f_to_vvv,

            tex_h=tex_h,
            tex_w=tex_w,
        )

        tex_pix_to_face = tex_frags.pix_to_face.reshape(tex_h, tex_w)

        tex_bary_coord = tex_frags.bary_coords.reshape(tex_h, tex_w, 3)

        tex_vert_inlier_color_mean[TV] = 1.0

        tex = (tex_vert_inlier_color_mean[tex_f_to_vvv[
            (tex_pix_to_face + (F + 1)) % (F + 1)
        ]] * tex_bary_coord[..., None]).sum(dim=-2)

        """

        pix_color_interp = torchrbf.RBFInterpolator(
            y=texture_utils.tex_coord_to_img_coord(
                tex_vert_pos, tex_h, tex_w)[:-1].to(utils.CPU_DEVICE),
            # [TV, 2]

            d=tex_vert_inlier_color_mean.detach()[:-1].to(utils.CPU_DEVICE),
            # [TV, C]

            smoothing=1.0,
            kernel="thin_plate_spline",
        ).to(self.__config.device)

        tex_pix_idx_grid = utils.idx_grid(
            (tex_h, tex_w), dtype=torch.float, device=self.__config.device)
        # [H, W, 2]

        tex = torch.empty(
            (tex_h, tex_w, C),
            dtype=torch.float,
            device=self.__config.device
        )

        for h in range(tex_h):
            tex[h, :, :] = pix_color_interp(tex_pix_idx_grid[h, :, :])

        """

        vision_utils.write_image(
            path=self.__config.proj_dir / f"tex_{utils.timestamp_sec()}.png",
            img=vision_utils.denormalize_image(
                einops.rearrange(tex, "h w c -> c h w")),
        )
