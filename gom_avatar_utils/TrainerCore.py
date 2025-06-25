import collections
import dataclasses
import math
import typing

import einops
import numpy as np
import tabulate
import torch
import tqdm
from beartype import beartype

import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures

from .. import (dataset_utils, mesh_utils, smplx_utils, training_utils, utils,
                vision_utils, gom_avatar_utils)
from .Dataset import *
from .Module import *


@dataclasses.dataclass
class TrainerCoreConfig:
    proj_dir: utils.PathLike

    device: torch.device

    batch_size: int

    lr: typing.Callable[[int], float]
    betas: tuple[float, float]

    silhouette_sigma: float
    silhouette_opacity: float

    vert_grad_norm_threshold: float

    alpha_img_diff: typing.Callable[[int], float]
    alpha_gp_mask_diff: typing.Callable[[int], list[float]]
    alpha_lap_diff: typing.Callable[[int], float]
    alpha_nor_sim: typing.Callable[[int], float]
    alpha_edge_var: typing.Callable[[int], float]


@beartype
class TrainerCore(training_utils.TrainerCore):
    def __init__(
        self,
        config: TrainerCoreConfig,
        module: Module,
        dataset: Dataset,
    ):
        self.config = dataclasses.replace(config)
        self.config.proj_dir = utils.to_pathlib_path(self.config.proj_dir)

        self.epoch = 0

        self.module = module
        self.dataset = dataset

        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()

    # --

    def get_epoch(self) -> int:
        return self.epoch

    # --

    def _make_optimizer(self) -> torch.optim.Optimizer:
        base_lr = self.config.lr(0)

        return torch.optim.Adam(
            utils.get_param_groups(self.module, base_lr),
            lr=base_lr,
            betas=self.config.betas,
        )

    def _make_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda epoch: self.config.lr(epoch) / self.config.lr(0),
            last_epoch=self.epoch - 1,
        )

    # --

    def state_dict(self, full: bool) \
            -> collections.OrderedDict[str, typing.Any]:
        l = [
            ("epoch", self.epoch),
            ("full", full),
        ]

        if full:
            l.append(("module", self.module))
        else:
            l.append(("module", self.module.state_dict()))

        l.append(("optimizer", self.optimizer.state_dict()))
        l.append(("scheduler", self.scheduler.state_dict()))

        return collections.OrderedDict(l)

    def load_state_dict(self, state_dict: typing.Mapping[str, typing.Any]) \
            -> None:
        self.epoch = state_dict["epoch"]

        full = state_dict["full"]

        if full:
            self.module = state_dict["module"]
        else:
            self.module.load_state_dict(state_dict["module"])

        self.optimizer = self._make_optimizer()
        self.optimizer.load_state_dict(state_dict["optimizer"])

        self.scheduler = self._make_scheduler()
        self.scheduler.load_state_dict(state_dict["scheduler"])

    # --

    def calc_loss(
        self,
        forward_result: ModuleForwardResult,
    ) -> torch.Tensor:

        loss_table = utils.LossTable()

        # ---

        print(f"{forward_result.img_diff.min()=}")
        print(f"{forward_result.img_diff.max()=}")

        img_diff = forward_result.img_diff.mean()
        w_img_diff = \
            self.config.alpha_img_diff(self.epoch) * img_diff

        loss_table.add("img_diff", img_diff, w_img_diff)

        # ---

        for sigma_idx in range(len(self.config.alpha_gp_mask_diff)):
            gp_mask_diff = forward_result.gp_mask_diff[sigma_idx].mean()

            w_gp_mask_diff = \
                self.config.alpha_gp_mask_diff[sigma_idx](self.epoch) * \
                gp_mask_diff

            loss_table.add(
                f"gp_mask_diff_{sigma_idx}",
                gp_mask_diff,
                w_gp_mask_diff,
            )

        # ---

        lap_diff = forward_result.lap_diff.mean()
        w_lap_diff = \
            self.config.alpha_lap_diff(self.epoch) * lap_diff

        loss_table.add("lap_diff", lap_diff, w_lap_diff)

        # ---

        nor_sim = forward_result.nor_sim.mean()
        w_nor_sim = \
            self.config.alpha_nor_sim(self.epoch) * nor_sim

        loss_table.add("nor_sim", nor_sim, w_nor_sim)

        # ---

        edge_var = forward_result.edge_var.mean()
        w_edge_var = \
            self.config.alpha_edge_var(self.epoch) * edge_var

        loss_table.add("edge_var", edge_var, w_edge_var)

        # ---

        loss_table.show()

        loss = loss_table.get_weighted_sum_loss()

        assert loss.isfinite().all()

        print(f"{loss=}")

        return loss

    # --

    @utils.mem_clear
    def train(self) -> training_utils.TrainingResult:
        for batch_size, batch_idx, sample in tqdm.tqdm(dataset_utils.load(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )):
            self.module.refresh()

            self.optimizer.zero_grad()

            sample: Sample

            cur_camera_transform = sample.camera_transform.to(
                self.config.device)

            cur_camera_config = sample.camera_config

            cur_img = sample.img.to(self.config.device)

            cur_mask = sample.mask.to(self.config.device, torch.float64)

            cur_dilated_mask = [
                x.to(self.config.device, torch.float64)
                for x in sample.dilated_mask
            ]

            cur_blending_param = sample.blending_param \
                .to(self.config.device)

            result: ModuleForwardResult = self.module(
                camera_transform=cur_camera_transform,
                camera_config=cur_camera_config,
                img=cur_img,

                mask=cur_mask,
                dilated_mask=cur_dilated_mask,

                blending_param=cur_blending_param,

                silhouette_sigma=self.config.silhouette_sigma,
                silhouette_opacity=self.config.silhouette_opacity,
            )

            loss = self.calc_loss(result)

            loss.backward()

            vert_grad_norm_th = self.config.vert_grad_norm_threshold

            if (
                isinstance(self.module.avatar_blender,
                           smplx_utils.ModelBlender) and
                isinstance(self.module.avatar_blender.model_builder,
                           smplx_utils.DeformableModelBuilder)
            ):
                vert_pos = self.module.avatar_blender.model_builder.vert_pos
                # [V, D]

                if vert_pos.grad is not None:
                    with torch.no_grad():
                        grad_norm = utils.vec_norm(vert_pos.grad, keepdim=True)
                        # [V, 1]

                        if 0 < vert_grad_norm_th:
                            vert_pos.grad *= vert_grad_norm_th / \
                                grad_norm.clamp(vert_grad_norm_th, None)

                        new_grad_norm = utils.vec_norm(vert_pos.grad)

                        print(f"{new_grad_norm.min()=}")
                        print(f"{new_grad_norm.max()=}")

            self.optimizer.step()

        self.scheduler.step()

        self.epoch += 1

        return training_utils.TrainingResult("")

    @utils.mem_clear
    def show_params(self) -> None:
        utils.show_tensor_info(self.module.named_parameters())

    @utils.mem_clear
    @torch.no_grad()
    def output_model_data(self):
        self.module.refresh()

        if (
            isinstance(self.module.avatar_blender,
                       smplx_utils.ModelBlender) and
            isinstance(self.module.avatar_blender.model_builder,
                       smplx_utils.DeformableModelBuilder)
        ):
            model_data = self.module.avatar_blender.model_builder.model_data

            model_data.show()

            model_data_state_dict = model_data.state_dict()

            utils.write_pickle(
                self.config.proj_dir /
                f"model_data_{utils.timestamp_sec()}.pkl",

                model_data_state_dict,
            )

            utils.write_pickle(
                self.config.proj_dir /
                f"model_data.pkl",

                model_data_state_dict,
            )

    @utils.mem_clear
    @torch.no_grad()
    def output_rgb_video(self):
        # call output_rgb_video
        self.module.refresh()

        T, C, H, W = self.dataset.sample.img.shape

        avatar_model: smplx_utils.Model = \
            self.module.avatar_blender.get_avatar_model()

        avatar_model.mesh_graph.show(avatar_model.vert_pos)

        video_writer = vision_utils.VideoWriter(
            path=self.config.proj_dir / f"rgb_{utils.timestamp_sec()}.avi",
            height=H,
            width=W,
            color_type="RGB",
            fps=25.0,
        )

        for batch_size, batch_idx, sample in tqdm.tqdm(dataset_utils.load(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )):
            sample: Sample

            cur_camera_transform = sample.camera_transform.to(
                self.config.device)

            cur_camera_config = sample.camera_config

            cur_img = sample.img.to(self.config.device)

            cur_mask = sample.mask.to(self.config.device, torch.float64)

            cur_dilated_mask = [
                x.to(self.config.device, torch.float64)
                for x in sample.dilated_mask
            ]

            cur_blending_param = sample.blending_param.to(
                self.config.device)

            result: ModuleForwardResult = self.module(
                camera_transform=cur_camera_transform,
                camera_config=cur_camera_config,
                img=cur_img,

                mask=cur_mask,
                dilated_mask=cur_dilated_mask,

                blending_param=cur_blending_param,

                silhouette_sigma=self.config.silhouette_sigma,
                silhouette_opacity=self.config.silhouette_opacity,
            )

            rendered_img = result.gp_render_img.reshape(-1, C, H, W)
            # [K, C, H, W]

            for i in rendered_img:
                video_writer.write(utils.rct(i * 255, dtype=torch.uint8))

        video_writer.close()

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

        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()

    @utils.mem_clear
    @torch.no_grad()
    def output_figure(self):
        frame_idx = 30

        self.module.refresh()

        sample: Sample = self.dataset.sample[frame_idx]

        cur_camera_transform = sample.camera_transform.to(
            self.config.device)

        cur_camera_config = sample.camera_config

        cur_img = sample.img.to(self.config.device)

        cur_mask = sample.mask.to(self.config.device, torch.float64)

        cur_dilated_mask = [
            x.to(self.config.device, torch.float64)
            for x in sample.dilated_mask
        ]

        cur_blending_param = sample.blending_param \
            .to(self.config.device)

        result: ModuleForwardResult = self.module(
            camera_transform=cur_camera_transform,
            camera_config=cur_camera_config,
            img=cur_img,

            mask=cur_mask,
            dilated_mask=cur_dilated_mask,

            blending_param=cur_blending_param,

            silhouette_sigma=self.config.silhouette_sigma,
            silhouette_opacity=self.config.silhouette_opacity,
        )

        avatar_model: smplx_utils.Model = result.avatar_model

        light_img = rendering_utils.make_light_map(
            camera_config=cur_camera_config,
            camera_transform=cur_camera_transform,

            mesh_data=avatar_model.mesh_data,

            pix_to_face=None,
            bary_coord=None,
        )  # [4, H, W]

        vision_utils.write_image(
            self.config.proj_dir /
            f"video_{utils.timestamp_sec()}.png",
            utils.rct(cur_img * 255, dtype=torch.uint8),
        )

        vision_utils.write_image(
            self.config.proj_dir /
            f"light_map_{utils.timestamp_sec()}.png",
            utils.rct(light_img * 255, dtype=torch.uint8),
        )

        vision_utils.write_image(
            self.config.proj_dir /
            f"gaussian_rendered_image_{utils.timestamp_sec()}.png",
            utils.rct(result.gp_render_img * 255, dtype=torch.uint8),
        )

        cur_blending_param.body_shape = None
        cur_blending_param.expr_shape = None

        cur_blending_param.body_pose = None
        cur_blending_param.jaw_pose = None
        cur_blending_param.leye_pose = None
        cur_blending_param.reye_pose = None

        cur_blending_param.lhand_pose = None
        cur_blending_param.rhand_pose = None

        canonical_result: ModuleForwardResult = self.module(
            camera_transform=cur_camera_transform,
            camera_config=cur_camera_config,
            img=cur_img,

            mask=cur_mask,
            dilated_mask=cur_dilated_mask,

            blending_param=cur_blending_param,

            silhouette_sigma=self.config.silhouette_sigma,
            silhouette_opacity=self.config.silhouette_opacity,
        )

        vision_utils.write_image(
            self.config.proj_dir /
            f"canonical_gaussian_rendered_image_{utils.timestamp_sec()}.png",
            utils.rct(canonical_result.gp_render_img * 255, dtype=torch.uint8),
        )

        canonical_avatar_model: smplx_utils.Model = \
            canonical_result.avatar_model

        canonical_mesh_rendered_image = rendering_utils.make_light_map(
            camera_config=cur_camera_config,
            camera_transform=cur_camera_transform,
            mesh_data=mesh_utils.MeshData(
                mesh_graph=canonical_avatar_model.mesh_graph,
                vert_pos=canonical_avatar_model.vert_pos
            ),
            pix_to_face=None,
            bary_coord=None,
        )

        vision_utils.write_image(
            self.config.proj_dir /
            f"canonical_mesh_rendered_image_{utils.timestamp_sec()}.png",
            utils.rct(canonical_mesh_rendered_image *
                      255, dtype=torch.uint8),
        )
