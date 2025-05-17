import collections
import dataclasses
import math
import os
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
                vision_utils)
from .Dataset import *
from .Module import *


@dataclasses.dataclass
class TrainerCoreConfig:
    proj_dir: os.PathLike

    device: torch.device

    batch_size: int

    lr: float
    betas: tuple[float, float]
    gamma: float

    vert_grad_norm_threshold: float

    lap_diff_clamp_norm: float

    alpha_img_diff: float
    alpha_lap_diff: float
    alpha_nor_sim: float
    alpha_edge_var: float
    alpha_gp_color_diff: float
    alpha_gp_scale_diff: float


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
        return torch.optim.Adam(
            self.module.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
        )

    def _make_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR:
        return torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.config.gamma,
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
        def _f(x):
            return f"{x.item():+.4e}"

        losses = dict()

        loss_tab = [("", "loss", "weighted loss")]

        # ---

        img_diff = forward_result.img_diff.mean()
        w_img_diff = self.config.alpha_img_diff * img_diff

        loss_tab.append(("img_loss", _f(img_diff), _f(w_img_diff)))

        # ---

        lap_diff = forward_result.lap_diff.mean()
        w_lap_diff = self.config.alpha_lap_diff * lap_diff

        loss_tab.append(("lap_loss", _f(lap_diff), _f(w_lap_diff)))

        # ---

        nor_sim = forward_result.nor_sim.mean()
        w_nor_sim = self.config.alpha_nor_sim * nor_sim

        loss_tab.append(("nor_sim", _f(nor_sim), _f(w_nor_sim)))

        # ---

        edge_var = forward_result.edge_var.mean()
        w_edge_var = self.config.alpha_edge_var * edge_var

        loss_tab.append(("edge_var", _f(edge_var), _f(w_edge_var)))

        # ---

        gp_color_diff = forward_result.edge_var.mean()
        w_gp_color_diff = self.config.alpha_gp_color_diff * gp_color_diff

        loss_tab.append((
            "gp_color_diff", _f(gp_color_diff), _f(w_gp_color_diff)))

        # ---

        gp_scale_diff = forward_result.edge_var.mean()
        w_gp_scale_diff = self.config.alpha_gp_scale_diff * gp_scale_diff

        loss_tab.append((
            "gp_scale_diff", _f(gp_scale_diff), _f(w_gp_scale_diff)))

        # ---

        print(tabulate.tabulate(zip(*loss_tab), tablefmt="grid"))

        return \
            w_img_diff + \
            w_lap_diff + \
            w_nor_sim + \
            w_edge_var + \
            w_gp_color_diff + \
            w_gp_scale_diff

    # --

    @utils.mem_clear
    def train(self) -> training_utils.TrainingResult:
        sum_loss = 0.0

        for batch_size, batch_idx, sample in tqdm.tqdm(dataset_utils.load(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )):
            sample: Sample

            self.module.refresh()

            result: ModuleForwardResult = self.module(
                camera_transform=sample.camera_transform,
                camera_config=sample.camera_config,
                img=sample.img,
                mask=sample.mask,
                blending_param=sample.blending_param,
                lap_diff_clamp_norm=self.config.lap_diff_clamp_norm,
            )

            loss: torch.Tensor = self.calc_loss(result)

            sum_loss += float(loss) * batch_size

            self.optimizer.zero_grad()

            loss.backward()

            avatar_blender = self.module.avatar_blender

            if isinstance(avatar_blender, smplx_utils.ModelBlender) and isinstance(avatar_blender.model_builder, smplx_utils.DeformableModelBuilder):
                model_builder: smplx_utils.DeformableModelBuilder = avatar_blender.model_builder

                vert_pos = model_builder.model_data.vert_pos

                vert_grad_norm_threshold = self.config.vert_grad_norm_threshold

                if vert_pos.grad is not None and vert_grad_norm_threshold is not None:
                    grad_norm = utils.vec_norm(vert_pos.grad, -1, True)

                    vert_pos.grad *= vert_grad_norm_threshold / \
                        grad_norm.clamp(vert_grad_norm_threshold)

            self.optimizer.step()

        avg_loss = sum_loss / self.dataset.shape.numel()

        self.scheduler.step()

        return training_utils.TrainingResult(
            message=str(avg_loss)
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
        self.dataset: Dataset

        T, C, H, W = self.dataset.sample.img.shape

        avatar_model: smplx_utils.Model = \
            self.module.avatar_blender.get_avatar_model()

        avatar_model.mesh_graph.show(avatar_model.vert_pos)

        video_writer = vision_utils.VideoWriter(
            path=self.config.proj_dir / f"rgb_{utils.timestamp_sec()}.avi",
            height=H,
            width=W,
            color_type=vision_utils.ColorType.RGB,
            fps=25.0,
        )

        for batch_size, batch_idx, sample in tqdm.tqdm(dataset_utils.load(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )):
            sample: Sample

            result: ModuleForwardResult = self.module(
                camera_config=sample.camera_config,
                camera_transform=sample.camera_transform,
                img=sample.img,
                mask=sample.mask,
                blending_param=sample.blending_param,
                lap_diff_clamp_norm=self.config.lap_diff_clamp_norm,
            )

            rendered_img = result.gp_render_img.reshape(-1, C, H, W)
            # [K, C, H, W]

            for i in rendered_img:
                video_writer.write(vision_utils.denormalize_image(i))

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
