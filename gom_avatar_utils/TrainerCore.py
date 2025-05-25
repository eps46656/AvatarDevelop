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
                vision_utils, gom_avatar_utils)
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
            utils.get_param_groups(self.module, self.config.lr),
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

        assert img_diff.isfinite().all()
        assert lap_diff.isfinite().all()
        assert nor_sim.isfinite().all()
        assert edge_var.isfinite().all()
        assert gp_color_diff.isfinite().all()
        assert gp_scale_diff.isfinite().all()

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

            cur_mask = sample.mask.to(self.config.device)

            cur_blending_param = sample.blending_param.to(
                self.config.device)

            result: ModuleForwardResult = self.module(
                camera_transform=cur_camera_transform,
                camera_config=cur_camera_config,
                img=cur_img,
                mask=cur_mask,
                blending_param=cur_blending_param,
            )

            loss = self.calc_loss(result)

            loss.backward()

            if (
                isinstance(self.module,
                           gom_avatar_utils.Module) and
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

                        th = self.config.vert_grad_norm_threshold

                        vert_pos.grad *= th / grad_norm.clamp(th, None)

                        new_grad_norm = utils.vec_norm(vert_pos.grad)

                        print(f"{new_grad_norm.min()=}")
                        print(f"{new_grad_norm.max()=}")

            self.optimizer.step()

        self.scheduler.step()

        self.epoch += 1

        return training_utils.TrainingResult("")

    @utils.mem_clear
    def show_params(self):
        def _f(x):
            return None if x is None else f"{x:+.4e}"

        tab = [(
            "name",
            "dtype",
            "shape",
            "device",
            "requires_grad",
            "grad_min",
            "grad_max",
            "grad_mean",
            "grad_sum",
            "grad_std",
        )]

        for name, param in self.module.named_parameters():
            dtype = param.dtype
            shape = param.shape
            device = param.device
            requires_grad = param.requires_grad

            grad = param.grad

            if grad is None:
                grad_min = None
                grad_max = None
                grad_mean = None
                grad_sum = None
                grad_std = None
            else:
                grad_min = grad.min().item()
                grad_max = grad.max().item()
                grad_mean = grad.mean().item()
                grad_sum = grad.sum().item()
                grad_std = grad.std().item()

            tab.append((
                name,
                dtype,
                shape,
                device,
                requires_grad,
                _f(grad_min),
                _f(grad_max),
                _f(grad_mean),
                _f(grad_sum),
                _f(grad_std),
            ))

        print(tabulate.tabulate(tab, tablefmt="grid"))

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

            cur_camera_transform = sample.camera_transform.to(
                self.config.device)

            cur_camera_config = sample.camera_config

            cur_img = sample.img.to(self.config.device)

            cur_mask = sample.mask.to(self.config.device)

            cur_blending_param = sample.blending_param.to(
                self.config.device)

            result: ModuleForwardResult = self.module(
                camera_config=cur_camera_config,
                camera_transform=cur_camera_transform,
                img=cur_img,
                mask=cur_mask,
                blending_param=cur_blending_param,
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
