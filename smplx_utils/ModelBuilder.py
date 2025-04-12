from __future__ import annotations

import copy

import torch
import torchrbf
from beartype import beartype

from .. import utils
from .ModelData import ModelData


@beartype
class ModelBuilder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_model_data(self) -> ModelData:
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        raise NotImplementedError()

    def to(self) -> ModelBuilder:
        raise NotImplementedError()

    def forward(self) -> ModelData:
        raise NotImplementedError()


@beartype
class StaticModelBuilder(ModelBuilder):
    def __init__(self, model_data: ModelData):
        super().__init__()
        self.model_data = model_data

    def get_model_data(self) -> ModelData:
        return self.model_data

    @property
    def device(self) -> torch.device:
        return self.model_data.device

    def to(self, *args, **kwargs) -> StaticModelBuilder:
        self.model_data = self.model_data.to(*args, **kwargs)
        return self

    def forward(self) -> ModelData:
        return self.model_data


@beartype
class DeformableModelBuilder(ModelBuilder):
    def __init__(self, model_data: ModelData):
        super().__init__()
        self.model_data = copy.copy(model_data)

        self.model_data.vert_pos = torch.nn.Parameter(
            self.model_data.vert_pos.to(
                dtype=torch.float64, copy=True,
            ))

        self.register_parameter("vert_pos", self.model_data.vert_pos)

        cpu_vert_pos = self.model_data.vert_pos.detach().to(
            device=utils.CPU_DEVICE, dtype=torch.float)

        dd = (utils.CPU_DEVICE, cpu_vert_pos.dtype)

        J = self.model_data.joints_cnt
        V = self.model_data.verts_cnt
        BS = self.model_data.body_shapes_cnt

        body_shape_vert_dir = self.model_data.body_shape_vert_dir
        # [V, 3, BS]

        if body_shape_vert_dir is None:
            self.body_shape_vert_dir_rbf_interp = None
        else:
            self.body_shape_vert_dir_rbf_interp = torchrbf.RBFInterpolator(
                y=cpu_vert_pos,  # [V, 3]
                d=body_shape_vert_dir.detach().to(*dd).reshape(V, 3 * BS),
                smoothing=1.0,
                kernel="thin_plate_spline",
            )

            self.body_shape_vert_dir_rbf_interp.to(self.model_data.device)

        pose_vert_dir = self.model_data.pose_vert_dir
        # [V, 3, (J - 1) * 3 * 3]

        if pose_vert_dir is None:
            self.pose_vert_dir_rbf_interp = None
        else:
            self.pose_vert_dir_rbf_interp = torchrbf.RBFInterpolator(
                y=cpu_vert_pos,  # [V, 3]
                d=pose_vert_dir.detach().to(*dd).reshape(
                    V, 3 * (J - 1) * 3 * 3),
                smoothing=1.0,
                kernel="thin_plate_spline",
            )

            self.pose_vert_dir_rbf_interp.to(self.model_data.device)

        lbs_weight = self.model_data.lbs_weight
        # [V, J]

        if lbs_weight is None:
            self.lbs_weight_rbf_interp = None
        else:
            self.lbs_weight_rbf_interp = torchrbf.RBFInterpolator(
                y=cpu_vert_pos,  # [V, 3]
                d=lbs_weight.detach().to(*dd),  # [V, J]
                smoothing=1.0,
                kernel="thin_plate_spline",
            )

            self.lbs_weight_rbf_interp.to(self.model_data.device)

        self.refresh()

    def get_model_data(self) -> ModelData:
        return self.model_data

    @property
    def device(self) -> torch.device:
        return self.model_data.device

    def to(self, *args, **kwargs) -> DeformableModelBuilder:
        self.model_data = self.model_data.to(*args, **kwargs)
        return self

    def freeze(self):
        self.model_data.vert_pos.requires_grad = False

    def unfreeze(self):
        self.model_data.vert_pos.requires_grad = True

    def get_param_groups(self, base_lr: float):
        ret = list()

        if self.model_data.vert_pos.requires_grad:
            ret.append({
                "params": [self.model_data.vert_pos],
                "lr": base_lr,
            })

        return ret

    def forward(self) -> ModelData:
        return self.model_data

    def refresh(self) -> None:
        J = self.model_data.joints_cnt
        V = self.model_data.verts_cnt
        BS = self.model_data.body_shapes_cnt

        vert_pos = self.model_data.vert_pos.to(torch.float)

        if self.body_shape_vert_dir_rbf_interp is not None:
            self.model_data.body_shape_vert_dir = \
                self.body_shape_vert_dir_rbf_interp(vert_pos).view(
                    V, 3, BS)

        if self.pose_vert_dir_rbf_interp is not None:
            self.model_data.pose_vert_dir = \
                self.pose_vert_dir_rbf_interp(vert_pos).view(
                    V, 3, (J - 1) * 3 * 3)

        if self.lbs_weight_rbf_interp is not None:
            self.model_data.lbs_weight = \
                self.lbs_weight_rbf_interp(vert_pos)  # [V, J]

        utils.print_cur_pos()
