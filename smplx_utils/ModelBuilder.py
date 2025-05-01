from __future__ import annotations

import collections
import copy
import dataclasses
import typing

import torch
import torchrbf
from beartype import beartype

from .. import utils
from .ModelData import ModelData, ModelDataSubdivisionResult


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

    def subdivide(
        self,
        *,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> ModelDataSubdivisionResult:
        raise NotImplementedError()

    def show(self) -> None:
        self.get_model_data().show()


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

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("model_data", self.model_data.state_dict())])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        self.model_data.load_state_dict(state_dict["model_data"])

    def forward(self) -> ModelData:
        return self.model_data


@beartype
@dataclasses.dataclass
class DeoformingBlendingCoeff:
    body_shape_vert_dir: torch.Tensor  # [..., 3, BS]
    expr_shape_vert_dir: torch.Tensor  # [..., 3, BS]
    pose_vert_dir: torch.Tensor  # [..., 3, (J - 1) * D * D]
    lbs_weight: torch.Tensor  # [..., J]


@beartype
class DeformableModelBuilder(ModelBuilder):
    def __init__(
        self,
        temp_model_data: ModelData,
        model_data: ModelData,
    ):
        super().__init__()
        self.model_data = copy.copy(model_data)

        self.model_data.vert_pos = torch.nn.Parameter(
            self.model_data.vert_pos.to(dtype=torch.float64, copy=True))

        self.vert_pos = self.model_data.vert_pos

        cpu_temp_vert_pos = temp_model_data.vert_pos.detach().to(
            device=utils.CPU_DEVICE, dtype=torch.float)

        dd = (utils.CPU_DEVICE, cpu_temp_vert_pos.dtype)

        J = self.model_data.joints_cnt
        V = self.model_data.verts_cnt
        BS = self.model_data.body_shapes_cnt
        ES = self.model_data.expr_shapes_cnt

        if BS == 0:
            self.body_shape_vert_dir_interp = None
        else:
            self.body_shape_vert_dir_interp = torchrbf.RBFInterpolator(
                y=cpu_temp_vert_pos,  # [V, 3]

                d=temp_model_data.body_shape_vert_dir
                .detach().to(*dd).reshape(V, 3 * BS),

                smoothing=1.0,
                kernel="thin_plate_spline",
            ).to(self.model_data.device, torch.float64)

        if ES == 0:
            self.expr_shape_vert_dir_interp = None
        else:
            self.expr_shape_vert_dir_interp = torchrbf.RBFInterpolator(
                y=cpu_temp_vert_pos,  # [V, 3]

                d=temp_model_data.expr_shape_vert_dir
                .detach().to(*dd).reshape(V, 3 * ES),

                smoothing=1.0,
                kernel="thin_plate_spline",
            ).to(self.model_data.device, torch.float64)

        if J == 0:
            self.pose_vert_dir_interp = None
        else:
            self.pose_vert_dir_interp = torchrbf.RBFInterpolator(
                y=cpu_temp_vert_pos,  # [V, 3]

                d=temp_model_data.pose_vert_dir
                .detach().to(*dd).reshape(V, 3 * (J - 1) * 3 * 3),

                smoothing=1.0,
                kernel="thin_plate_spline",
            ).to(self.model_data.device, torch.float64)

        if J == 0:
            self.lbs_weight_interp = None
        else:
            self.lbs_weight_interp = torchrbf.RBFInterpolator(
                y=cpu_temp_vert_pos,  # [V, 3]

                d=temp_model_data.lbs_weight.detach().to(*dd),  # [V, J]

                smoothing=1.0,
                kernel="thin_plate_spline",
            ).to(self.model_data.device, torch.float64)

    def get_model_data(self) -> ModelData:
        return self.model_data

    @property
    def device(self) -> torch.device:
        return self.model_data.device

    def to(self, *args, **kwargs) -> DeformableModelBuilder:
        self.model_data = self.model_data.to(*args, **kwargs)
        return self

    def freeze(self) -> None:
        self.model_data.vert_pos.requires_grad = False

    def unfreeze(self) -> None:
        self.model_data.vert_pos.requires_grad = True

    def get_param_groups(self, base_lr: float):
        ret = list()

        if self.model_data.vert_pos.requires_grad:
            ret.append({
                "params": [self.model_data.vert_pos],
                "lr": min(1e-6, base_lr * 1e-2),
            })

        return ret

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("model_data", self.model_data.state_dict()),

            ("body_shape_vert_dir_interp",
             self.body_shape_vert_dir_interp.state_dict()),

            ("expr_shape_vert_dir_interp",
             self.expr_shape_vert_dir_interp.state_dict()),

            ("pose_vert_dir_interp",
             self.pose_vert_dir_interp.state_dict()),

            ("lbs_weight_interp",
             self.lbs_weight_interp.state_dict()),
        ])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        self.model_data.load_state_dict(state_dict["model_data"])

        if self.body_shape_vert_dir_interp is not None:
            self.body_shape_vert_dir_interp.load_state_dict(
                state_dict["body_shape_vert_dir_interp"])

        if self.expr_shape_vert_dir_interp is not None:
            self.expr_shape_vert_dir_interp.load_state_dict(
                state_dict["expr_shape_vert_dir_interp"])

        if self.pose_vert_dir_interp is not None:
            self.pose_vert_dir_interp.load_state_dict(
                state_dict["pose_vert_dir_interp"])

        if self.lbs_weight_interp is not None:
            self.lbs_weight_interp.load_state_dict(
                state_dict["lbs_weight_interp"])

    def subdivide(
        self,
        *,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> ModelDataSubdivisionResult:
        model_data_subdivision_result = self.model_data.subdivide(
            target_edges=target_edges,
            target_faces=target_faces,
        )

        utils.torch_cuda_sync()

        self.model_data = model_data_subdivision_result.model_data

        utils.torch_cuda_sync()

        self.model_data.vert_pos = torch.nn.Parameter(
            self.model_data.vert_pos.to(dtype=torch.float64, copy=True))

        utils.torch_cuda_sync()

        self.vert_pos = self.model_data.vert_pos
        print(f"set vert_pos, {self.vert_pos.shape=}")

        utils.torch_cuda_sync()

        return model_data_subdivision_result

    def forward(self) -> ModelData:
        return self.model_data

    def query_blending_coeff(
        self,
        vert_pos: torch.Tensor,  # [..., 3]
    ) -> DeoformingBlendingCoeff:
        J = self.model_data.joints_cnt
        BS = self.model_data.body_shapes_cnt
        ES = self.model_data.expr_shapes_cnt

        vert_pos = vert_pos.to(self.device)

        if BS == 0:
            body_shape_vert_dir = utils.zeros_like(
                vert_pos, shape=(*vert_pos.shape[:-1], 3, BS))
        else:
            body_shape_vert_dir = self.body_shape_vert_dir_interp(
                vert_pos).view(*vert_pos.shape[:-1], 3, BS)

        if ES == 0:
            expr_shape_vert_dir = utils.zeros_like(
                vert_pos, shape=(*vert_pos.shape[:-1], 3, ES))
        else:
            expr_shape_vert_dir = self.expr_shape_vert_dir_interp(
                vert_pos).view(*vert_pos.shape[:-1], 3, ES)

        if self.pose_vert_dir_interp is None:
            pose_vert_dir = utils.zeros_like(
                vert_pos, shape=(*vert_pos.shape[:-1], 3, 0))
        else:
            pose_vert_dir = self.pose_vert_dir_interp(
                vert_pos).view(*vert_pos.shape[:-1], 3, (J - 1) * 3 * 3)

        if self.lbs_weight_interp is None:
            lbs_weight = utils.zeros_like(
                vert_pos, shape=(*vert_pos.shape[:-1], 0))
        else:
            lbs_weight = self.lbs_weight_interp(vert_pos)

        print(f"{vert_pos.shape=}")
        print(f"{lbs_weight.shape=}")

        return DeoformingBlendingCoeff(
            body_shape_vert_dir=body_shape_vert_dir,
            expr_shape_vert_dir=expr_shape_vert_dir,
            pose_vert_dir=pose_vert_dir,
            lbs_weight=lbs_weight,
        )

    def refresh(self) -> None:
        print(f"1 {self.vert_pos.shape=}")

        self.model_data.vert_pos = self.vert_pos

        blending_coeff = self.query_blending_coeff(self.vert_pos)

        print(f"2 {self.vert_pos.shape=}")

        self.model_data.body_shape_vert_dir = \
            blending_coeff.body_shape_vert_dir

        self.model_data.expr_shape_vert_dir = \
            blending_coeff.expr_shape_vert_dir

        self.model_data.pose_vert_dir = \
            blending_coeff.pose_vert_dir

        self.model_data.lbs_weight = blending_coeff.lbs_weight

    def remesh(self, remesh_arg_pack: utils.ArgPack) -> DeformableModelBuilder:
        remeshed_model_data = self.model_data.remesh(remesh_arg_pack)

        remeshed_model_data.vert_pos = torch.nn.Parameter(
            remeshed_model_data.vert_pos.to(dtype=torch.float64, copy=True))

        self.model_data = remeshed_model_data

        self.vert_pos = remeshed_model_data.vert_pos

        return self
