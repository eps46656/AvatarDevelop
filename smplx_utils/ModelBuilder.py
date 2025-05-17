from __future__ import annotations

import collections
import copy
import dataclasses
import typing

import torch
from beartype import beartype

from .. import rbf_utils, utils
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

    def refresh(self) -> None:
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

    def refresh(self) -> None:
        return

    def forward(self) -> ModelData:
        return self.model_data


@beartype
class BlendingCoeffField:
    def __init__(
        self,
        body_shape_vert_dir_interp: typing.Optional[rbf_utils.RBFInterpolator],
        expr_shape_vert_dir_interp: typing.Optional[rbf_utils.RBFInterpolator],
        pose_vert_dir_interp: typing.Optional[rbf_utils.RBFInterpolator],
        lbs_weight_interp: typing.Optional[rbf_utils.RBFInterpolator],
    ):
        self.body_shape_vert_dir_interp = body_shape_vert_dir_interp
        self.expr_shape_vert_dir_interp = expr_shape_vert_dir_interp
        self.pose_vert_dir_interp = pose_vert_dir_interp
        self.lbs_weight_interp = lbs_weight_interp

    @staticmethod
    def from_temp(
        *,
        vert_pos: torch.Tensor,  # [V, 3]
        body_shape_vert_dir: typing.Optional[torch.Tensor],  # [V, 3, BS]
        expr_shape_vert_dir: typing.Optional[torch.Tensor],  # [V, 3, ES]

        pose_vert_dir: typing.Optional[torch.Tensor],
        # [V, 3, (J - 1) * D * D]

        lbs_weight: typing.Optional[torch.Tensor],  # [V, J]
    ):
        V = utils.check_shapes(vert_pos, (-1, 3))

        kernel = rbf_utils.radial_func.ThinPlateSplineRadialFunc()

        def _f(x):
            return None if x is None else \
                rbf_utils.RBFInterpolator.from_data_point(
                    data_pos=vert_pos,  # [V, 3]

                    data_val=x.reshape(V, -1),

                    kernel=kernel,

                    degree=3,
                    smoothness=1.0,
                )

        body_shape_vert_dir_interp = _f(body_shape_vert_dir)
        expr_shape_vert_dir_interp = _f(expr_shape_vert_dir)
        pose_vert_dir_interp = _f(pose_vert_dir)
        lbs_weight_interp = _f(lbs_weight)

        return BlendingCoeffField(
            body_shape_vert_dir_interp=body_shape_vert_dir_interp,
            expr_shape_vert_dir_interp=expr_shape_vert_dir_interp,
            pose_vert_dir_interp=pose_vert_dir_interp,
            lbs_weight_interp=lbs_weight_interp,
        )

    @staticmethod
    def from_model_data(
        model_data: ModelData,
    ) -> BlendingCoeffField:
        return BlendingCoeffField.from_temp(
            vert_pos=model_data.vert_pos,  # [V, 3]

            body_shape_vert_dir=model_data.body_shape_vert_dir,
            # [V, 3, BS]

            expr_shape_vert_dir=model_data.expr_shape_vert_dir,
            # [V, 3, ES]

            pose_vert_dir=model_data.pose_vert_dir,
            # [V, 3, (J - 1) * D * D]

            lbs_weight=model_data.lbs_weight,
            # [V, J]
        )

    @staticmethod
    def from_state_dict(self, state_dict: typing.Mapping[str, object]) -> BlendingCoeffField:
        def _f(x):
            return None if x is None else \
                rbf_utils.RBFInterpolator.from_state_dict(x)

        return BlendingCoeffField(
            body_shape_vert_dir_interp=_f(
                state_dict["body_shape_vert_dir_interp"]),

            expr_shape_vert_dir_interp=_f(
                state_dict["expr_shape_vert_dir_interp"]),

            pose_vert_dir_interp=_f(
                state_dict["pose_vert_dir_interp"]),

            lbs_weight_interp=_f(
                state_dict["lbs_weight_interp"]),
        )

    def state_dict(self) -> collections.OrderedDict[str, object]:
        def _f(x):
            return None if x is None else x.state_dict()

        return collections.OrderedDict([
            ("body_shape_vert_dir_interp", _f(self.body_shape_vert_dir_interp)),
            ("expr_shape_vert_dir_interp", _f(self.expr_shape_vert_dir_interp)),
            ("pose_vert_dir_interp", _f(self.pose_vert_dir_interp)),
            ("lbs_weight_interp", _f(self.lbs_weight_interp)),
        ])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        def _f(x):
            return None if x is None else \
                rbf_utils.RBFInterpolator.from_state_dict(x)

        self.body_shape_vert_dir_interp = \
            _f(state_dict["body_shape_vert_dir_interp"])

        self.expr_shape_vert_dir_interp = \
            _f(state_dict["expr_shape_vert_dir_interp"])

        self.pose_vert_dir_interp = \
            _f(state_dict["pose_vert_dir_interp"])

        self.lbs_weight_interp = \
            _f(state_dict["lbs_weight_interp"])

    def query_body_shape_vert_dir(self, vert_pos: torch.Tensor) -> torch.Tensor:
        V = utils.check_shapes(vert_pos, (-1, 3))

        return None if vert_pos is None else \
            self.body_shape_vert_dir_interp(vert_pos).reshape(V, 3, -1)
        # [V, 3, BS]

    def query_expr_shape_vert_dir(self, vert_pos: torch.Tensor) -> torch.Tensor:
        V = utils.check_shapes(vert_pos, (-1, 3))

        return None if vert_pos is None else \
            self.expr_shape_vert_dir_interp(vert_pos).reshape(V, 3, -1)
        # [V, 3, ES]

    def query_pose_vert_dir(self, vert_pos: torch.Tensor) -> torch.Tensor:
        V = utils.check_shapes(vert_pos, (-1, 3))

        return None if vert_pos is None else \
            self.pose_vert_dir_interp(vert_pos).reshape(V, 3, -1)
        # [V, 3, (J - 1) * D * D]

    def query_lbs_weight(self, vert_pos: torch.Tensor) -> torch.Tensor:
        V = utils.check_shapes(vert_pos, (-1, 3))

        return None if vert_pos is None else \
            self.lbs_weight_interp(vert_pos).reshape(V, -1)
        # [V, J]

    def query_model_data(self, model_data: ModelData) -> ModelData:
        new_lbs_weight = self.query_lbs_weight(model_data.vert_pos)

        if new_lbs_weight is None:
            new_lbs_weight = model_data.lbs_weight

        # ---

        new_body_shape_vert_dir = self.query_body_shape_vert_dir(
            model_data.vert_pos)

        if new_body_shape_vert_dir is None:
            new_body_shape_vert_dir = model_data.body_shape_vert_dir

        # ---

        new_expr_shape_vert_dir = self.query_expr_shape_vert_dir(
            model_data.vert_pos)

        if new_expr_shape_vert_dir is None:
            new_expr_shape_vert_dir = model_data.expr_shape_vert_dir

        # ---

        new_pose_vert_dir = self.query_pose_vert_dir(
            model_data.vert_pos)

        if new_pose_vert_dir is None:
            new_pose_vert_dir = model_data.pose_vert_dir

        # ---

        return ModelData(
            body_joints_cnt=model_data.body_joints_cnt,
            jaw_joints_cnt=model_data.jaw_joints_cnt,
            eye_joints_cnt=model_data.eye_joints_cnt,

            kin_tree=model_data.kin_tree,

            mesh_graph=model_data.mesh_graph,
            tex_mesh_graph=model_data.tex_mesh_graph,

            joint_t_mean=model_data.joint_t_mean,

            vert_pos=model_data.vert_pos,  # [V, 3]
            tex_vert_pos=model_data.tex_vert_pos,  # [TV, 2]

            lbs_weight=new_lbs_weight,  # [V, J]

            body_shape_joint_dir=model_data.body_shape_joint_dir,
            expr_shape_joint_dir=model_data.expr_shape_joint_dir,

            body_shape_vert_dir=new_body_shape_vert_dir,
            expr_shape_vert_dir=new_expr_shape_vert_dir,

            lhand_pose_mean=model_data.lhand_pose_mean,
            rhand_pose_mean=model_data.rhand_pose_mean,

            pose_vert_dir=new_pose_vert_dir,
        )


@beartype
@dataclasses.dataclass
class DeoformingBlendingCoeff:
    body_shape_vert_dir: typing.Optional[torch.Tensor]  # [..., 3, BS]
    expr_shape_vert_dir: typing.Optional[torch.Tensor]  # [..., 3, BS]
    pose_vert_dir: typing.Optional[torch.Tensor]  # [..., 3, (J - 1) * D * D]
    lbs_weight: typing.Optional[torch.Tensor]  # [..., J]


@beartype
class DeformableModelBuilder(ModelBuilder):
    def __init__(
        self,
        temp_model_data: ModelData,
        model_data: ModelData,
    ):
        super().__init__()

        self.temp_model_data = temp_model_data
        self.model_data = copy.copy(model_data)

        self.model_data.vert_pos = torch.nn.Parameter(
            self.model_data.vert_pos.to(dtype=torch.float64, copy=True))

        self.vert_pos = self.model_data.vert_pos

        self.blending_coeff_field = BlendingCoeffField.from_temp(
            vert_pos=temp_model_data.vert_pos,  # [V, 3]

            body_shape_vert_dir=temp_model_data.body_shape_vert_dir,
            # [V, 3, BS]

            expr_shape_vert_dir=temp_model_data.expr_shape_vert_dir,
            # [V, 3, ES]

            pose_vert_dir=temp_model_data.pose_vert_dir,
            # [V, 3, (J - 1) * D * D]

            lbs_weight=temp_model_data.lbs_weight,
            # [V, J]
        )

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

            ("blending_coeff_field",
             self.blending_coeff_field.state_dict()),
        ])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        self.model_data.load_state_dict(state_dict["model_data"])

        self.blending_coeff_field.load_state_dict(
            state_dict["blending_coeff_field"])

    def refresh(self) -> None:
        self.model_data.vert_pos = self.vert_pos

        blending_coeff = self.query_blending_coeff(self.vert_pos)

        self.model_data.body_shape_vert_dir = \
            blending_coeff.body_shape_vert_dir

        self.model_data.expr_shape_vert_dir = \
            blending_coeff.expr_shape_vert_dir

        self.model_data.pose_vert_dir = \
            blending_coeff.pose_vert_dir

        self.model_data.lbs_weight = blending_coeff.lbs_weight

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

        self.model_data = model_data_subdivision_result.model_data

        self.model_data.vert_pos = torch.nn.Parameter(
            self.model_data.vert_pos.to(dtype=torch.float64, copy=True))

        self.vert_pos = self.model_data.vert_pos

        return model_data_subdivision_result

    def forward(self) -> ModelData:
        return self.model_data

    def query_blending_coeff(
        self,
        vert_pos: torch.Tensor,  # [..., 3]
    ) -> DeoformingBlendingCoeff:
        BS = self.model_data.body_shapes_cnt
        ES = self.model_data.expr_shapes_cnt

        vert_pos = vert_pos.to(self.device)

        body_shape_vert_dir = \
            self.blending_coeff_field.query_body_shape_vert_dir(vert_pos)

        if body_shape_vert_dir is None:
            body_shape_vert_dir = utils.zeros_like(
                vert_pos, shape=(*vert_pos.shape[:-1], 3, BS))

        expr_shape_vert_dir = \
            self.blending_coeff_field.query_expr_shape_vert_dir(vert_pos)

        if expr_shape_vert_dir is None:
            expr_shape_vert_dir = utils.zeros_like(
                vert_pos, shape=(*vert_pos.shape[:-1], 3, ES))

        pose_vert_dir = expr_shape_vert_dir = \
            self.blending_coeff_field.query_pose_vert_dir(vert_pos)

        if pose_vert_dir is None:
            pose_vert_dir = utils.zeros_like(
                vert_pos, shape=(*vert_pos.shape[:-1], 3, 0))

        lbs_weight = expr_shape_vert_dir = \
            self.blending_coeff_field.query_lbs_weight(vert_pos)

        if lbs_weight is None:
            lbs_weight = utils.zeros_like(
                vert_pos, shape=(*vert_pos.shape[:-1], 0))

        return DeoformingBlendingCoeff(
            body_shape_vert_dir=body_shape_vert_dir,
            expr_shape_vert_dir=expr_shape_vert_dir,
            pose_vert_dir=pose_vert_dir,
            lbs_weight=lbs_weight,
        )

    def remesh(self, remesh_arg_pack: utils.ArgPack) -> DeformableModelBuilder:
        remeshed_model_data = self.model_data.remesh(remesh_arg_pack)

        remeshed_model_data.vert_pos = torch.nn.Parameter(
            remeshed_model_data.vert_pos.to(dtype=torch.float64, copy=True))

        self.model_data = remeshed_model_data

        self.vert_pos = remeshed_model_data.vert_pos

        return self
