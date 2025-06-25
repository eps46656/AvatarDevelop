from __future__ import annotations

import collections
import copy
import dataclasses
import typing

import torch
from beartype import beartype

from .. import rbf_utils, utils, kernel_splatting_utils
from .ModelData import ModelData, ModelDataSubdivideResult


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
    ) -> ModelDataSubdivideResult:
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
class LBSWeightInterpKernel:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (1e-8 + x).pow(-3)


@beartype
@dataclasses.dataclass
class BlendingCoeff:
    body_shape_vert_dir: typing.Optional[torch.Tensor]  # [..., 3, BS]
    expr_shape_vert_dir: typing.Optional[torch.Tensor]  # [..., 3, BS]
    pose_vert_dir: typing.Optional[torch.Tensor]  # [..., 3, (J - 1) * D * D]
    lbs_weight: typing.Optional[torch.Tensor]  # [..., J]


@beartype
class BlendingCoeffField:
    def __init__(
        self,
        device: torch.device,
        body_shape_vert_dir_interp: typing.Optional[rbf_utils.RBFInterpolator],
        expr_shape_vert_dir_interp: typing.Optional[rbf_utils.RBFInterpolator],
        pose_vert_dir_interp: typing.Optional[rbf_utils.RBFInterpolator],
        lbs_weight_interp: typing.Optional[
            kernel_splatting_utils.KernelSplattingInterpolator],
    ):
        self.device = device
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
    ) -> BlendingCoeffField:
        V = utils.check_shapes(vert_pos, (-1, 3))

        kernel = rbf_utils.radial_func.WendlandRadialFunc(50e-3)

        def _f(x):
            return None if x is None else \
                rbf_utils.RBFInterpolator.from_data_point(
                    interior=False,

                    d_pos=vert_pos,  # [V, 3]

                    d_val=x.reshape(V, -1),

                    kernel=kernel,

                    degree=3,
                    smoothness=0.0,
                )

        body_shape_vert_dir_interp = _f(body_shape_vert_dir)
        expr_shape_vert_dir_interp = _f(expr_shape_vert_dir)
        pose_vert_dir_interp = _f(pose_vert_dir)
        lbs_weight_interp = _f(lbs_weight)

        lbs_weight_interp = kernel_splatting_utils.KernelSplattingInterpolator.from_data_point(
            d_pos=vert_pos,  # [V, 3]

            d_val=lbs_weight.reshape(V, -1),

            kernel=LBSWeightInterpKernel(),
        )

        return BlendingCoeffField(
            device=vert_pos.device,
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
    def from_state_dict(
        state_dict: typing.Mapping[str, object],
        device: torch.device,
    ) -> BlendingCoeffField:
        body_shape_vert_dir_interp_state_dict = state_dict.get(
            "body_shape_vert_dir_interp", None)

        expr_shape_vert_dir_interp_state_dict = state_dict.get(
            "expr_shape_vert_dir_interp", None)

        pose_vert_dir_interp_state_dict = state_dict.get(
            "pose_vert_dir_interp", None)

        lbs_weight_interp_state_dict = state_dict.get(
            "lbs_weight_interp", None)

        return BlendingCoeffField(
            device,

            body_shape_vert_dir_interp=None
            if body_shape_vert_dir_interp_state_dict is None else
            rbf_utils.RBFInterpolator.from_state_dict(
                body_shape_vert_dir_interp_state_dict, device),

            expr_shape_vert_dir_interp=None
            if expr_shape_vert_dir_interp_state_dict is None else
            rbf_utils.RBFInterpolator.from_state_dict(
                expr_shape_vert_dir_interp_state_dict, device),

            pose_vert_dir_interp=None
            if pose_vert_dir_interp_state_dict is None else
            rbf_utils.RBFInterpolator.from_state_dict(
                pose_vert_dir_interp_state_dict, device),

            lbs_weight_interp=None
            if lbs_weight_interp_state_dict is None else
            kernel_splatting_utils.KernelSplattingInterpolator.from_state_dict(
                lbs_weight_interp_state_dict, device),
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
        body_shape_vert_dir_interp_state_dict = state_dict.get(
            "body_shape_vert_dir_interp", None)

        expr_shape_vert_dir_interp_state_dict = state_dict.get(
            "expr_shape_vert_dir_interp", None)

        pose_vert_dir_interp_state_dict = state_dict.get(
            "pose_vert_dir_interp", None)

        lbs_weight_interp_state_dict = state_dict.get(
            "lbs_weight_interp", None)

        self.body_shape_vert_dir_interp = None \
            if body_shape_vert_dir_interp_state_dict is None else \
            rbf_utils.RBFInterpolator.from_state_dict(
                body_shape_vert_dir_interp_state_dict, self.device)

        self.expr_shape_vert_dir_interp = None \
            if expr_shape_vert_dir_interp_state_dict is None else \
            rbf_utils.RBFInterpolator.from_state_dict(
                expr_shape_vert_dir_interp_state_dict, self.device)

        self.pose_vert_dir_interp = None \
            if pose_vert_dir_interp_state_dict is None else \
            rbf_utils.RBFInterpolator.from_state_dict(
                pose_vert_dir_interp_state_dict, self.device)

        self.lbs_weight_interp = None \
            if lbs_weight_interp_state_dict is None else \
            kernel_splatting_utils.KernelSplattingInterpolator.from_state_dict(
                state_dict["lbs_weight_interp"], self.device)

    def __query_body_shape_vert_dir(
        self,
        vert_pos: torch.Tensor,  # [..., D]
        vert_kernel_dist: typing.Optional[torch.Tensor] = None,  # [..., V]
        vert_poly_val: typing.Optional[torch.Tensor] = None,  # [..., V]
    ) -> tuple[
        typing.Optional[torch.Tensor],  # body_shape_vert_dir[..., 3, BS]
        typing.Optional[torch.Tensor],  # vert_kernel_dist[..., V]
        typing.Optional[torch.Tensor],  # vert_poly_val[..., P]
    ]:
        if self.body_shape_vert_dir_interp is None or vert_pos is None:
            return None, vert_kernel_dist, vert_poly_val

        body_shape_vert_dir, vert_kernel_dist, vert_poly_val = \
            self.body_shape_vert_dir_interp(
                q_pos=vert_pos,
                q_kernel_dist=vert_kernel_dist,
                q_poly_val=vert_poly_val,
            )
        # body_shape_vert_dir[..., 3 * BS]
        # vert_kernel_dist[..., V]
        # vert_poly_val[..., P]

        body_shape_vert_dir = body_shape_vert_dir.view(
            *body_shape_vert_dir.shape[:-1],
            3,
            body_shape_vert_dir.shape[-1] // 3,
        )

        return body_shape_vert_dir, vert_kernel_dist, vert_poly_val

    def query_body_shape_vert_dir(
        self,
        vert_pos: torch.Tensor,  # [..., D]
    ) -> typing.Optional[torch.Tensor]:  # [..., 3, BS]
        return self.__query_body_shape_vert_dir(vert_pos)[0]

    def __query_expr_shape_vert_dir(
        self,
        vert_pos: torch.Tensor,  # [..., D]
        vert_kernel_dist: typing.Optional[torch.Tensor] = None,  # [..., V]
        vert_poly_val: typing.Optional[torch.Tensor] = None,  # [..., V]
    ) -> tuple[
        typing.Optional[torch.Tensor],  # expr_shape_vert_dir[..., 3, BS]
        typing.Optional[torch.Tensor],  # vert_kernel_dist[..., V]
        typing.Optional[torch.Tensor],  # vert_poly_val[..., P]
    ]:
        if self.expr_shape_vert_dir_interp is None or vert_pos is None:
            return None, vert_kernel_dist, vert_poly_val

        expr_shape_vert_dir, vert_kernel_dist, vert_poly_val = \
            self.expr_shape_vert_dir_interp(
                q_pos=vert_pos,
                q_kernel_dist=vert_kernel_dist,
                q_poly_val=vert_poly_val,
            )
        # expr_shape_vert_dir[..., 3 * BS]
        # vert_kernel_dist[..., V]
        # vert_poly_val[..., P]

        expr_shape_vert_dir = expr_shape_vert_dir.view(
            *expr_shape_vert_dir.shape[:-1],
            3,
            expr_shape_vert_dir.shape[-1] // 3,
        )

        return expr_shape_vert_dir, vert_kernel_dist, vert_poly_val

    def query_expr_shape_vert_dir(
        self,
        vert_pos: torch.Tensor,  # [..., D]
    ) -> typing.Optional[torch.Tensor]:  # [..., 3, BS]
        return self.__query_expr_shape_vert_dir(vert_pos)[0]

    def __query_pose_vert_dir(
        self,
        vert_pos: torch.Tensor,
        vert_kernel_dist: typing.Optional[torch.Tensor] = None,  # [..., V]
        vert_poly_val: typing.Optional[torch.Tensor] = None,  # [..., V]
    ) -> tuple[
        typing.Optional[torch.Tensor],
        # pose_vert_dir[..., 3, (J - 1) * D * D]

        typing.Optional[torch.Tensor],  # vert_kernel_dist[..., V]
        typing.Optional[torch.Tensor],  # vert_poly_val[..., P]
    ]:
        if self.pose_vert_dir_interp is None or vert_pos is None:
            return None, vert_kernel_dist, vert_poly_val

        pose_vert_dir, vert_kernel_dist, vert_poly_val = \
            self.pose_vert_dir_interp(
                q_pos=vert_pos,
                q_kernel_dist=vert_kernel_dist,
                q_poly_val=vert_poly_val,
            )
        # [..., 3 * (J - 1) * D * D]

        pose_vert_dir = pose_vert_dir.view(
            *pose_vert_dir.shape[:-1],
            3,
            pose_vert_dir.shape[-1] // 3,
        )
        # [..., 3, (J - 1) * D * D]

        return pose_vert_dir, vert_kernel_dist, vert_poly_val

    def query_pose_vert_dir(
        self,
        vert_pos: torch.Tensor,
    ) -> typing.Optional[torch.Tensor]:  # [..., 3, (J - 1) * D * D]
        return self.__query_pose_vert_dir(vert_pos)[0]

    def __query_lbs_weight(
        self,
        vert_pos: torch.Tensor,  # [..., D]
    ) -> typing.Optional[torch.Tensor]:  # lbs_weight[..., J]
        if self.lbs_weight_interp is None or vert_pos is None:
            return None

        lbs_weight = self.lbs_weight_interp(vert_pos)
        # [..., J]

        return lbs_weight

    def query_lbs_weight(
        self,
        vert_pos: torch.Tensor,  # [..., D]
    ) -> typing.Optional[torch.Tensor]:  # lbs_weight[..., J]
        return self.__query_lbs_weight(vert_pos)

    def query_blending_coeff(
        self,
        vert_pos: torch.Tensor,  # [..., D]
    ) -> BlendingCoeff:
        vert_kernel_dist = None
        vert_poly_val = None

        body_shape_vert_dir, vert_kernel_dist, vert_poly_val = \
            self.__query_body_shape_vert_dir(
                vert_pos, vert_kernel_dist, vert_poly_val)

        expr_shape_vert_dir, vert_kernel_dist, vert_poly_val = \
            self.__query_expr_shape_vert_dir(
                vert_pos, vert_kernel_dist, vert_poly_val)

        pose_vert_dir, vert_kernel_dist, vert_poly_val = \
            self.__query_pose_vert_dir(
                vert_pos, vert_kernel_dist, vert_poly_val)

        lbs_weight = self.__query_lbs_weight(vert_pos)

        return BlendingCoeff(
            body_shape_vert_dir=body_shape_vert_dir,
            # [..., 3, BS]

            expr_shape_vert_dir=expr_shape_vert_dir,
            # [..., 3, ES]

            pose_vert_dir=pose_vert_dir,
            # [..., 3, (J - 1) * D * D]

            lbs_weight=lbs_weight,
            # [..., J]
        )

    def query_model_data(self, model_data: ModelData) -> ModelData:
        blending_coeff = self.query_blending_coeff(
            model_data.vert_pos,  # [V, 3]
        )

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

            body_shape_joint_dir=model_data.body_shape_joint_dir,
            expr_shape_joint_dir=model_data.expr_shape_joint_dir,

            body_shape_vert_dir=utils.fallback_if_none(
                blending_coeff.body_shape_vert_dir,
                model_data.body_shape_vert_dir,
            ),

            expr_shape_vert_dir=utils.fallback_if_none(
                blending_coeff.expr_shape_vert_dir,
                model_data.expr_shape_vert_dir,
            ),

            lhand_pose_mean=model_data.lhand_pose_mean,
            rhand_pose_mean=model_data.rhand_pose_mean,

            pose_vert_dir=utils.fallback_if_none(
                blending_coeff.pose_vert_dir,
                model_data.pose_vert_dir,
            ),

            lbs_weight=utils.fallback_if_none(
                blending_coeff.lbs_weight,
                model_data.lbs_weight,
            ),  # [V, J]
        )


@beartype
class DeformableModelBuilder(ModelBuilder):
    def __init__(
        self,
        temp_model_data: ModelData,
        model_data: typing.Optional[ModelData],
    ):
        super().__init__()

        if model_data is not None:
            assert temp_model_data.body_shapes_cnt == \
                model_data.body_shapes_cnt

            assert temp_model_data.expr_shapes_cnt == \
                model_data.expr_shapes_cnt

            assert temp_model_data.body_joints_cnt == \
                model_data.body_joints_cnt

            assert temp_model_data.jaw_joints_cnt == \
                model_data.jaw_joints_cnt

            assert temp_model_data.eye_joints_cnt == \
                model_data.eye_joints_cnt

            assert temp_model_data.hand_joints_cnt == \
                model_data.hand_joints_cnt

            assert temp_model_data.kin_tree == \
                model_data.kin_tree

        self.temp_model_data = temp_model_data
        self.model_data = copy.copy(
            temp_model_data if model_data is None else model_data)

        self.model_data.vert_pos = torch.nn.Parameter(
            self.model_data.vert_pos
            .detach()
            .to(dtype=torch.float64, copy=True)
            .requires_grad_()
        )

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
                "lr": min(1e-4, base_lr),
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

        self.model_data.vert_pos = torch.nn.Parameter(
            self.model_data.vert_pos
            .detach()
            .to(dtype=torch.float64, copy=True)
            .requires_grad_()
        )

        self.vert_pos = self.model_data.vert_pos

        self.blending_coeff_field.load_state_dict(
            state_dict["blending_coeff_field"])

    def refresh(self) -> None:
        blending_coeff = self.blending_coeff_field.query_blending_coeff(
            self.vert_pos)

        self.model_data.body_shape_vert_dir = blending_coeff.body_shape_vert_dir

        self.model_data.expr_shape_vert_dir = blending_coeff.expr_shape_vert_dir

        self.model_data.pose_vert_dir = blending_coeff.pose_vert_dir

        self.model_data.lbs_weight = blending_coeff.lbs_weight

    def subdivide(
        self,
        *,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> ModelDataSubdivideResult:
        model_data_subdivide_result = self.model_data.subdivide(
            target_edges=target_edges,
            target_faces=target_faces,
        )

        self.model_data = model_data_subdivide_result.model_data

        self.model_data.vert_pos = torch.nn.Parameter(
            self.model_data.vert_pos.to(dtype=torch.float64, copy=True))

        self.vert_pos = self.model_data.vert_pos

        return model_data_subdivide_result

    def forward(self) -> ModelData:
        return self.model_data

    def remesh(self, remesh_arg_pack: utils.ArgPack) -> DeformableModelBuilder:
        remeshed_model_data = self.model_data.remesh(remesh_arg_pack)

        remeshed_model_data.vert_pos = torch.nn.Parameter(
            remeshed_model_data.vert_pos.to(dtype=torch.float64, copy=True))

        self.model_data = remeshed_model_data

        self.vert_pos = remeshed_model_data.vert_pos

        return self
