from __future__ import annotations

import collections
import typing

import torch
from beartype import beartype

from .. import avatar_utils, mesh_utils, utils
from .blending_utils import BlendingParam, blending
from .ModelBuilder import ModelBuilder


@beartype
class ModelBlender(avatar_utils.AvatarBlender):
    def __init__(
        self,
        model_builder: ModelBuilder,
    ):
        super().__init__()

        self.model_builder = model_builder

        device = self.model_builder.device

        # ---

        model_data = model_builder.get_model_data()

        self.canonical_blending_param = BlendingParam(
            body_shape=torch.zeros(
                (model_data.body_shapes_cnt,), dtype=utils.FLOAT, device=device),

            expr_shape=torch.zeros(
                (model_data.expr_shapes_cnt,), dtype=utils.FLOAT, device=device),

            global_transl=torch.zeros(
                (3,), dtype=utils.FLOAT, device=device),

            global_rot=torch.zeros(
                (3,), dtype=utils.FLOAT, device=device),

            body_pose=torch.zeros(
                (model_data.body_joints_cnt - 1, 3), dtype=utils.FLOAT, device=device),

            jaw_pose=torch.zeros(
                (model_data.jaw_joints_cnt, 3), dtype=utils.FLOAT, device=device),

            leye_pose=torch.zeros(
                (model_data.eye_joints_cnt, 3), dtype=utils.FLOAT, device=device),

            reye_pose=torch.zeros(
                (model_data.eye_joints_cnt, 3), dtype=utils.FLOAT, device=device),

            lhand_pose=torch.zeros(
                (model_data.hand_joints_cnt, 3), dtype=utils.FLOAT, device=device),

            rhand_pose=torch.zeros(
                (model_data.hand_joints_cnt, 3), dtype=utils.FLOAT, device=device),
        )

    def get_avatar_model(self) -> avatar_utils.AvatarModel:
        return self(self.canonical_blending_param)

    @property
    def body_shapes_cnt(self):
        return self.model_builder.get_model_data().body_shapes_cnt

    @property
    def expr_shapes_cnt(self):
        return self.model_builder.get_model_data().expr_shapes_cnt

    @property
    def device(self):
        return self.model_builder.device

    def to(self, *args, **kwargs) -> ModelBlender:
        self.model_builder.to(*args, **kwargs)

        self.canonical_blending_param = self.canonical_blending_param \
            .to(*args, **kwargs)

        return self

    def get_param_groups(self, base_lr: float) -> list[dict]:
        return utils.get_param_groups(self.model_builder, base_lr)

    def state_dict(self) -> collections.OrderedDict:
        return collections.OrderedDict([
            ("model_builder", self.model_builder.state_dict())])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        self.model_builder.load_state_dict(state_dict["model_builder"])

    def subdivide(
        self,
        *,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
    ) -> mesh_utils.MeshSubdivisionResult:
        model_data_subdivision_result = self.model_builder.subdivide(
            target_edges=target_edges,
            target_faces=target_faces,
        )

        return model_data_subdivision_result.mesh_subdivision_result

    def forward(self, blending_param: BlendingParam):
        def f(x, y):
            return y if x is None else x

        a = blending_param
        b = self.canonical_blending_param

        return blending(
            model_data=self.model_builder.get_model_data(),

            blending_param=BlendingParam(
                shape=a.shape,

                body_shape=f(a.body_shape, b.body_shape),
                expr_shape=f(a.expr_shape, b.expr_shape),

                global_transl=f(a.global_transl, b.global_transl),
                global_rot=f(a.global_rot, b.global_rot),

                body_pose=f(a.body_pose, b.body_pose),
                jaw_pose=f(a.jaw_pose, b.jaw_pose),

                leye_pose=f(a.leye_pose, b.leye_pose),
                reye_pose=f(a.reye_pose, b.reye_pose),

                lhand_pose=f(a.lhand_pose, b.lhand_pose),
                rhand_pose=f(a.rhand_pose, b.rhand_pose),
            ),

            device=self.model_builder.device,
        )

    def refresh(self) -> None:
        if hasattr(self.model_builder, "refresh"):
            self.model_builder.refresh()
