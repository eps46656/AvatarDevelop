from __future__ import annotations

import collections
import dataclasses
import typing

import torch
from beartype import beartype

from .. import avatar_utils, mesh_utils, utils
from .blending_utils import BlendingParam, blending
from .Model import Model
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

        self.default_blending_param = BlendingParam(
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
        return self(self.default_blending_param)

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

        self.default_blending_param = self.default_blending_param \
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
        return blending(
            model_data=self.model_builder.get_model_data(),
            blending_param=blending_param.combine(self.default_blending_param),
            device=self.model_builder.device,
        )

    def forward2(self, blending_param: BlendingParam):
        pass

    def refresh(self) -> None:
        if hasattr(self.model_builder, "refresh"):
            self.model_builder.refresh()
