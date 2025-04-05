from __future__ import annotations

import dataclasses

import torch
from beartype import beartype

from .. import avatar_utils, utils
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
            body_shapes=torch.zeros(
                (model_data.body_shapes_cnt,), dtype=utils.FLOAT, device=device),

            expr_shapes=torch.zeros(
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

            blending_vert_nor=False,
        )

    def get_avatar_model(self) -> avatar_utils.AvatarModel:
        model_data = self.model_builder.get_model_data()

        return Model(
            kin_tree=model_data.kin_tree,

            mesh_data=model_data.mesh_data,
            tex_mesh_data=model_data.tex_mesh_data,

            vert_pos=model_data.vert_pos,
            vert_nor=None,

            tex_vert_pos=model_data.tex_vert_pos,

            joint_T=None,
        )

    @property
    def body_shapes_cnt(self):
        return self.model_builder.get_model_data().body_shapes_cnt

    @property
    def expr_shapes_cnt(self):
        return self.model_builder.get_model_data().expr_shapes_cnt

    def set_default_blending_param(self, blending_param: BlendingParam):
        blending_param.check(self.model_data, True)

        for field in dataclasses.fields(BlendingParam):
            value = getattr(blending_param, field)

            if value is not None:
                setattr(self.default_blending_param, field, value)

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

    def forward(self, blending_param: BlendingParam):
        return blending(
            model_data=self.model_builder.get_model_data(),
            blending_param=blending_param.combine(
                self.default_blending_param),
            device=self.model_builder.device,
        )
