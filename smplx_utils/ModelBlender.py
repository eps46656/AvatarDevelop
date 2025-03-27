import dataclasses
import typing

import torch
from beartype import beartype

from .. import avatar_utils, utils
from .blending_utils import blending, BlendingParam
from .Model import Model
from .ModelBuilder import ModelBuilder


@beartype
class ModelBlender(avatar_utils.AvatarBlender):
    def __init__(
        self,
        model_builder: ModelBuilder,
    ):
        super(ModelBlender, self).__init__()

        self.model_builder = model_builder

        device = self.model_builder.device

        # ---

        model_config = model_builder.get_model_data().get_model_config()

        self.default_blending_param = BlendingParam(
            body_shapes=torch.zeros(
                (model_config.body_shapes_cnt,), dtype=utils.FLOAT, device=device),

            expr_shapes=torch.zeros(
                (model_config.expr_shapes_cnt,), dtype=utils.FLOAT, device=device),

            global_transl=torch.zeros(
                (3,), dtype=utils.FLOAT, device=device),

            global_rot=torch.zeros(
                (3,), dtype=utils.FLOAT, device=device),

            body_poses=torch.zeros(
                (model_config.body_joints_cnt, 3), dtype=utils.FLOAT, device=device),

            jaw_poses=torch.zeros(
                (model_config.jaw_joints_cnt, 3), dtype=utils.FLOAT, device=device),

            leye_poses=torch.zeros(
                (model_config.eye_joints_cnt, 3), dtype=utils.FLOAT, device=device),

            reye_poses=torch.zeros(
                (model_config.eye_joints_cnt, 3), dtype=utils.FLOAT, device=device),

            lhand_poses=torch.zeros(
                (model_config.hand_joints_cnt, 3), dtype=utils.FLOAT, device=device),

            rhand_poses=torch.zeros(
                (model_config.hand_joints_cnt, 3), dtype=utils.FLOAT, device=device),

            blending_vertex_normal=False,
        )

    def get_avatar_model(self) -> avatar_utils.AvatarModel:
        model_data = self.model_builder.get_model_data()

        kin_tree = model_data.kin_tree

        vertices_cnt = model_data.vertex_positions.shape[-2]

        texture_vertices_cnt = model_data.vertex_positions.shape[-2]

        faces_cnt = 0 if model_data.faces is None else model_data.faces.shape[-2]

        joints_cnt = model_data.kin_tree.joints_cnt

        vertex_positions = None

        vertex_normals = None

        texture_vertex_positions = model_data.texture_vertex_positions

        faces = model_data.faces

        texture_faces = model_data.texture_faces

        joint_Ts = None

        mesh_data = model_data.mesh_data

        return Model(
            kin_tree=kin_tree,
            vertices_cnt=vertices_cnt,
            texture_vertices_cnt=texture_vertices_cnt,
            faces_cnt=faces_cnt,
            joints_cnt=joints_cnt,
            vertex_positions=vertex_positions,
            vertex_normals=vertex_normals,
            texture_vertex_positions=texture_vertex_positions,
            faces=faces,
            texture_faces=texture_faces,
            joint_Ts=joint_Ts,
            mesh_data=mesh_data,
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

    def to(self, *args, **kwargs) -> typing.Self:
        self.model_builder.to(*args, **kwargs)

        self.default_blending_param = self.default_blending_param \
            .to(*args, **kwargs)

        return self

    def forward(self, blending_param: BlendingParam):
        return blending(
            model_data=self.model_builder.get_model_data(),
            blending_param=blending_param.combine(
                self.default_blending_param),
            device=self.model_builder.device,
        )
