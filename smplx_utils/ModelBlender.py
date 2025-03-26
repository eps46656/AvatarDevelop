import dataclasses

import torch
from beartype import beartype

from .. import avatar_utils, utils
from .blending_utils import Blending, BlendingParam
from .Model import Model
from .ModelBuilder import ModelBuilder


@beartype
class ModelBlender(avatar_utils.AvatarBlender):
    def __init__(
        self,
        model_builder: ModelBuilder,
        device: torch.device,
    ):
        super(ModelBlender, self).__init__()

        self.model_builder = model_builder

        # ---

        self.device = device

        # ---

        model_config = model_builder.GetModelData().GetModelConfig()

        self.default_blending_param = BlendingParam(
            body_shapes=torch.zeros(
                (model_config.body_shapes_cnt,), dtype=utils.FLOAT, device=self.device),

            expr_shapes=torch.zeros(
                (model_config.expr_shapes_cnt,), dtype=utils.FLOAT, device=self.device),

            global_transl=torch.zeros(
                (3,), dtype=utils.FLOAT, device=self.device),

            global_rot=torch.zeros(
                (3,), dtype=utils.FLOAT, device=self.device),

            body_poses=torch.zeros(
                (model_config.body_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            jaw_poses=torch.zeros(
                (model_config.jaw_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            leye_poses=torch.zeros(
                (model_config.eye_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            reye_poses=torch.zeros(
                (model_config.eye_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            lhand_poses=torch.zeros(
                (model_config.hand_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            rhand_poses=torch.zeros(
                (model_config.hand_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            blending_vertex_normal=False,
        )

    def GetAvatarModel(self) -> avatar_utils.AvatarModel:
        model_data = self.model_builder.GetModelData()

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

    def GetBodyShapesCnt(self):
        return self.model_builder.GetModelData().GetBodyShapesCnt()

    def GetExprShapesCnt(self):
        return self.model_builder.GetModelData().GetExprShapesCnt()

    def SetDefaultBlendingParam(self, blending_param: BlendingParam):
        blending_param.Check(self.model_data, True)

        for field in dataclasses.fields(BlendingParam):
            value = getattr(blending_param, field)

            if value is not None:
                setattr(self.default_blending_param, field, value)

    def forward(self, blending_param: BlendingParam):
        return Blending(
            model_data=self.model_builder.GetModelData(),
            blending_param=blending_param.GetCombined(
                self.default_blending_param),
            device=self.device,
        )
