import dataclasses
import typing

import torch
from beartype import beartype

from .. import avatar_utils, kin_utils, mesh_utils, utils
from .blending_utils import Blending, BlendingParam
from .ModelData import ModelData


@beartype
class ModelBlender(avatar_utils.AvatarBlender):
    def _TryRegistParameter(self, x: typing.Optional[torch.Tensor]):
        if x is not None and isinstance(x, torch.nn.Parameter):
            self.register_parameter("vertex_positions", x)

    def __init__(
        self,
        model_data: ModelData,
        device: torch.device,
    ):
        super(ModelBlender, self).__init__()

        self.model_data = model_data

        self._TryRegistParameter(self.model_data.vertex_positions)

        # ---

        self.device = device

        # ---

        self.default_blending_param = BlendingParam(
            body_shapes=torch.zeros(
                (self.model_data.GetBodyShapesCnt(),), dtype=utils.FLOAT, device=self.device),

            expr_shapes=torch.zeros(
                (self.model_data.GetExprShapesCnt(),), dtype=utils.FLOAT, device=self.device),

            global_transl=torch.zeros(
                (3,), dtype=utils.FLOAT, device=self.device),

            global_rot=torch.zeros(
                (3,), dtype=utils.FLOAT, device=self.device),

            body_poses=torch.zeros(
                (self.model_data.body_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            jaw_poses=torch.zeros(
                (self.model_data.jaw_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            leye_poses=torch.zeros(
                (self.model_data.eye_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            reye_poses=torch.zeros(
                (self.model_data.eye_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            lhand_poses=torch.zeros(
                (self.model_data.hand_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            rhand_poses=torch.zeros(
                (self.model_data.hand_joints_cnt, 3), dtype=utils.FLOAT, device=self.device),

            blending_vertex_normal=False,
        )

    def GetKinTree(self) -> kin_utils.KinTree:
        return self.kin_tree

    def GetVerticesCnt(self) -> int:  # V
        return self.model_data.vertex_positions.shape[-2]

    def GetTextureVerticesCnt(self) -> int:  # TV
        return 0 if self.model_data.texture_vertex_positions is None else self.model_data.texture_vertex_positions.shape[-2]

    def GetFacesCnt(self) -> int:  # F
        return 0 if self.model_data.faces is None else self.model_data.faces.shape[-2]

    def GetjointsCnt(self) -> int:  # J
        return 0 if self.model_data.texture_faces is None else self.model_data.texture_faces.shape[-2]

    def GetVertexPositions(self) -> typing.Optional[torch.Tensor]:
        # [..., V, 3]
        return self.model_data.kin_tree.joints_cnt

    def GetVertexNormals(self) -> typing.Optional[torch.Tensor]:  # [..., V, 3]
        return self.model_data.body_shape_dirs.shape[-1]

    def GetTextureVertexPositions(self) -> typing.Optional[torch.Tensor]:
        # [..., TV, 2]
        return self.model_data.texture_vertex_positions

    def GetFaces(self) -> typing.Optional[torch.Tensor]:  # [..., F, 3]
        return self.model_data.faces

    def GetTextureFaces(self) -> typing.Optional[torch.Tensor]:  # [..., TF, 3]
        return self.model_data.texture_faces

    def GetMeshData(self) -> typing.Optional[mesh_utils.MeshData]:
        raise utils.UnimplementationError()

    def GetBodyShapesCnt(self):
        return 0 if self.model_data.body_shape_dirs is None else self.model_data.body_shape_dirs.shape[-1]

    def GetExprShapesCnt(self):
        return 0 if self.model_data.expr_shape_dirs is None else self.model_data.expr_shape_dirs.shape[-1]

    def SetDefaultBlendingParam(self, blending_param: BlendingParam):
        blending_param.Check(self.model_data, True)

        for field in dataclasses.fields(BlendingParam):
            value = getattr(blending_param, field)

            if value is not None:
                setattr(self.default_blending_param, field, value)

    def forward(self, blending_param: BlendingParam):
        return Blending(
            model_data=self.model_data,
            blending_param=blending_param.GetCombined(
                self.default_blending_param),
            device=self.device,
        )
