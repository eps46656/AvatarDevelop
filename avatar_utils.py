
import typing

import torch
from beartype import beartype

from . import kin_utils, mesh_utils, utils


@beartype
class AvatarModel:
    def GetKinTree(self) -> kin_utils.KinTree:
        raise utils.UnimplementationError()

    def GetVerticesCnt(self) -> int:  # V
        raise utils.UnimplementationError()

    def GetTextureVerticesCnt(self) -> int:  # TV
        raise utils.UnimplementationError()

    def GetFacesCnt(self) -> int:  # F
        raise utils.UnimplementationError()

    def GetJointsCnt(self) -> int:  # J
        raise utils.UnimplementationError()

    def GetVertexPositions(self) -> torch.Tensor:  # [..., V, 3]
        raise utils.UnimplementationError()

    def GetVertexNormals(self) -> torch.Tensor:  # [..., V, 3]
        raise utils.UnimplementationError()

    def GetTextureVertexPositions(self) -> torch.Tensor:  # [..., TV, 2]
        raise utils.UnimplementationError()

    def GetFaces(self) -> torch.Tensor:  # [..., F, 3]
        raise utils.UnimplementationError()

    def GetTextureFaces(self) -> torch.Tensor:  # [..., TF, 3]
        raise utils.UnimplementationError()

    def GetJointsTs(self) -> torch.Tensor:  # [..., J, 4, 4]
        raise utils.UnimplementationError()

    def GetMeshData(self) -> mesh_utils.MeshData:
        raise utils.UnimplementationError()


@beartype
class AvatarBlendingLayer(torch.nn.Module):
    def __init__(self):
        super(AvatarBlendingLayer, self).__init__()

    def GetKinTree(self) -> kin_utils.KinTree:
        raise utils.UnimplementationError()

    def GetVerticesCnt(self) -> int:  # V
        raise utils.UnimplementationError()

    def GetTextureVerticesCnt(self) -> int:  # TV
        raise utils.UnimplementationError()

    def GetFacesCnt(self) -> int:  # F
        raise utils.UnimplementationError()

    def GetjointsCnt(self) -> int:  # J
        raise utils.UnimplementationError()

    def GetVertexPositions(self) -> typing.Optional[torch.Tensor]:
        # [..., V, 3]
        raise utils.UnimplementationError()

    def GetVertexNormals(self) -> typing.Optional[torch.Tensor]:  # [..., V, 3]
        raise utils.UnimplementationError()

    def GetTextureVertexPositions(self) -> typing.Optional[torch.Tensor]:
        # [..., TV, 2]
        raise utils.UnimplementationError()

    def GetFaces(self) -> typing.Optional[torch.Tensor]:  # [..., F, 3]
        raise utils.UnimplementationError()

    def GetTextureFaces(self) -> typing.Optional[torch.Tensor]:  # [..., TF, 3]
        raise utils.UnimplementationError()

    def GetMeshData(self) -> typing.Optional[mesh_utils.MeshData]:
        raise utils.UnimplementationError()

    def forward(self, blending_param) -> AvatarModel:
        assert False, "Unoverridden abstract method."
