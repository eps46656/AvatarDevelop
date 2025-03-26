import dataclasses
import typing

import torch
from beartype import beartype

from .. import kin_utils, mesh_utils


@beartype
@dataclasses.dataclass
class Model:
    kin_tree: kin_utils.KinTree

    joint_Ts: torch.Tensor  # [..., J, 4, 4]

    vertex_positions: torch.Tensor  # [..., V, 3]
    vertex_normals: torch.Tensor  # [..., V, 3]

    texture_vertex_positions: typing.Optional[torch.Tensor]  # [..., TV, 2]

    faces: typing.Optional[torch.Tensor]  # [F, 3]
    texture_faces: typing.Optional[torch.Tensor]  # [F, 3]

    mesh_data: mesh_utils.MeshData

    def GetKinTree(self) -> kin_utils.KinTree:
        return self.kin_tree

    def GetVerticesCnt(self) -> int:  # V
        return self.vertex_positions.shape[-2]

    def GetTextureVerticesCnt(self) -> int:  # TV
        return self.texture_vertex_positions.shape[-2]

    def GetFacesCnt(self) -> int:  # F
        return self.faces.shape[-2]

    def GetJointsCnt(self) -> int:
        return self.kin_tree.joints_cnt

    def GetVertexPositions(self) -> torch.Tensor:  # [..., V, 3]
        return self.vertex_positions

    def GetVertexNormals(self) -> torch.Tensor:  # [..., V, 3]
        return self.vertex_normals

    def GetTextureVertexPositions(self) -> torch.Tensor:  # [..., TV, 2]
        return self.texture_vertex_positions

    def GetFaces(self) -> torch.Tensor:  # [..., F, 3]
        return self.faces

    def GetTextureFaces(self) -> torch.Tensor:  # [..., TF, 3]
        return self.texture_faces

    def GetJointsTs(self) -> torch.Tensor:  # [..., J, 4, 4]
        return self.joint_Ts

    def GetMeshData(self) -> mesh_utils.MeshData:
        return self.mesh_data
