
import dataclasses

import torch

import mesh_utils


@dataclasses.dataclass
class AvatarModel:
    vertices_cnt: int  # V
    texture_vertices_cnt: int  # VT

    faces_cnt: int  # F

    joints_cnt: int  # J

    vertex_positions: torch.Tensor  # [..., V, 3]
    vertex_normals: torch.Tensor  # [..., V, 3]

    texture_vertex_positions: torch.Tensor  # [..., VT, 2]

    faces: torch.Tensor  # [..., F, 3]
    texture_faces: torch.Tensor  # [..., F, 3]

    joints_rs: torch.Tensor  # [..., J, 3, 3]
    joints_ts: torch.Tensor  # [..., J, 3, 3]

    mesh_data: mesh_utils.MeshData


class AvatarBuilder(torch.nn.Model):
    def GetVerticesCnt(self) -> int:  # V
        assert False, "Unoverridden abstract method."

    def GetTextureVerticesCnt(self) -> torch.Tensor:  # VT
        assert False, "Unoverridden abstract method."

    def GetFacesCnt(self) -> int:  # F
        assert False, "Unoverridden abstract method."

    def GetJointsCnt(self) -> int:  # J
        assert False, "Unoverridden abstract method."

    def GetTextureVertexPositions(self) -> torch.Tensor:  # [VT, 2]
        assert False, "Unoverridden abstract method."

    def GetFaces(self) -> torch.Tensor:  # [..., F, 3]
        assert False, "Unoverridden abstract method."

    def GetTextureFaces(self) -> torch.Tensor:  # [..., F, 3]
        assert False, "Unoverridden abstract method."

    def GetMeshData(self) -> mesh_utils.MeshData:
        assert False, "Unoverridden abstract method."

    def forward(self, blending_param) -> AvatarModel:
        assert False, "Unoverridden abstract method."
