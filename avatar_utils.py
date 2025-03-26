
import dataclasses
import typing

import torch
from beartype import beartype

from . import kin_utils, mesh_utils, utils


@beartype
@dataclasses.dataclass
class AvatarModel:
    kin_tree: kin_utils.KinTree

    vertices_cnt: int  # V
    texture_vertices_cnt: int  # TV

    faces_cnt: int  # F
    joints_cnt: int  # J

    vertex_positions: typing.Optional[torch.Tensor]  # [..., V, 3]
    vertex_normals: typing.Optional[torch.Tensor]  # [..., V, 3]

    texture_vertex_positions: typing.Optional[torch.Tensor]  # [..., TV, 2]

    faces: torch.Tensor  # [..., F, 3]

    texture_faces: typing.Optional[torch.Tensor]  # [..., TF, 3]

    joint_Ts: typing.Optional[torch.Tensor]  # [..., J, 4, 4]

    mesh_data: mesh_utils.MeshData


@beartype
class AvatarBlender(torch.nn.Module):
    def __init__(self):
        super(AvatarBlender, self).__init__()

    def GetAvatarModel() -> AvatarModel:
        raise utils.UnimplementationError()

    def forward(self, blending_param) -> AvatarModel:
        assert False, "Unoverridden abstract method."
