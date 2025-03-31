
import typing

import torch
from beartype import beartype

from . import kin_utils, mesh_utils, utils


@beartype
class AvatarModel:
    def __init__(
        self,
        *,

        kin_tree: kin_utils.KinTree,

        vertices_cnt: int,  # V
        texture_vertices_cnt: int,  # TV

        vertex_positions: typing.Optional[torch.Tensor] = None,  # [..., V, 3]
        vertex_normals: typing.Optional[torch.Tensor] = None,  # [..., V, 3]
        vertex_rotations: typing.Optional[torch.Tensor] = None,  # [..., V, 4]

        texture_vertex_positions: typing.Optional[torch.Tensor] = None,
        # [..., TV, 2]

        faces: torch.Tensor,  # [..., F, 3]

        texture_faces: typing.Optional[torch.Tensor] = None,  # [..., F, 3]

        joint_Ts: typing.Optional[torch.Tensor] = None,  # [..., J, 4, 4]

        mesh_data: mesh_utils.MeshData,
    ):
        J = kin_tree.joints_cnt

        assert 0 <= vertices_cnt
        assert 0 <= texture_vertices_cnt

        V = vertices_cnt
        TV = texture_vertices_cnt

        if vertex_positions is not None:
            utils.check_shapes(vertex_positions, (..., V, 3))

        if vertex_normals is not None:
            utils.check_shapes(vertex_normals, (..., V, 3))

        if vertex_rotations is not None:
            utils.check_shapes(vertex_rotations, (..., V, 4))

        if texture_vertex_positions is not None:
            utils.check_shapes(texture_vertex_positions, (..., TV, 2))

        F = utils.check_shapes(faces, (..., -1, 3))

        if texture_faces is not None:
            utils.check_shapes(texture_faces, (..., F, 3))

        if joint_Ts is not None:
            utils.check_shapes(joint_Ts, (..., J, 4, 4))

        assert mesh_data.vertices_cnt == V
        assert mesh_data.faces_cnt == F

        # ---

        self.kin_tree = kin_tree

        self.vertices_cnt = vertices_cnt
        self.texture_vertices_cnt = texture_vertices_cnt

        self.vertex_positions = vertex_positions
        self.vertex_normals = vertex_normals

        self.texture_vertex_positions = texture_vertex_positions

        self.faces = faces

        self.texture_faces = texture_faces

        self.joint_Ts = joint_Ts

        self.mesh_data = mesh_data

    @property
    def faces_cnt(self) -> int:
        return self.faces.shape[-2]

    @property
    def joints_cnt(self) -> int:
        return self.kin_tree.joints_cnt

    @property
    def shape(self) -> torch.Size:
        batch_dim_table = {
            "vertex_positions": -2,
            "vertex_normals": -2,
            "texture_vertex_positions": -2,
            "faces": -2,
            "texture_faces": -2,
            "joint_Ts": -3,
        }

        l = list()

        for key, val in batch_dim_table.items():
            field_val = getattr(self, key)

            if field_val is None:
                continue

            l.append(field_val.shape[:val])

        return utils.broadcast_shapes(*l)


@beartype
class AvatarBlender(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_avatar_model() -> AvatarModel:
        raise utils.UnimplementationError()

    def forward(self, blending_param) -> AvatarModel:
        raise utils.UnimplementationError()
