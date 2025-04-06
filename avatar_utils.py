from __future__ import annotations

import typing

import torch
from beartype import beartype

from . import kin_utils, mesh_utils, utils


@beartype
class AvatarModel:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = torch.Size(),

        kin_tree: kin_utils.KinTree,

        mesh_data: mesh_utils.MeshData,
        tex_mesh_data: typing.Optional[mesh_utils.MeshData] = None,

        vert_pos: typing.Optional[torch.Tensor] = None,  # [..., V, 3]
        vert_nor: typing.Optional[torch.Tensor] = None,  # [..., V, 3]

        tex_vert_pos: typing.Optional[torch.Tensor] = None,
        # [..., TV, 2]

        joint_T: typing.Optional[torch.Tensor] = None,  # [..., J, 4, 4]
    ):
        J = kin_tree.joints_cnt

        V = mesh_data.verts_cnt

        TV = 0 if tex_mesh_data is None else tex_mesh_data.verts_cnt

        F = mesh_data.faces_cnt
        assert tex_mesh_data.faces_cnt == F

        utils.check_shapes(
            vert_pos, (..., V, 3),
            vert_nor, (..., V, 3),
            tex_vert_pos, (..., TV, 2),
            joint_T, (..., J, 4, 4),
        )

        # ---

        self.shape = utils.broadcast_shapes(
            shape,
            utils.try_get_batch_shape(vert_pos, -2),
            utils.try_get_batch_shape(vert_nor, -2),
            utils.try_get_batch_shape(tex_vert_pos, -2),
            utils.try_get_batch_shape(joint_T, -3),
        )

        self.kin_tree = kin_tree

        self.mesh_data = mesh_data
        self.tex_mesh_data = tex_mesh_data

        self.vert_pos = vert_pos  # [..., V, 3]
        self.vert_nor = vert_nor  # [..., V, 3]

        self.tex_vert_pos = tex_vert_pos  # [..., TV, 2]

        self.joint_T = joint_T  # [..., J, 4, 4]

    @property
    def joints_cnt(self) -> int:
        return self.kin_tree.joints_cnt

    @property
    def verts_cnt(self) -> int:
        return self.mesh_data.verts_cnt

    @property
    def tex_verts_cnt(self) -> int:
        return self.tex_mesh_data.verts_cnt

    @property
    def faces_cnt(self) -> int:
        return self.mesh_data.faces_cnt

    def expand(self, shape: tuple[int, ...]) -> AvatarModel:
        return AvatarModel(
            shape=shape,

            mesh_data=self.mesh_data,
            tex_mesh_data=self.tex_mesh_data,

            vert_pos=self.vert_pos,

            vert_nor=self.vert_nor,

            tex_vert_pos=self.tex_vert_pos,

            joint_T=self.joint_T,
        )

    def __getitem__(self, idx) -> AvatarModel:
        return AvatarModel(
            kin_tree=self.kin_tree,

            mesh_data=self.mesh_data,
            tex_mesh_data=self.tex_mesh_data,

            vert_pos=utils.try_batch_indexing(
                self.vert_pos, self.shape, -2, idx),

            vert_nor=utils.try_batch_indexing(
                self.vert_nor, self.shape, -2, idx),

            tex_vert_pos=utils.try_batch_indexing(
                self.tex_vert_pos, self.shape, -2, idx),

            joint_T=utils.try_batch_indexing(
                self.joint_T, self.shape, -3, idx),
        )


@beartype
class AvatarBlender(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_avatar_model() -> AvatarModel:
        raise utils.UnimplementationError()

    def forward(self, blending_param) -> AvatarModel:
        raise utils.UnimplementationError()
