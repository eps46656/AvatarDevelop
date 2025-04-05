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

        if vert_pos is not None:
            utils.check_shapes(vert_pos, (..., V, 3))

        if vert_nor is not None:
            utils.check_shapes(vert_nor, (..., V, 3))

        if tex_vert_pos is not None:
            utils.check_shapes(tex_vert_pos, (..., TV, 2))

        if joint_T is not None:
            utils.check_shapes(joint_T, (..., J, 4, 4))

        batch_shape = utils.broadcast_shapes(
            utils.try_get_batch_shape(vert_pos, -2),
            utils.try_get_batch_shape(vert_nor, -2),
            utils.try_get_batch_shape(tex_vert_pos, -2),
            utils.try_get_batch_shape(joint_T, -3),
        )

        vert_pos = utils.try_batch_expand(vert_pos, batch_shape, -2)
        vert_nor = utils.try_batch_expand(vert_nor, batch_shape, -2)
        tex_vert_pos = utils.try_batch_expand(tex_vert_pos, batch_shape, -2)
        joint_T = utils.try_batch_expand(joint_T, batch_shape, -3)

        # ---

        self.kin_tree = kin_tree

        self.mesh_data = mesh_data
        self.tex_mesh_data = tex_mesh_data

        self.vert_pos = vert_pos
        self.vert_nor = vert_nor

        self.tex_vert_pos = tex_vert_pos

        self.joint_T = joint_T

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

    @property
    def shape(self) -> torch.Size:
        return self.vert_pos.shape[:-2]

    def __getitem__(self, idx):
        vert_pos = utils.try_batch_index(self.vert_pos, -2, idx)
        vert_nor = utils.try_batch_index(self.vert_nor, -2, idx)
        tex_vert_pos = utils.try_batch_index(self.tex_vert_pos, -2, idx)
        joint_T = utils.try_batch_index(self.joint_T, -2, idx)

        return AvatarModel(
            kin_tree=self.kin_tree,

            mesh_data=self.mesh_data,
            tex_mesh_data=self.tex_mesh_data,

            vert_pos=vert_pos,
            vert_nor=vert_nor,

            tex_vert_pos=tex_vert_pos,

            joint_T=joint_T,
        )


@beartype
class AvatarBlender(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_avatar_model() -> AvatarModel:
        raise utils.UnimplementationError()

    def forward(self, blending_param) -> AvatarModel:
        raise utils.UnimplementationError()
