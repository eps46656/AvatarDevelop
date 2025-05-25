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
        shape: tuple[int, ...] = (),

        kin_tree: typing.Optional[kin_utils.KinTree],

        mesh_graph: typing.Optional[mesh_utils.MeshGraph],
        tex_mesh_graph: typing.Optional[mesh_utils.MeshGraph],

        joint_T: typing.Optional[torch.Tensor],  # [..., J, 4, 4]

        vert_pos: typing.Optional[torch.Tensor],  # [..., V, 3]

        tex_vert_pos: typing.Optional[torch.Tensor],  # [..., TV, 2]

        vert_trans: typing.Optional[torch.Tensor],  # [..., V, 4, 4]
    ):
        J, V, TV = -1, -2, -3

        J, V, TV = utils.check_shapes(
            joint_T, (..., J, 4, 4),

            vert_pos, (..., V, 3),
            tex_vert_pos, (..., TV, 2),

            vert_trans, (..., V, 4, 4),

            set_zero_if_undet=False,
        )

        F = -1

        if kin_tree is not None:
            assert J < 0 or kin_tree.joints_cnt == J
            J = kin_tree.joints_cnt

        if mesh_graph is not None:
            assert V < 0 or mesh_graph.verts_cnt == V
            V = mesh_graph.verts_cnt

            assert F < 0 or mesh_graph.faces_cnt == F
            F = mesh_graph.faces_cnt

        if tex_mesh_graph is not None:
            assert TV < 0 or tex_mesh_graph.verts_cnt == TV
            TV = tex_mesh_graph.verts_cnt

            assert F < 0 or tex_mesh_graph.faces_cnt == F
            F = tex_mesh_graph.faces_cnt

        J = max(0, J)

        V = max(0, V)
        TV = max(0, TV)

        F = max(0, F)

        # ---

        self.shape = utils.broadcast_shapes(
            shape,
            utils.try_get_batch_shape(joint_T, -3),
            utils.try_get_batch_shape(vert_pos, -2),
            utils.try_get_batch_shape(tex_vert_pos, -2),
            utils.try_get_batch_shape(vert_trans, -3),
        )

        self.joints_cnt = J

        self.verts_cnt = V
        self.tex_verts_cnt = TV

        self.faces_cnt = F

        self.kin_tree = kin_tree

        self.mesh_graph = mesh_graph
        self.tex_mesh_graph = tex_mesh_graph

        self.joint_T = joint_T  # [..., J, 4, 4]

        self.vert_pos = vert_pos  # [..., V, 3]

        self.tex_vert_pos = tex_vert_pos  # [..., TV, 2]

        self.vert_trans = vert_trans  # [..., V, 4, 4]

        self.mesh_data = mesh_utils.MeshData(self.mesh_graph, self.vert_pos)

        self.tex_mesh_data = mesh_utils.MeshData(
            self.tex_mesh_graph, self.tex_vert_pos)

    def __getitem__(self, idx) -> AvatarModel:
        return AvatarModel(
            kin_tree=self.kin_tree,

            mesh_graph=self.mesh_graph,
            tex_mesh_graph=self.tex_mesh_graph,

            joint_T=utils.try_batch_indexing(
                self.joint_T, self.shape, 3, idx),

            vert_pos=utils.try_batch_indexing(
                self.vert_pos, self.shape, 2, idx),

            tex_vert_pos=utils.try_batch_indexing(
                self.tex_vert_pos, self.shape, 2, idx),

            vert_trans=utils.try_batch_indexing(
                self.vert_trans, self.shape, 3, idx),
        )


@beartype
class AvatarBlender(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_avatar_model() -> AvatarModel:
        raise NotImplementedError()

    def subdivide(
        self,
        *,
        target_faces: typing.Optional[typing.Iterable[int]] = None,
        target_edges: typing.Optional[typing.Iterable[int]] = None,
    ) -> mesh_utils.MeshSubdivideResult:
        raise NotImplementedError()

    def forward(self, blending_param) -> AvatarModel:
        raise NotImplementedError()
