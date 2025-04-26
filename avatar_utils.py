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

        mesh_graph: mesh_utils.MeshGraph,
        tex_mesh_graph: mesh_utils.MeshGraph,

        joint_T: torch.Tensor,  # [..., J, 4, 4]

        vert_pos: torch.Tensor,  # [..., V, 3]

        tex_vert_pos: torch.Tensor,  # [..., TV, 2]

        vert_trans: torch.Tensor,  # [..., V, 4, 4]
    ):
        J = kin_tree.joints_cnt

        V, F = mesh_graph.verts_cnt, mesh_graph.faces_cnt
        TV, TF = tex_mesh_graph.verts_cnt, tex_mesh_graph.faces_cnt

        assert F == 0 or TF == 0 or F == TF

        utils.check_shapes(
            joint_T, (..., J, 4, 4),

            vert_pos, (..., V, 3),
            tex_vert_pos, (..., TV, 2),

            vert_trans, (..., V, 4, 4),
        )

        # ---

        self.shape = utils.broadcast_shapes(
            shape,
            utils.try_get_batch_shape(joint_T, -3),
            utils.try_get_batch_shape(vert_pos, -2),
            utils.try_get_batch_shape(tex_vert_pos, -2),
            utils.try_get_batch_shape(vert_trans, -3),
        )

        self.kin_tree = kin_tree

        self.mesh_graph = mesh_graph
        self.tex_mesh_graph = tex_mesh_graph

        self.joint_T = joint_T  # [..., J, 4, 4]

        self.vert_pos = vert_pos  # [..., V, 3]

        self.tex_vert_pos = tex_vert_pos  # [..., TV, 2]

        self.vert_trans = vert_trans  # [..., V, 4, 4]

        self.mesh_data = mesh_utils.MeshData(self.mesh_graph, self.vert_pos)

    @property
    def joints_cnt(self) -> int:
        return self.kin_tree.joints_cnt

    @property
    def verts_cnt(self) -> int:
        return self.mesh_graph.verts_cnt

    @property
    def tex_verts_cnt(self) -> int:
        return self.tex_mesh_graph.verts_cnt

    @property
    def faces_cnt(self) -> int:
        return self.mesh_graph.faces_cnt

    def __getitem__(self, idx) -> AvatarModel:
        return AvatarModel(
            kin_tree=self.kin_tree,

            mesh_graph=self.mesh_graph,
            tex_mesh_graph=self.tex_mesh_graph,

            joint_T=utils.try_batch_indexing(
                self.joint_T, self.shape, -3, idx),

            vert_pos=utils.try_batch_indexing(
                self.vert_pos, self.shape, -2, idx),

            tex_vert_pos=utils.try_batch_indexing(
                self.tex_vert_pos, self.shape, -2, idx),

            vert_trans=utils.try_batch_indexing(
                self.vert_trans, self.shape, -3, idx),
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
    ) -> mesh_utils.MeshSubdivisionResult:
        raise NotImplementedError()

    def forward(self, blending_param) -> AvatarModel:
        raise NotImplementedError()
