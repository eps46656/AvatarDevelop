
import copy
import typing

import torch
from typeguard import typechecked

import graph_utils


@typechecked
class Skeloton:
    def __init__(self,
                 graph: graph_utils.Graph,
                 base_transforms: dict[object, torch.Tensor],
                 art_transforms: typing.Optional[dict[object, torch.Tensor]],
                 ):
        self.graph = copy.deepcopy(graph)

        for coord in base_transforms.keys():
            assert coord in self.graph

        # --

        self.parents = {
            v: None for v in self.graph.vertices()}

        for v, u in self.graph.edges():
            assert self.parents[u] is None
            self.parents[u] = v

        # --

        roots = [v for v, p in self.parents.items() if p is None]

        assert len(roots) == 1

        self.root = roots[0]

        # --

        self.tp_vertices = self.graph.TPS((self.root))

        assert self.tp_vertices is not None

        # --

        dtype = base_transforms.type
        device = base_transforms.device

        # --

        self.base_transforms = {
            coord: base_transforms[coord]
            for coord in self.graph.vertices()}

        # --

        if art_transforms is None:
            self.art_transforms = {
                v: torch.identity(4).to(dtype=dtype, device=device)
                for v in self.graph.vertices()}
        else:
            self.art_transforms = None
            self.SetArtTransforms(art_transforms)

        # --

        self.local_transforms = None
        self.world_transforms = None

    def __CalcLocalTransforms(self):
        if self.local_transforms is not None:
            return

        self.local_transforms = {
            coord: self.base_transforms[coord] @ self.art_transforms[coord]
            for coord in self.graph.vertices()}

    def GetLocalTransforms(self):
        self.__CalcLocalTransforms()

        return {
            coord: self.local_transforms[coord]
            for coord in self.graph.vertices()}

    def __CalcWorldTransforms(self):
        if self.world_transforms is not None:
            return

        self.__CalcLocalTransforms()

        self.world_transforms = dict()

        self.world_transforms[self.root] = self.local_transforms[self.root]

        for coord in self.tp_vertices:
            p_coord = self.parents[coord]

            if p_coord is None:
                continue

            self.world_transforms[coord] = self.world_transforms[p_coord] @ self.local_transforms[coord]

    def GetWorldTransforms(self):
        self.__CalcWorldTransforms()

        return {
            coord: self.world_transforms[coord]
            for coord in self.graph.vertices()}

    def SetArtTransforms(self, art_transforms: dict[object, torch.Tensor]):
        for v in self.graph.vertices():
            assert v in art_transforms
            assert art_transforms[v].shape == (4, 4)

        self.art_transforms = {
            v: art_transforms[v]
            for v in self.graph.vertices()}

        self.local_transforms = None
        self.world_transforms = None
