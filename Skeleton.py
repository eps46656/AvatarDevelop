
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
        self.__graph = copy.deepcopy(graph)

        for coord in base_transforms.keys():
            assert coord in self.__graph

        # --

        self.__parents = {
            v: None for v in self.__graph.vertices()}

        for v, u in self.__graph.edges():
            assert self.__parents[u] is None
            self.__parents[u] = v

        # --

        roots = [v for v, p in self.__parents.items() if p is None]

        assert len(roots) == 1

        self.__root = roots[0]

        # --

        self.__tp_vertices = list(self.__graph.TPS((self.__root)))

        assert len(self.__tp_vertices) == len(self.__graph.vertices())

        # --

        dtype = base_transforms.type
        device = base_transforms.device

        # --

        self.__base_transforms = {
            coord: base_transforms[coord]
            for coord in self.__graph.vertices()}

        # --

        if art_transforms is None:
            self.__art_transforms = {
                v: torch.identity(4).to(dtype=dtype, device=device)
                for v in self.__graph.vertices()}
        else:
            self.__art_transforms = None
            self.SetArtTransforms(art_transforms)

        # --

        self.__local_transforms = None
        self.__world_transforms = None

    def __CalcLocalTransforms(self):
        if self.__local_transforms is not None:
            return

        self.__local_transforms = {
            coord: self.__base_transforms[coord] @ self.__art_transforms[coord]
            for coord in self.__graph.vertices()}

    def GetLocalTransforms(self):
        self.__CalcLocalTransforms()

        return {
            coord: self.__local_transforms[coord]
            for coord in self.__graph.vertices()}

    def __CalcWorldTransforms(self):
        if self.__world_transforms is not None:
            return

        self.__CalcLocalTransforms()

        self.__world_transforms = dict()

        self.__world_transforms[self.__root] = self.__local_transforms[self.__root]

        for coord in self.__tp_vertices:
            p_coord = self.__parents[coord]

            if p_coord is None:
                continue

            self.__world_transforms[coord] = self.__world_transforms[p_coord] @ self.__local_transforms[coord]

    def GetWorldTransforms(self):
        self.__CalcWorldTransforms()

        return {
            coord: self.__world_transforms[coord]
            for coord in self.__graph.vertices()}

    def SetArtTransforms(self, art_transforms: dict[object, torch.Tensor]):
        for v in self.__graph.vertices():
            assert v in art_transforms
            assert art_transforms[v].shape == (4, 4)

        self.__art_transforms = {
            v: art_transforms[v]
            for v in self.__graph.vertices()}

        self.__local_transforms = None
        self.__world_transforms = None
