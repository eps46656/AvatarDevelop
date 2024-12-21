
import random

import graph_utils
import utils


class AbstractMesh:
    __MOD = 1000000007

    def __init__(self):
        self.__vertices: dict[int, object] = dict()

        self.__faces: dict[int, object] = dict()

        self.__vis_to_fi: dict[tuple[int, int, int], int] = dict()
        self.__fi_to_vis: dict[int, tuple[int, int, int]] = dict()

        self.__graph = graph_utils.Graph()

        self.__idx_allocation_seed = random.randint(1, AbstractMesh.__MOD-1)

    def __AllocateIdx(self):
        self.__idx_allocation_seed = (self.__idx_allocation_seed * 23) \
            % AbstractMesh.__MOD

        return self.__idx_allocation_seed

    def ContainFace(self, vis: tuple[int, int, int]):
        vis = tuple(sorted(vis))

        assert vis[0] != vis[1]
        assert vis[1] != vis[2]

        assert vis[0] in self.__vertices
        assert vis[1] in self.__vertices
        assert vis[2] in self.__vertices

        return vis in self.__vis_to_fi

    def GetFaceIdx(self, vis: tuple[int, int, int]):
        return self.__vis_to_fi.get(tuple(sorted(vis)))

    def GetVIS(self, fi: int):
        return self.__fi_to_vis.get(fi)

    def GetAdjacentVerticesFromVertex(self, vi: int):
        s = {vi}

        for fi in self.__graph.GetAdjacents(vi):
            for adj_vi in self.__graph.GetAdjacents(fi):
                if utils.SetAdd(s, adj_vi):
                    yield adj_vi

    def GetAdjacentVerticesFromFace(self, fi: int):
        vis = self.__fi_to_vis.get(fi)
        return list() if vis is None else list(vis)

    def GetAdjacentFacesFromVertex(self, vi: int):
        assert vi in self.__vertices
        return self.__graph.GetAdjacents(vi)

    def GetAdjacentFacesFromFace(self, fi: int):
        assert fi in self.__faces

        va, vb, vc = self.__fi_to_vis[fi]

        fisa = set(fia for fia in self.__graph.GetAdjacents(va) if fia != fi)
        fisb = set(fib for fib in self.__graph.GetAdjacents(vb) if fib != fi)
        fisc = set(fic for fic in self.__graph.GetAdjacents(vc) if fic != fi)

        yield from set.intersection(fisa, fisb)
        yield from set.intersection(fisb, fisc)
        yield from set.intersection(fisc, fisa)

    def AddVertex(self, vertex_attr: object):
        vi = self.__AllocateIdx()

        self.__vertices[vi] = vertex_attr

        self.__graph.AddVertex(vi)

        return vi

    def AddFace(self,
                vis: tuple[int, int, int],
                face_attr: object):
        assert vis[0] in self.__vertices
        assert vis[1] in self.__vertices
        assert vis[2] in self.__vertices

        sorted_vis = tuple(sorted(vis))

        assert sorted_vis[0] != sorted_vis[1]
        assert sorted_vis[1] != sorted_vis[2]

        fi = self.__vis_to_fi.get(sorted_vis)

        if fi is not None:
            return fi, False

        fi = self.__AllocateIdx()

        self.__faces[fi] = face_attr

        self.__fi_to_vis[fi] = vis
        self.__vis_to_fi[sorted_vis] = fi

        self.__graph.AddVertex(fi)

        self.__graph.AddEdge(vis[0], fi, bidirectional=True)
        self.__graph.AddEdge(vis[1], fi, bidirectional=True)
        self.__graph.AddEdge(vis[2], fi, bidirectional=True)

        return fi, True

    def RemoveVertex(self, vi: int):
        if vi not in self.__vertices:
            return False

        fis = list(self.__graph.GetAdjacents(vi))

        for fi in fis:
            self.RemoveFace(fi)

        self.__vertices.pop(vi)

        return True

    def RemoveFace(self, fi: int):
        if not utils.DictPop(self.__faces, fi):
            return False

        self.__graph.RemoveVertex(fi)

        return True

    def Output(self):
        ret_vs = {
            vi: vertex_attr
            for vi, vertex_attr in self.__vertices.items()}

        ret_fs = {
            fi: (self.__fi_to_vis[fi], face_attr)
            for fi, face_attr in self.__faces.items()}

        return ret_vs, ret_fs
