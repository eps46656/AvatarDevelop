
import graph_utils
import utils


class Mesh:
    def __init__(self):
        self.vertices: dict[int, object] = dict()

        self.face_attrs: dict[int, object] = dict()

        self.vns_to_fn: dict[tuple[int, int, int], int] = dict()
        self.fn_to_vns: dict[int, tuple[int, int, int]] = dict()

        self.graph = graph_utils.Graph()

        self.last_allocated_idx = -1

    def __AllocateIdx(self):
        self.last_allocated_idx += 1
        return self.last_allocated_idx

    def ContainFace(self, vis: tuple[int, int, int]):
        vis = tuple(sorted(vis))

        assert vis[0] != vis[1]
        assert vis[1] != vis[2]

        assert vis[0] in self.vertices
        assert vis[1] in self.vertices
        assert vis[2] in self.vertices

        return vis in self.vns_to_fn

    def GetFN(self, vis: tuple[int, int, int]):
        return self.vns_to_fn.get(tuple(sorted(vis)))

    def GetVNS(self, fi: int):
        return self.fn_to_vns.get(fi)

    def AddVertex(self, vertex_attr: object):
        vi = self.__AllocateIdx()

        self.vertices[vi] = vertex_attr

        self.graph.AddVertex(vi)

        return vi

    def AddFace(self,
                vis: tuple[int, int, int],
                face_attr: object):
        assert vis[0] in self.vertices
        assert vis[1] in self.vertices
        assert vis[2] in self.vertices

        sorted_vns = tuple(sorted(vis))

        assert sorted_vns[0] != sorted_vns[1]
        assert sorted_vns[1] != sorted_vns[2]

        if sorted_vns in self.vns_to_fn:
            return False

        fi = self.__AllocateIdx()

        self.face_attrs[fi] = face_attr

        self.fn_to_vns[fi] = vis
        self.vns_to_fn[sorted_vns] = fi

        self.graph.AddVertex(fi)

        self.graph.AddEdge(vis[0], fi, bidirectional=True)
        self.graph.AddEdge(vis[1], fi, bidirectional=True)
        self.graph.AddEdge(vis[2], fi, bidirectional=True)

        return True

    def RemoveVertex(self, vi: int):
        if vi not in self.vertices:
            return False

        fis = list(self.graph.adjacents(vi))

        for fi in fis:
            self.RemoveFace(fi)

        self.vertices.pop(vi)

        return True

    def RemoveFace(self, fi: int):
        return utils.DictPop(self.face_attrs, fi)

    def Output(self):
        ret_vs = {
            vi: vertex_attr
            for vi, vertex_attr in self.vertices.items()}

        ret_fs = {
            fi: (self.fn_to_vns[fi], face_attr)
            for fi, face_attr in self.face_attrs.items()}

        return ret_vs, ret_fs
