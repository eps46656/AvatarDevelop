
import graph_utils
import utils


class Mesh:
    def __init__(self):
        self.vertices: dict[int, object] = dict()

        self.face_attrs: dict[int, object] = dict()

        self.vns_to_fn: dict[tuple[int, int, int], int] = dict()
        self.fn_to_vns: dict[int, tuple[int, int, int]] = dict()

        self.graph = graph_utils.Graph()

        self.last_allocated_num = -1

    def __AllocateNum(self):
        self.last_allocated_num += 1
        return self.last_allocated_num

    def ContainFace(self, vns: tuple[int, int, int]):
        vns = tuple(sorted(vns))

        assert vns[0] != vns[1]
        assert vns[1] != vns[2]

        assert vns[0] in self.vertices
        assert vns[1] in self.vertices
        assert vns[2] in self.vertices

        return vns in self.vns_to_fn

    def GetFN(self, vns: tuple[int, int, int]):
        return self.vns_to_fn.get(tuple(sorted(vns)))

    def GetVNS(self, fn: int):
        return self.fn_to_vns.get(fn)

    def AddVertex(self, vertex_attr: object):
        vn = self.__AllocateNum()

        self.vertices[vn] = vertex_attr

        self.graph.AddVertex(vn)

        return vn

    def AddFace(self,
                vns: tuple[int, int, int],
                face_attr: object):
        assert vns[0] in self.vertices
        assert vns[1] in self.vertices
        assert vns[2] in self.vertices

        sorted_vns = tuple(sorted(vns))

        assert sorted_vns[0] != sorted_vns[1]
        assert sorted_vns[1] != sorted_vns[2]

        if sorted_vns in self.vns_to_fn:
            return False

        fn = self.__AllocateNum()

        self.face_attrs[fn] = face_attr

        self.fn_to_vns[fn] = vns
        self.vns_to_fn[sorted_vns] = fn

        self.graph.AddVertex(fn)

        self.graph.AddEdge(vns[0], fn, bidirectional=True)
        self.graph.AddEdge(vns[1], fn, bidirectional=True)
        self.graph.AddEdge(vns[2], fn, bidirectional=True)

        return True

    def RemoveVertex(self, vn: int):
        if vn not in self.vertices:
            return False

        fns = list(self.graph.adjacents(vn))

        for fn in fns:
            self.RemoveFace(fn)

        self.vertices.pop(vn)

        return True

    def RemoveFace(self, fn: int):
        return utils.DictPop(self.face_attrs, fn)

    def Output(self):
        ret_vs = {
            vn: vertex_attr
            for vn, vertex_attr in self.vertices.items()}

        ret_fs = {
            fn: (self.fn_to_vns[fn], face_attr)
            for fn, face_attr in self.face_attrs.items()}

        return ret_vs, ret_fs
