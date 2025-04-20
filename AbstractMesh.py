
import random

import graph_utils
import utils


class AbstractMesh:
    __MOD = 1000000007

    def __init__(self):
        self.__verts: dict[int, object] = dict()

        self.__faces: dict[int, object] = dict()

        self.__f_to_vvv: dict[int, tuple[int, int, int]] = dict()
        self.__vvv_to_f: dict[tuple[int, int, int], int] = dict()

        self.__graph = graph_utils.Graph()

        self.__idx_allocation_seed = random.randint(1, AbstractMesh.__MOD-1)

    def __allocate_idx(self):
        self.__idx_allocation_seed = (self.__idx_allocation_seed * 23) \
            % AbstractMesh.__MOD

        return self.__idx_allocation_seed

    def get_face(self, vvv: tuple[int, int, int]) -> int:
        return self.__vvv_to_f.get(tuple(sorted(vvv)), -1)

    def get_vvv(self, f: int) -> tuple[int, int, int]:
        return self.__f_to_vvv.get(f)

    def get_adj_verts_from_vert(self, v: int):
        s = {v}

        for f in self.__graph.get_adjs(v):
            for adj_v in self.__graph.get_adjs(f):
                if utils.set_add(s, adj_v):
                    yield adj_v

    def get_adj_verts_from_face(self, f: int):
        vvv = self.__f_to_vvv.get(f)
        return list() if vvv is None else list(vvv)

    def get_adj_faces_from_vert(self, v: int):
        assert v in self.__verts
        return self.__graph.get_adjs(v)

    def get_adj_faces_from_face(self, f: int):
        assert f in self.__faces

        va, vb, vc = self.__f_to_vvv[f]

        fas = set(fa for fa in self.__graph.get_adjs(va) if fa != f)
        fbs = set(fb for fb in self.__graph.get_adjs(vb) if fb != f)
        fcs = set(fc for fc in self.__graph.get_adjs(vc) if fc != f)

        yield from set.intersection(fas, fbs)
        yield from set.intersection(fbs, fcs)
        yield from set.intersection(fcs, fas)

    def add_vert(self, vertex_attr: object):
        v = self.__allocate_idx()

        self.__verts[v] = vertex_attr

        self.__graph.add_vert(v)

        return v

    def add_face(
        self,
        vvv: tuple[int, int, int],
        face_attr: object,
    ):
        assert vvv[0] in self.__verts
        assert vvv[1] in self.__verts
        assert vvv[2] in self.__verts

        sorted_vvv = tuple(sorted(vvv))

        assert sorted_vvv[0] != sorted_vvv[1]
        assert sorted_vvv[1] != sorted_vvv[2]

        f = self.__vvv_to_f.get(sorted_vvv, -1)

        if f != -1:
            return f, False

        f = self.__allocate_idx()

        self.__faces[f] = face_attr

        self.__f_to_vvv[f] = vvv
        self.__vvv_to_f[sorted_vvv] = f

        self.__graph.add_vert(f)

        self.__graph.add_edge(vvv[0], f, bidirectional=True)
        self.__graph.add_edge(vvv[1], f, bidirectional=True)
        self.__graph.add_edge(vvv[2], f, bidirectional=True)

        return f, True

    def remove_vert(self, v: int):
        if v not in self.__verts:
            return False

        fs = list(self.__graph.get_adjs(v))

        for f in fs:
            self.remove_face(f)

        self.__verts.pop(v)

        return True

    def remove_face(self, f: int) -> bool:
        if not utils.dict_pop(self.__faces, f):
            return False

        self.__graph.remove_vert(f)

        return True

    def output(self) -> tuple[
        dict[int, object],
        dict[int, tuple[tuple[int, int, int], object]],
    ]:
        ret_vs = {
            v: vert_attr
            for v, vert_attr in self.__verts.items()}

        ret_fs = {
            f: (self.__f_to_vvv[f], face_attr)
            for f, face_attr in self.__faces.items()}

        return ret_vs, ret_fs
