import collections
import types
import typing

import torch
from beartype import beartype

import utils


Vertex = typing.TypeVar("Vertex")


@beartype
def CheckGraph(adj_lists: dict[Vertex, set[Vertex]]):
    for adj_list in adj_lists.values():
        for u in adj_list:
            assert u in adj_lists


@beartype
def FindRoots_(adj_lists: dict[Vertex, set[Vertex]]):
    CheckGraph(adj_lists)

    indegs = {v: 0 for v in adj_lists.keys()}

    for adj_list in adj_lists.values():
        for u in adj_list:
            indegs[u] += 1

            if 2 <= indegs[u]:
                return

    yield from (v for v, indeg in indegs.items() if indeg == 0)


@beartype
def FindRoots(adj_lists: dict[Vertex, set[Vertex]]):
    CheckGraph(adj_lists)
    return FindRoots_(adj_lists)


@beartype
def BFS_(adj_lists: dict[Vertex, set[Vertex]],
         sources: typing.Iterable[Vertex]):
    passed = set()

    q = collections.deque()

    for source in sources:
        assert source in adj_lists

        if not utils.SetAdd(passed, source):
            continue

        yield source
        q.append(source)

        while 0 < len(q):
            v = q.popleft()

            for u in adj_lists[v]:
                if utils.SetAdd(passed, u):
                    yield u
                    q.append(u)


@beartype
def BFS(adj_lists: dict[Vertex, set[Vertex]],
        sources: typing.Iterable[Vertex]):
    CheckGraph(adj_lists)
    return BFS_(adj_lists, sources)


@beartype
def TPS_(adj_lists: dict[Vertex, set[Vertex]],
         sources: typing.Iterable[Vertex]):
    indegs = {v: 0 for v in BFS_(adj_lists, sources)}

    for v in indegs.keys():
        for u in adj_lists[v]:
            if u in indegs:
                indegs[u] += 1

    q = collections.deque()

    for v, indeg in indegs.items():
        if indeg == 0:
            yield v
            q.append(v)

    while 0 < len(q):
        v = q.popleft()

        for u in adj_lists[v]:
            if u not in indegs:
                continue

            indegs[u] -= 1

            if indegs[u] == 0:
                yield v
                q.append(u)


@beartype
def TPS(adj_lists: dict[Vertex, set[Vertex]],
        sources: typing.Iterable[Vertex]):
    CheckGraph(adj_lists)
    return TPS_(adj_lists, sources)


@beartype
class Graph(typing.Generic[Vertex]):
    def __init__(self):
        self.__adj_lists: dict[Vertex, set[Vertex]] = dict()
        self.__inv_adj_lists: dict[Vertex, set[Vertex]] = dict()

    def GetVertices(self):
        return self.__adj_lists.keys()

    def GetEdges(self):
        for v, adj_list in self.__adj_lists.items():
            for u in adj_list:
                yield (v, u)

    def GetAdjs(self, src: Vertex):
        assert src in self.__adj_lists
        return iter(self.__adj_lists[src])

    def GetInvAdjs(self, src: Vertex):
        assert src in self.__inv_adj_lists
        return iter(self.__inv_adj_lists[src])

    def GetEdgesCount(self):
        return sum(len(adj_list) for adj_list in self.__adj_lists.values())

    def GetDegrees(self, v: Vertex):
        assert v in self.__adj_lists
        return len(self.__adj_lists[v]), len(self.__inv_adj_lists[v])

    def ContainVertex(self, v: Vertex):
        return v in self.__adj_lists

    def ContainEdge(self, src: Vertex, dst: Vertex, bidirectional: bool):
        assert src in self.__adj_lists
        assert dst in self.__adj_lists

        return dst in self.__adj_lists[src] or (bidirectional and src in self.__adj_lists[dst])

    def AddVertex(self, v: Vertex):
        if not utils.DictInsert(self.__adj_lists, v, set())[2]:
            return False

        utils.DictInsert(self.__inv_adj_lists, v, set())

        return True

    def AddEdge(self,
                src: Vertex,
                dst: Vertex,
                *,
                auto_add_vertex: bool = False,
                bidirectional: bool = False):
        if auto_add_vertex:
            self.AddVertex(src)
            self.AddVertex(dst)

        assert src in self.__adj_lists
        assert dst in self.__adj_lists

        ret = 0

        if utils.SetAdd(self.__adj_lists[src], dst):
            ret += 1

        if bidirectional and utils.SetAdd(self.__adj_lists[dst], src):
            ret += 1

        return ret

    def RemoveVertex(self, src: Vertex):
        assert src in self.__adj_lists

        ret = 0

        for u in set(self.__adj_lists[src] | self.__inv_adj_lists[src]):
            ret += self.RemoveEdge(src, u, bidirectional=True)

        self.__adj_lists.pop(src)
        self.__inv_adj_lists.pop(src)

        return ret

    def RemoveEdge(self,
                   src: Vertex,
                   dst: Vertex,
                   *,
                   bidirectional: bool = False):
        assert src in self.__adj_lists
        assert dst in self.__adj_lists

        ret = 0

        if utils.SetDiscard(self.__adj_lists[src], dst):
            utils.SetDiscard(self.__inv_adj_lists[dst], src)
            ret += 1

        if bidirectional and utils.SetDiscard(self.__adj_lists[dst], src):
            utils.SetDiscard(self.__inv_adj_lists[src], dst)
            ret += 1

        return ret

    def ImportFromAdjList(self, imported_adj_lists: dict[Vertex, set[Vertex]]):
        for imported_v, imported_adj_list in imported_adj_lists.items():
            for imported_u in imported_adj_list:
                self.AddEdge(imported_v, imported_u, auto_add_vertex=True)

    def ImportFromAdjRelList(self, imported_adj_rel_lists: typing.Iterable[typing.Iterable[Vertex]], bidirectional: bool = False):
        for imported_v, imported_u in imported_adj_rel_lists:
            self.AddVertex(imported_v)
            self.AddVertex(imported_u)

            self.AddEdge(imported_v, imported_u,
                         auto_add_vertex=True, bidirectional=bidirectional)

    def BFS(self, srcs: typing.Iterable[Vertex]):
        global BFS
        return BFS(self.__adj_lists, srcs)

    def TPS(self, srcs: typing.Iterable[Vertex]):
        global TPS
        return TPS(self.__adj_lists, srcs)

    def Inverse(self):
        self.__adj_lists, self.__inv_adj_lists = \
            self.__inv_adj_lists, self.__adj_lists


class RelationBuffer:
    BASIC_CAPACITY = 16

    @staticmethod
    def Allocate_(capacity: int, device: torch.device):
        return torch.empty((2, max(RelationBuffer.BASIC_CAPACITY, capacity)),
                           dtype=torch.long, device=device)

    def __init__(self, device: torch.device):
        self.data = RelationBuffer.Allocate_(
            RelationBuffer.BASIC_CAPACITY, device)

        self.cur_size: int = 0

        self.rel_to_idx: dict[tuple[int, int], int] = dict()

    def __len__(self):
        return self.cur_size

    def __contains__(self, rel: tuple[int, int]) -> bool:
        return rel in self.rel_to_idx

    def reserve(self, capacity: int):
        if capacity <= self.data.shape[1]:
            return False

        nxt_capacity = max(self.data.shape[1] * 3 // 2, capacity)

        new_data = RelationBuffer.Allocate_(nxt_capacity, self.data.device)
        new_data[:, :self.cur_size] = self.data
        self.data = new_data

        return True

    def add(self, rel: tuple[int, int]) -> bool:
        if self.rel_to_idx.setdefault(rel, self.cur_size) != self.cur_size:
            return False

        self.reserve(self.cur_size + 1)
        self.data[:, self.cur_size] = rel
        self.cur_size += 1

        return True

    def erase(self, rel: tuple[int, int]) -> bool:
        idx = self.rel_to_idx.pop(rel, -1)

        if idx == -1:
            return False

        self.cur_size -= 1

        if idx != self.cur_size:
            r = tuple(int(v) for v in self.data[:, self.cur_size])
            self.data[:, idx] = r
            self.rel_to_idx[r] = idx

        return True

    def view(self):
        return self.data[:, :self.cur_size]
