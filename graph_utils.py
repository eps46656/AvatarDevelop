import collections
import itertools
import typing

from typeguard import typechecked

import utils


@typechecked
def CheckGraph(adj_lists: dict[object, set[object]]):
    for adj_list in adj_lists.values():
        for u in adj_list:
            assert u in adj_lists


@typechecked
def FindRoots_(adj_lists: dict[object, set[object]]):
    CheckGraph(adj_lists)

    indegs = {v: 0 for v in adj_lists.keys()}

    for adj_list in adj_lists.values():
        for u in adj_list:
            indegs[u] += 1

            if 2 <= indegs[u]:
                return

    yield from (v for v, indeg in indegs.items() if indeg == 0)


@typechecked
def FindRoots(adj_lists: dict[object, set[object]]):
    CheckGraph(adj_lists)
    return FindRoots_(adj_lists)


@typechecked
def BFS_(adj_lists: dict[object, set[object]],
         sources: typing.Iterable[object]):
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


@typechecked
def BFS(adj_lists: dict[object, set[object]],
        sources: typing.Iterable[object]):
    CheckGraph(adj_lists)
    return BFS_(adj_lists, sources)


@typechecked
def TPS_(adj_lists: dict[object, set[object]],
         sources: typing.Iterable[object]):
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


@typechecked
def TPS(adj_lists: dict[object, set[object]],
        sources: typing.Iterable[object]):
    CheckGraph(adj_lists)
    return TPS_(adj_lists, sources)


@typechecked
class Graph:
    def __init__(self,
                 adj_lists: typing.Optional[dict[object, set[object]]] = None):
        self.__adj_lists: dict[object, set[object]] = dict()
        self.__inv_adj_lists: dict[object, set[object]] = dict()

        self.__idegs: dict[object, int] = dict()
        self.__odegs: dict[object, int] = dict()

        if adj_lists is not None:
            self.Import(adj_lists)

    def vertices(self):
        return self.__adj_lists.keys()

    def edges(self):
        for v, adj_list in self.__adj_lists.items():
            for u in adj_list:
                yield (v, u)

    def adjacents(self, src: object):
        assert src in self.__adj_lists
        return iter(self.__adj_lists[src])

    def inv_adjacents(self, src: object):
        assert src in self.__inv_adj_lists
        return iter(self.__inv_adj_lists[src])

    def GetEdgesCount(self):
        return sum(len(adj_list) for adj_list in self.__adj_lists.values())

    def GetDegrees(self, v: object):
        assert v in self.__adj_lists
        return self.__idegs[v], self.__odegs[v]

    def ContainVertex(self, vertex: object):
        return vertex in self.__adj_lists

    def ContainEdge(self, src: object, dst: object, bidirectional: bool):
        return (dst in self.__adj_lists.get(src, set()) or (bidirectional and src in self.__adj_lists.get(dst)))

    def AddVertex(self, vertex: object):
        v, adj_list, success = utils.DictInsert(
            self.__adj_lists, vertex, set())

        if not success:
            return False

        utils.DictInsert(
            self.__inv_adj_lists, vertex, set())

        self.__idegs[vertex] = 0
        self.__odegs[vertex] = 0

        return True

    def AddEdge(self,
                src: object,
                dst: object,
                *,
                auto_add_vertex: bool = False,
                bidirectional: bool = False):
        if auto_add_vertex:
            self.AddVertex(src)
            self.AddVertex(dst)

        assert src in self.__adj_lists
        assert dst in self.__adj_lists

        if utils.SetAdd(self.__adj_lists[src], dst):
            self.__idegs[dst] += 1
            self.__odegs[src] += 1

        if bidirectional and utils.SetAdd(self.__adj_lists[dst], src):
            self.__idegs[src] += 1
            self.__odegs[dst] += 1

    def RemoveVertex(self, src: object):
        assert src in self.__adj_lists

        us = [v for v in itertools.chain(
            self.__adj_lists[src], self.__inv_adj_lists[src])]

        for u in us:
            self.RemoveEdge(src, u, bidirectional=True)

        assert len(self.__adj_lists[src]) == 0
        assert len(self.__inv_adj_lists[src]) == 0

        self.__adj_lists.pop(src)
        self.__inv_adj_lists.pop(src)

    def RemoveEdge(self,
                   src: object,
                   dst: object,
                   *,
                   bidirectional: bool = False):
        assert src in self.__adj_lists
        assert dst in self.__adj_lists

        if utils.SetDiscard(self.__adj_lists[src], dst):
            utils.SetDiscard(self.__inv_adj_lists[dst], src)

            self.__idegs[dst] -= 1
            self.__odegs[src] -= 1

        if bidirectional and utils.SetDiscard(self.__adj_lists[dst], src):
            utils.SetDiscard(self.__inv_adj_lists[src], dst)

            self.__idegs[src] -= 1
            self.__odegs[dst] -= 1

    def Import(self, imported_adj_lists: dict[object, set[object]]):
        for imported_adj_list in imported_adj_lists.values():
            for u in imported_adj_list:
                assert u in self.__adj_lists or u in imported_adj_lists

        for imported_v in imported_adj_lists.keys():
            self.AddVertex(imported_v)

        for imported_v, imported_adj_list in imported_adj_lists.keys():
            for u in imported_adj_list:
                self.AddEdge(imported_v, u)

    def BFS(self, srcs: typing.Iterable[object]):
        global BFS
        return BFS(self.__adj_lists, srcs)

    def TPS(self, srcs: typing.Iterable[object]):
        global TPS
        return TPS(self.__adj_lists, srcs)

    def Inverse(self):
        self.__adj_list, self.__inv_adj_list = \
            self.__inv_adj_list, self.__adj_list

        self.__idegs, self.__odegs = \
            self.__odegs, self.__idegs
