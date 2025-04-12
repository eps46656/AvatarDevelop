import collections
import types
import typing

import torch
from beartype import beartype

import utils
import heapq


Vertex = typing.TypeVar("Vertex")


@beartype
def check_graph(adj_lists: dict[Vertex, set[Vertex]]):
    for adjs in adj_lists.values():
        for u in adjs:
            assert u in adj_lists


@beartype
def _find_roots(adj_lists: dict[Vertex, set[Vertex]]):
    check_graph(adj_lists)

    in_degs = {v: 0 for v in adj_lists.keys()}

    for adjs in adj_lists.values():
        for u in adjs:
            in_degs[u] += 1

            if 2 <= in_degs[u]:
                return

    yield from (v for v, indeg in in_degs.items() if indeg == 0)


@beartype
def find_roots(adj_lists: dict[Vertex, set[Vertex]]):
    check_graph(adj_lists)
    return _find_roots(adj_lists)


@beartype
def _bfs(
    adj_lists: dict[Vertex, set[Vertex]],
    srcs: typing.Iterable[Vertex],
):
    passed = set()

    q = collections.deque()

    for src in srcs:
        assert src in adj_lists

        if not utils.set_add(passed, src):
            continue

        q.append(src)

        while 0 < len(q):
            v = q.popleft()
            yield v

            for u in adj_lists[v]:
                if utils.set_add(passed, u):
                    q.append(u)


@beartype
def bfs(
    adj_lists: dict[Vertex, set[Vertex]],
    srcs: typing.Iterable[Vertex],
) -> typing.Iterator[Vertex]:
    check_graph(adj_lists)
    return _bfs(adj_lists, srcs)


@beartype
def _tps(
    adj_lists: dict[Vertex, set[Vertex]],
    srcs: typing.Iterable[Vertex],
) -> typing.Iterator[Vertex]:
    in_degs = {v: 0 for v in _bfs(adj_lists, srcs)}

    for v in in_degs.keys():
        for u in adj_lists[v]:
            if u in in_degs:
                in_degs[u] += 1

    q: list[Vertex] = list()

    for v, in_deg in in_degs.items():
        if in_deg == 0:
            heapq.heappush(q, v)

    while 0 < len(q):
        v = heapq.heappop(q)
        yield v

        for u in adj_lists[v]:
            in_degs[u] -= 1

            if in_degs[u] == 0:
                heapq.heappush(q, v)


@beartype
def tps(
    adj_lists: dict[Vertex, set[Vertex]],
    sources: typing.Iterable[Vertex],
) -> typing.Iterator[Vertex]:
    check_graph(adj_lists)
    return _tps(adj_lists, sources)


@beartype
class Graph(typing.Generic[Vertex]):
    def __init__(self):
        self.__adj_lists: dict[Vertex, set[Vertex]] = dict()
        self.__inv_adj_lists: dict[Vertex, set[Vertex]] = dict()

    @property
    def verts_cnt(self) -> int:
        return len(self.__adj_lists)

    @property
    def edges_cnt(self) -> int:
        return sum(len(adj_list) for adj_list in self.__adj_lists.values())

    def get_verts(self) -> typing.Iterator[Vertex]:
        return self.__adj_lists.keys()

    def get_edges(self) -> typing.Iterator[tuple[Vertex, Vertex]]:
        for v, adj_list in self.__adj_lists.items():
            for u in adj_list:
                yield (v, u)

    def get_adjs(self, src: Vertex) -> typing.Iterator[set[Vertex]]:
        assert src in self.__adj_lists
        return iter(self.__adj_lists[src])

    def get_inv_adjs(self, src: Vertex) -> typing.Iterator[set[Vertex]]:
        assert src in self.__inv_adj_lists
        return iter(self.__inv_adj_lists[src])

    def get_deg(self, v: Vertex) -> tuple[int, int]:
        assert v in self.__adj_lists
        return len(self.__adj_lists[v]), len(self.__inv_adj_lists[v])

    def contain_vert(self, v: Vertex) -> bool:
        return v in self.__adj_lists

    def contain_edge(
        self,
        src: Vertex,
        dst: Vertex,
        bidirectional: bool,
    ) -> bool:
        assert src in self.__adj_lists
        assert dst in self.__adj_lists

        return dst in self.__adj_lists[src] or (bidirectional and src in self.__adj_lists[dst])

    def add_vert(self, v: Vertex) -> bool:
        if not utils.dict_insert(self.__adj_lists, v, set())[2]:
            return False

        utils.dict_insert(self.__inv_adj_lists, v, set())

        return True

    def add_edge(
        self,
        src: Vertex,
        dst: Vertex,
        auto_add_vert: bool = False,
        bidirectional: bool = False,
    ) -> int:  # number of edge added
        if auto_add_vert:
            self.add_vert(src)
            self.add_vert(dst)

        assert src in self.__adj_lists
        assert dst in self.__adj_lists

        ret = 0

        if utils.set_add(self.__adj_lists[src], dst):
            ret += 1

        if bidirectional and utils.set_add(self.__adj_lists[dst], src):
            ret += 1

        return ret

    def remove_vert(self, src: Vertex) -> int:  # number of vert removed
        assert src in self.__adj_lists

        ret = 0

        for u in set(self.__adj_lists[src] | self.__inv_adj_lists[src]):
            ret += self.remove_edge(src, u, bidirectional=True)

        self.__adj_lists.pop(src)
        self.__inv_adj_lists.pop(src)

        return ret

    def remove_edge(
        self,
        src: Vertex,
        dst: Vertex,
        bidirectional: bool = False,
    ) -> int:  # number of edge removed
        assert src in self.__adj_lists
        assert dst in self.__adj_lists

        ret = 0

        if utils.set_discard(self.__adj_lists[src], dst):
            utils.set_discard(self.__inv_adj_lists[dst], src)
            ret += 1

        if bidirectional and utils.set_discard(self.__adj_lists[dst], src):
            utils.set_discard(self.__inv_adj_lists[src], dst)
            ret += 1

        return ret

    def import_from_adj_list(
        self,
        adj_lists: dict[Vertex, set[Vertex]],
        bidirectional: bool = False,
    ) -> None:
        for v, adjs in adj_lists.items():
            for u in adjs:
                self.add_edge(v, u, True, bidirectional)

    def import_from_adj_rel_list(
        self,
        adj_rel_lists: typing.Iterable[typing.Iterable[Vertex]],
        bidirectional: bool = False,
    ) -> None:
        for v, u in adj_rel_lists:
            self.add_edge(v, u, True, bidirectional)

    def bfs(self, srcs: typing.Iterable[Vertex]):
        global bfs
        return bfs(self.__adj_lists, srcs)

    def tps(self, srcs: typing.Iterable[Vertex]):
        global tps
        return tps(self.__adj_lists, srcs)

    def inverse(self):
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
