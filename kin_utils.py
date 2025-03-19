import dataclasses
import heapq


@dataclasses.dataclass
class KinTree:
    joints_cnt: int
    root: int
    parents: list[int]
    joints_tp: list[int]

    @staticmethod
    def FromLinks(links, null_value=-1):
        J = len(links)

        root = -1
        parents = [-1 for _ in range(J)]
        indegrees = [0 for _ in range(J)]

        for p, u in links:
            assert 0 <= u and u < J

            if p == null_value:
                if not (root == -1):
                    raise ValueError(
                        f"Multiple root nodes detected: {root} and {u}.")

                root = u
            else:
                if not (0 <= p and p < J):
                    raise ValueError(
                        f"Invalid joint index detected: {p}.")

                parents[u] = p
                indegrees[p] += 1

        if not (root != -1):
            raise ValueError("No root node detected.")

        q = [-u for u, indegree in enumerate(indegrees) if indegree == 0]
        heapq.heapify(q)

        reversed_joints_tp = list()

        while 0 < len(q):
            u = -heapq.heappop(q)
            reversed_joints_tp.append(u)

            v = parents[u]

            if v == -1:
                continue

            indegrees[v] -= 1

            if indegrees[v] == 0:
                heapq.heappush(q, -v)

        if not (len(reversed_joints_tp) == J):
            raise ValueError("Loop detected.")

        return KinTree(
            joints_cnt=J,
            root=root,
            parents=parents,
            joints_tp=list(reversed(reversed_joints_tp))
        )

    @staticmethod
    def FromParents(parents: list[int], null_value=-1):
        return KinTree.FromLinks(
            ((p, u) for u, p in enumerate(parents)),
            null_value)
