from __future__ import annotations

import collections
import heapq
import typing

import torch
from beartype import beartype

from . import utils


@beartype
class KinTree:
    def __init__(
        self,
        *,
        root: int,
        parents: list[int],
        joints_tp: list[int],
    ):
        self.root = root
        self.parents = parents
        self.joints_tp = joints_tp

    @staticmethod
    def empty() -> KinTree:
        return KinTree(
            root=-1,
            parents=list(),
            joints_tp=list(),
        )

    @staticmethod
    def from_links(links: typing.Sequence[tuple[int, int]]) -> KinTree:
        J = len(links)

        root = -1
        parents = [-1] * J
        indegrees = [0] * J

        for p, u in links:
            assert 0 <= u and u < J

            if p < 0 or J <= p:
                if root != -1:
                    raise ValueError(
                        f"Multiple root nodes detected: {root} and {u}.")

                root = u
            else:
                if p < 0 or J <= p:
                    raise ValueError(
                        f"Invalid joint index detected: {p}.")

                parents[u] = p
                indegrees[p] += 1

        if root == -1:
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
            root=root,
            parents=parents,
            joints_tp=list(reversed(reversed_joints_tp))
        )

    @staticmethod
    def from_parents(parents: typing.Iterable[int]) -> KinTree:
        return KinTree.from_links([(p, u) for u, p in enumerate(parents)])

    @staticmethod
    def from_state_dict(state_dict: typing.Mapping[str, object]) -> KinTree:
        return KinTree.from_parents(state_dict["parents"])

    @property
    def joints_cnt(self) -> int:
        return len(self.parents)

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([("parents", self.parents)])

    def __eq__(self, other: KinTree) -> bool:
        return self.parents == other.parents

    def __ne__(self, other: KinTree) -> bool:
        return not (self == other)

    def get_joint_T(
        self,
        pose_r: torch.Tensor,  # [..., J, D, D]
        pose_t: torch.Tensor,  # [..., J, D]
    ) -> torch.Tensor:  # joint_T[..., J, D + 1, D + 1]:
        J = self.joints_cnt

        D = utils.check_shapes(
            pose_r, (..., J, -1, -1),
            pose_t, (..., J, -1),
        )

        device = utils.check_devices(pose_r, pose_t)

        batch_shape = utils.broadcast_shapes(
            pose_r.shape[:-3],
            pose_t.shape[:-2],
        )

        joint_T = torch.empty(
            (*batch_shape, J, D + 1, D + 1),
            dtype=utils.promote_dtypes(pose_r, pose_t),
            device=device,
        )
        # [..., J, D + 1, D + 1]

        joint_T[..., D, :D] = 0
        joint_T[..., D, D] = 1

        joint_T[..., self.root, :D, :D] = pose_r[..., self.root, :, :]
        joint_T[..., self.root, :D, D] = pose_t[..., self.root, :]

        for u in self.joints_tp[1:]:
            p = self.parents[u]

            cur_joint_T = joint_T[..., u, :, :]  # [..., D + 1, D + 1]
            parent_joint_T = joint_T[..., p, :, :]  # [... D + 1, D + 1]

            cur_joint_T[..., :D, :D], cur_joint_T[..., :D, D] = utils.merge_rt(
                parent_joint_T[..., :D, :D],
                parent_joint_T[..., :D, D],
                pose_r[..., u, :, :],
                pose_t[..., u, :],
            )

        return joint_T
