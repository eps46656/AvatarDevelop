import heapq

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
    def from_links(links, null_value=-1):
        J = len(links)

        root = -1
        parents = [-1] * J
        indegrees = [0] * J

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
            root=root,
            parents=parents,
            joints_tp=list(reversed(reversed_joints_tp))
        )

    @staticmethod
    def from_parents(parents: list[int], null_value=-1):
        return KinTree.from_links(
            ((p, u) for u, p in enumerate(parents)),
            null_value)

    @property
    def joints_cnt(self) -> int:
        return len(self.parents)

    def get_joint_rts(
        self,
        pose_rs: torch.Tensor,  # [..., J, D, D]
        pose_ts: torch.Tensor,  # [..., J, D]
    ) -> torch.Tensor:  # joint_Ts[..., J, D+1, D+1]:
        J = self.joints_cnt

        D = utils.check_shapes(
            pose_rs, (..., J, -1, -1),
            pose_ts, (..., J, -1),
        )

        device = utils.check_devices(pose_rs, pose_ts)

        batch_shape = utils.broadcast_shapes(
            pose_rs.shape[:-3],
            pose_ts.shape[:-2],
        )

        joint_Ts = torch.empty(
            batch_shape + (J, D+1, D+1),
            dtype=torch.promote_types(pose_rs.dtype, pose_ts.dtype),
            device=device,
        )
        # [..., J, D+1, D+1]

        joint_Ts[..., :, D, :D] = 0
        joint_Ts[..., :, D, D] = 1

        joint_Ts[..., self.root, :D, :D] = pose_rs[..., self.root, :, :]
        joint_Ts[..., self.root, :D, D] = pose_ts[..., self.root, :]

        for u in self.joints_tp[1:]:
            p = self.parents[u]

            cur_joint_T = joint_Ts[..., u, :, :]  # [..., D+1, D+1]
            parent_joint_T = joint_Ts[..., p, :, :]  # [... D+1, D+1]

            utils.merge_rt(
                parent_joint_T[..., :D, :D],
                parent_joint_T[..., :D, D],
                pose_rs[..., u, :, :],
                pose_ts[..., u, :],
                out_rs=cur_joint_T[..., :D, :D],
                out_ts=cur_joint_T[..., :D, D],
            )

        return joint_Ts
