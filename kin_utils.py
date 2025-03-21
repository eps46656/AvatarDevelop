import dataclasses
import heapq

import torch
from beartype import beartype

from . import utils


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


@beartype
def GetJointRTs(
    kin_tree: KinTree,
    pose_rs: torch.Tensor,  # [..., J, D, D]
    pose_ts: torch.Tensor,  # [..., J, D]
) -> torch.Tensor:  # joint_Ts[..., J, D+1, D+1]:
    J = kin_tree.joints_cnt

    D, = utils.CheckShapes(
        pose_rs, (..., J, -1, -1),
        pose_ts, (..., J, -1),
    )

    batch_shape = utils.BroadcastShapes(
        pose_rs.shape[:-3],
        pose_ts.shape[:-2],
    )

    joint_Ts = torch.empty(
        batch_shape + (J, D+1, D+1),
        dtype=torch.promote_types(pose_rs.dtype, pose_ts.dtype),
        device=pose_rs.device,
    )
    # [..., J, D+1, D+1]

    joint_Ts[..., :, D, :D] = 0
    joint_Ts[..., :, D, D] = 1

    joint_Ts[..., kin_tree.root, :D, :D] = pose_rs[..., kin_tree.root, :, :]
    joint_Ts[..., kin_tree.root, :D, D] = pose_ts[..., kin_tree.root, :]

    for u in kin_tree.joints_tp[1:]:
        p = kin_tree.parents[u]

        cur_joint_T = joint_Ts[..., u, :, :]  # [..., D+1, D+1]
        parent_joint_T = joint_Ts[..., p, :, :]  # [... D+1, D+1]

        utils.MergeRT(
            parent_joint_T[..., :D, :D],
            parent_joint_T[..., :D, D],
            pose_rs[..., u, :, :],
            pose_ts[..., u, :],
            out_rs=cur_joint_T[..., :D, :D],
            out_ts=cur_joint_T[..., :D, D],
        )

    return joint_Ts
