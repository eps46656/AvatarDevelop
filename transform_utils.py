import typing

import torch
from beartype import beartype

from . import utils


# Preresent a abstract transformatnio between object and a coordinate
@beartype
class ObjectTransform:
    def __init__(
        self,
        trans: torch.Tensor,  # [..., 4, 4]
        inv_trans: torch.Tensor,  # [..., 4, 4]
    ):
        utils.CheckShapes(
            trans, (..., 4, 4),
            inv_trans, (..., 4, 4),
        )

        s = utils.BroadcastShapes(
            trans.shape[:-2],
            inv_trans.shape[:-2],
        ) + (4, 4)

        self.trans = trans.expand(s)
        self.inv_trans = inv_trans.expand(s)

    @staticmethod
    def FromMatching(
        dirs: str | tuple[utils.Dir, utils.Dir, utils.Dir],
        pos: torch.Tensor = utils.ORIGIN,  # [..., 3]
        vecs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            utils.X_AXIS,
            utils.Y_AXIS,
            utils.Z_AXIS,
        ),  # [..., 3]
    ):
        if isinstance(dirs, str):
            assert len(dirs) == 3
            dirs = (utils.Dir[dirs[0]], utils.Dir[dirs[1]], utils.Dir[dirs[2]])

        assert dirs.count(utils.Dir.F) + dirs.count(utils.Dir.B) == 1
        assert dirs.count(utils.Dir.U) + dirs.count(utils.Dir.D) == 1
        assert dirs.count(utils.Dir.L) + dirs.count(utils.Dir.R) == 1

        utils.CheckShapes(
            pos, (..., 3),
            vecs[0], (..., 3),
            vecs[1], (..., 3),
            vecs[2], (..., 3),
        )

        batch_shape = utils.BroadcastShapes(
            pos.shape[:-1],
            vecs[0].shape[:-1],
            vecs[1].shape[:-1],
            vecs[2].shape[:-1],
        )

        s = batch_shape + (3,)

        f_vec, u_vec, l_vec = None, None, None

        for dir, vec in zip(dirs, vecs):
            match dir:
                case utils.Dir.F: f_vec = +vec
                case utils.Dir.B: f_vec = -vec
                case utils.Dir.U: u_vec = +vec
                case utils.Dir.D: u_vec = -vec
                case utils.Dir.L: l_vec = +vec
                case utils.Dir.R: l_vec = -vec

        trans = torch.empty(batch_shape + (4, 4),
                            dtype=pos.dtype, device=pos.device)

        trans[..., :3, 0] = f_vec.expand(s)
        trans[..., :3, 1] = u_vec.expand(s)
        trans[..., :3, 2] = l_vec.expand(s)
        trans[..., :3, 3] = pos.expand(s)
        trans[..., 3, :3] = 0
        trans[..., 3, 3] = 1

        inv_trans = trans.inverse()

        return ObjectTransform(trans, inv_trans)

    def GetPos(self) -> torch.Tensor:
        return self.trans[..., :3, 3]

    def GetVec(self, dir: utils.Dir) -> torch.Tensor:  # [..., 3]
        match dir:
            case utils.Dir.F: return self.GetFVec()
            case utils.Dir.B: return self.GetBVec()
            case utils.Dir.U: return self.GetUVec()
            case utils.Dir.D: return self.GetDVec()
            case utils.Dir.L: return self.GetLVec()
            case utils.Dir.R: return self.GetRVec()

        assert False, f"Unknown direction {dir}."

    def GetFVec(self) -> torch.Tensor:  # [..., 3]
        return +self.trans[..., :3, 0]

    def GetBVec(self) -> torch.Tensor:  # [..., 3]
        return -self.trans[..., :3, 0]

    def GetUVec(self) -> torch.Tensor:  # [..., 3]
        return +self.trans[..., :3, 1]

    def GetDVec(self) -> torch.Tensor:  # [..., 3]
        return -self.trans[..., :3, 1]

    def GetLVec(self) -> torch.Tensor:  # [..., 3]
        return +self.trans[..., :3, 2]

    def GetRVec(self) -> torch.Tensor:  # [..., 3]
        return -self.trans[..., :3, 2]

    def GetTransTo(self, dst: typing.Self):
        # self: object <-> coord_a
        # dst: object <-> coord_b

        # return: coord_a -> coord_b

        return dst.trans @ self.inv_trans

    def GetCollapsed(
        self,
        trans: torch.Tensor,  # [..., 4, 4]
    ):
        # self: object <-> coord_a
        # trans: coord_a -> coord_b

        # return: object <-> coord_b

        return ObjectTransform(
            self.trans @ trans,
            trans.inverse() @ self.inv_trans)
