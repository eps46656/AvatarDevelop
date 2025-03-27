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
        utils.check_shapes(
            trans, (..., 4, 4),
            inv_trans, (..., 4, 4),
        )

        s = utils.broadcast_shapes(
            trans.shape[:-2],
            inv_trans.shape[:-2],
        ) + (4, 4)

        self.trans = trans.expand(s)
        self.inv_trans = inv_trans.expand(s)

    @staticmethod
    def from_matching(
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

        utils.check_shapes(
            pos, (..., 3),
            vecs[0], (..., 3),
            vecs[1], (..., 3),
            vecs[2], (..., 3),
        )

        batch_shape = utils.broadcast_shapes(
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

    @property
    def shape(self) -> torch.Size:
        return self.trans.shape[:-2]

    @property
    def device(self) -> torch.device:
        return self.trans.device

    def to(self, *args, **kwargs) -> typing.Self:
        return ObjectTransform(
            self.trans.to(*args, **kwargs), self.inv_trans.to(*args, **kwargs))

    def __str__(self):
        return "\n".join([
            f"F: <{self.vec_f}>",
            f"B: <{self.vec_b}>",
            f"U: <{self.vec_u}>",
            f"D: <{self.vec_d}>",
            f"L: <{self.vec_l}>",
            f"R: <{self.vec_r}>",
        ])

    def __repr__(self):
        return str(self)

    @property
    def pos(self) -> torch.Tensor:
        return self.trans[..., :3, 3]

    def vec(self, dir: utils.Dir) -> torch.Tensor:  # [..., 3]
        match dir:
            case utils.Dir.F: return self.vec_f
            case utils.Dir.B: return self.vec_b
            case utils.Dir.U: return self.vec_u
            case utils.Dir.D: return self.vec_d
            case utils.Dir.L: return self.vec_l
            case utils.Dir.R: return self.vec_r

        assert False, f"Unknown direction {dir}."

    @property
    def vec_f(self) -> torch.Tensor:  # [..., 3]
        return +self.trans[..., :3, 0]

    @property
    def vec_b(self) -> torch.Tensor:  # [..., 3]
        return -self.trans[..., :3, 0]

    @property
    def vec_u(self) -> torch.Tensor:  # [..., 3]
        return +self.trans[..., :3, 1]

    @property
    def vec_d(self) -> torch.Tensor:  # [..., 3]
        return -self.trans[..., :3, 1]

    @property
    def vec_l(self) -> torch.Tensor:  # [..., 3]
        return +self.trans[..., :3, 2]

    @property
    def vec_r(self) -> torch.Tensor:  # [..., 3]
        return -self.trans[..., :3, 2]

    def get_trans_to(self, dst: typing.Self):
        # self: object <-> coord_a
        # dst: object <-> coord_b

        # return: coord_a -> coord_b

        return dst.trans @ self.inv_trans

    def collapse(
        self,
        trans: torch.Tensor,  # [..., 4, 4]
    ):
        # self: object <-> coord_a
        # trans: coord_a -> coord_b

        # return: object <-> coord_b

        new_trans = self.trans @ trans

        return ObjectTransform(new_trans, new_trans.inverse())

    def inverse(self):
        return ObjectTransform(self.inv_trans, self.trans)

    def expand(self, shape):
        s = tuple(shape) + (4, 4)

        return ObjectTransform(
            self.trans.expand(s),
            self.inv_trans.expand(s),
        )

    def batch_get(self, batch_idxes: tuple[torch.Tensor, ...]):
        assert len(batch_idxes) == len(self.shape)

        return ObjectTransform(
            self.trans[batch_idxes],
            self.inv_trans[batch_idxes],
        )
