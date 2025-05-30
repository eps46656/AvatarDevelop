from __future__ import annotations

import collections
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

        s = (*utils.broadcast_shapes(
            trans.shape[:-2],
            inv_trans.shape[:-2],
        ), 4, 4)

        self.trans = trans.expand(s)
        self.inv_trans = inv_trans.expand(s)

    @staticmethod
    def from_matching(
        dirs: str,
        pos: torch.Tensor = utils.ORIGIN,  # [..., 3]
        vecs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            utils.X_AXIS,
            utils.Y_AXIS,
            utils.Z_AXIS,
        ),  # [..., 3]
        *,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> ObjectTransform:
        assert len(dirs) == 3

        dirs = dirs.upper()

        assert dirs.count("F") + dirs.count("B") == 1
        assert dirs.count("U") + dirs.count("D") == 1
        assert dirs.count("L") + dirs.count("R") == 1

        vec_a, vec_b, vec_c = vecs

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

        if device is None:
            device = utils.check_devices(pos, vec_a, vec_b, vec_c)

        if dtype is None:
            dtype = utils.promote_dtypes(pos, vec_a, vec_b, vec_c)

        trans = torch.empty(
            (*batch_shape, 4, 4), dtype=dtype, device=device)

        for dir, vec in zip(dirs, vecs):
            match dir:
                case "F": trans[..., :3, 0] = +vec
                case "B": trans[..., :3, 0] = -vec
                case "U": trans[..., :3, 1] = +vec
                case "D": trans[..., :3, 1] = -vec
                case "L": trans[..., :3, 2] = +vec
                case "R": trans[..., :3, 2] = -vec

        trans[..., :3, 3] = pos
        trans[..., 3, :3] = 0
        trans[..., 3, 3] = 1

        inv_trans = trans.inverse()

        return ObjectTransform(trans, inv_trans)

    @staticmethod
    def from_state_dict(
        state_dict: collections.OrderedDict[str, torch.Tensor],
        dtype: typing.Optional[torch.dtype] = None,
        device: typing.Optional[torch.device] = None,
    ) -> ObjectTransform:
        return ObjectTransform(
            utils.deserialize_tensor(
                state_dict["trans"], dtype, device),
            utils.deserialize_tensor(
                state_dict["inv_trans"], dtype, device),
        )

    @property
    def shape(self) -> torch.Size:
        return self.trans.shape[:-2]

    @property
    def dtype(self) -> torch.dtype:
        return self.trans.dtype

    @property
    def device(self) -> torch.device:
        return self.trans.device

    def to(self, *args, **kwargs) -> ObjectTransform:
        return ObjectTransform(
            self.trans.to(*args, **kwargs), self.inv_trans.to(*args, **kwargs))

    def __str__(self) -> str:
        return "\n".join([
            f"F: <{self.vec_f}>",
            f"B: <{self.vec_b}>",
            f"U: <{self.vec_u}>",
            f"D: <{self.vec_d}>",
            f"L: <{self.vec_l}>",
            f"R: <{self.vec_r}>",
        ])

    def __repr__(self) -> str:
        return str(self)

    @property
    def pos(self) -> torch.Tensor:
        return self.trans[..., :3, 3]

    def vec(self, dir: str) -> torch.Tensor:  # [..., 3]
        match dir:
            case "F": return self.vec_f
            case "B": return self.vec_b
            case "U": return self.vec_u
            case "D": return self.vec_d
            case "L": return self.vec_l
            case "R": return self.vec_r
            case _: raise utils.MismatchException()

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

    def __getitem__(self, idx) -> ObjectTransform:
        if not isinstance(idx, tuple):
            idx = (idx,)

        return ObjectTransform(
            self.trans[*idx, :, :],
            self.inv_trans[*idx, :, :],
        )

    def state_dict(self) -> collections.OrderedDict[str, torch.Tensor]:
        return collections.OrderedDict([
            ("trans", utils.serialize_tensor(self.trans)),
            ("inv_trans", utils.serialize_tensor(self.inv_trans)),
        ])

    def load_state_dict(
        self,
        state_dict: collections.OrderedDict[str, torch.Tensor],
    ) -> None:
        self.trans = utils.deserialize_tensor(
            state_dict["trans"], self.dtype, self.device)

        self.inv_trans = utils.deserialize_tensor(
            state_dict["inv_trans"], self.dtype, self.device)

    def expand(self, shape) -> ObjectTransform:
        s = (*shape, 4, 4)

        return ObjectTransform(
            self.trans.expand(s),
            self.inv_trans.expand(s),
        )

    def get_trans_to(self, dst: ObjectTransform) -> torch.Tensor:
        # self: object <-> coord_a
        # dst: object <-> coord_b

        # return: coord_a -> coord_b

        return utils.mat_mul(dst.trans, self.inv_trans)

    def collapse(
        self,
        trans: torch.Tensor,  # [..., 4, 4]
    ) -> ObjectTransform:
        # self: object <-> coord_a
        # trans: coord_a -> coord_b

        # return: object <-> coord_b

        new_trans = utils.mat_mul(trans, self.trans)

        return ObjectTransform(new_trans, new_trans.inverse())

    def inverse(self) -> ObjectTransform:
        return ObjectTransform(self.inv_trans, self.trans)
