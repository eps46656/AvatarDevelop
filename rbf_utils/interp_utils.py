from __future__ import annotations

import collections
import itertools
import math
import typing

import numpy as np
import torch
from beartype import beartype

from .. import utils
from .radial_func import RadialFunc


@beartype
def make_monomial_power(
    dim: int,  # D
    degree: int,  # DEG
) -> torch.Tensor:  # [P, D]
    D = dim
    DEG = degree

    assert 0 <= D
    assert 0 <= DEG

    ret = torch.zeros((math.comb(DEG + D, D), D), dtype=torch.int16)

    i = 0

    for deg in range(degree + 1):
        for comb in itertools.combinations_with_replacement(range(D), deg):
            for j in comb:
                ret[i, j] += 1
            i += 1

    return ret


@beartype
def make_poly_value(
    x: torch.Tensor,  # [..., D]
    poly_power: torch.Tensor,  # [P, D]
) -> torch.Tensor:  # [..., P]
    P, D = -1, -2

    P, D = utils.check_shapes(
        x, (..., D),
        poly_power, (P, D)
    )

    # [..., 1, D] ** [P, D] = [..., P, D]
    # [..., P]
    return x[..., None, :].pow(poly_power).prod(-1)


@beartype
def build(
    data_pos: torch.Tensor,  # [N, D]
    data_val: torch.Tensor,  # [N, C]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
    degree: int,  # DEG
    smoothness: float,
) -> tuple[
    torch.Tensor,  # poly_power[P, D]
    torch.Tensor,  # w[N, C]
    torch.Tensor,  # p[P, C]
]:
    N, D, C, DEG = -1, -2, -3, degree

    N, D, C = utils.check_shapes(
        data_pos, (N, D),
        data_val, (N, C),
    )

    assert 0 <= DEG

    data_pos = data_pos.detach()
    data_val = data_val.detach()

    device = data_pos.device

    P = math.comb(D + DEG, D)

    poly_power = make_monomial_power(D, DEG).to(data_pos.device)
    # [P, D]

    A = utils.empty(like=data_pos, shape=(N + P, N + P))
    A[:N, :N] = utils.eye(like=data_pos, shape=(N, N)) * smoothness + \
        kernel(utils.vec_norm(data_pos[:, None, :] - data_pos[None, :, :]))
    A[:N, N:] = make_poly_value(data_pos, poly_power)  # [N, P]
    A[N:, :N] = A[:N, N:].T  # [P, N]
    A[N:, N:] = 0

    b = utils.empty(like=data_pos, shape=(N + P, C))
    b[:N] = data_val
    b[N:] = 0

    # A @ [w, lambda_const, *lambda_1, ...] = b

    coeff = torch.linalg.solve(A, b)
    # [(w, lambda_const, *lambda_1, ...), C]
    # [N + P, C]

    w = coeff[:N].to(device)  # [N, C]
    c = coeff[N:].to(device)  # [P, C]

    return poly_power, w, c


@beartype
def query(
    data_pos: torch.Tensor,  # [N, D]
    poly_power: torch.Tensor,  # [P, D]
    w: torch.Tensor,  # [N, C]
    c: torch.Tensor,  # [P, C]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
    point_pos: torch.Tensor,  # [..., D]
) -> torch.Tensor:  # [..., C]
    N, P, D, C = -1, -2, -3, -4

    N, P, D, C = utils.check_shapes(
        data_pos, (N, D),
        w, (N, C),
        c, (P, C),
    )

    w_val = utils.einsum(
        "nc, ...n -> ...c",
        w,  # [N, C]

        kernel(utils.vec_norm(point_pos[..., None, :] - data_pos[None, :, :])),
        # [..., N]
    )
    # [..., C]

    c_val = utils.einsum(
        "pc, ...p -> ...c",
        c,  # [P, C]

        make_poly_value(point_pos, poly_power),
        # [..., P]
    )
    # [..., C]

    return w_val + c_val


@beartype
class RBFInterpolator(torch.nn.Module):
    def __init__(
        self,
        data_pos: torch.Tensor,  # [N, D]
        poly_power: torch.Tensor,  # [P, D]
        w: torch.Tensor,  # [N, C]
        c: torch.Tensor,  # [P, C]
        kernel: RadialFunc,
    ):
        super().__init__()

        N, P, D, C = -1, -2, -3, -4

        N, P, D, C = utils.check_shapes(
            data_pos, (N, D),
            w, (N, C),
            c, (P, C),
        )

        self.data_pos = data_pos
        self.poly_power = poly_power
        self.w = w
        self.c = c
        self.kernel = kernel

    @staticmethod
    def from_data_point(
        data_pos: torch.Tensor,  # [N, D]
        data_val: torch.Tensor,  # [N, C]
        kernel: RadialFunc,
        degree: int = -1,  # DEG
        smoothness: float = 1.0,
    ) -> RBFInterpolator:
        if degree < 0:
            degree = kernel.min_degree

        assert kernel.min_degree <= degree

        poly_power, w, c = build(
            data_pos, data_val, kernel, degree, smoothness)

        return RBFInterpolator(data_pos, poly_power, w, c, kernel)

    @staticmethod
    def from_state_dict(
        state_dict: typing.Mapping[str, object],
        device: typing.Optional[torch.device] = None,
    ) -> RBFInterpolator:
        data_pos = utils.deserialize_tensor(
            state_dict["data_pos"], device=device)

        poly_power = utils.deserialize_tensor(
            state_dict["poly_power"], device=device)

        w = utils.deserialize_tensor(
            state_dict["w"], device=device)

        c = utils.deserialize_tensor(
            state_dict["c"], device=device)

        kernel = state_dict["kernel"]

        return RBFInterpolator(data_pos, poly_power, w, c, kernel)

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("data_pos", utils.serialize_tensor(self.data_pos)),
            ("poly_power", self.poly_power),
            ("w", utils.serialize_tensor(self.w)),
            ("c", utils.serialize_tensor(self.c)),
            ("kernel", self.kernel),
        ])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        self.data_pos = utils.deserialize_tensor(
            state_dict["data_pos"], device=self.data_pos.device)

        self.poly_power = utils.deserialize_tensor(
            state_dict["poly_power"], device=self.poly_power.device)

        self.w = utils.deserialize_tensor(
            state_dict["w"], device=self.w.device)

        self.c = utils.deserialize_tensor(
            state_dict["c"], device=self.c.device)

        self.kernel = state_dict["kernel"]

    def to(self, *args, **kwargs) -> RBFInterpolator:
        self.data_pos = self.data_pos.to(*args, **kwargs)
        self.poly_power = self.poly_power.to(*args, **kwargs)
        self.w = self.w.to(*args, **kwargs)
        self.c = self.c.to(*args, **kwargs)

        return self

    def forward(
        self,
        point_pos: torch.Tensor,  # [..., D]
    ) -> torch.Tensor:  # [..., C]
        return query(
            self.data_pos,
            self.poly_power, self.w, self.c,
            self.kernel, point_pos,
        )
