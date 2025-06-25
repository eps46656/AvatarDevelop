from __future__ import annotations

import collections
import itertools
import math
import typing

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

    ret = torch.zeros((math.comb(D + DEG, D), D), dtype=torch.int32)

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
    interior: bool,
    d_pos: torch.Tensor,  # [N, D]
    d_val: torch.Tensor,  # [N, C]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
    degree: int,  # DEG
    smoothness: float,
) -> tuple[
    torch.Tensor,  # poly_power[P, D]
    torch.Tensor,  # w[N, C]
    torch.Tensor,  # c[P, C]
]:
    N, D, C, DEG = -1, -2, -3, degree

    N, D, C = utils.check_shapes(
        d_pos, (N, D),
        d_val, (N, C),
    )

    assert 0 <= DEG

    d_pos = d_pos.detach()
    d_val = d_val.detach()

    device = d_pos.device

    P = math.comb(D + DEG, D)

    poly_power = make_monomial_power(D, DEG).to(device)
    # [P, D]

    kernel_dist = kernel(utils.vec_norm(
        d_pos[:, None, :] - d_pos[None, :, :]))
    # [N, N]

    if interior:
        kernel_dist = kernel_dist / kernel_dist.sum(-1, True)

    A = utils.empty(like=d_pos, shape=(N + P, N + P))
    A[:N, :N] = \
        utils.eye(like=d_pos, shape=(N, N)) * smoothness + kernel_dist
    A[:N, N:] = make_poly_value(d_pos, poly_power)  # [N, P]
    A[N:, :N] = A[:N, N:].T  # [P, N]
    A[N:, N:] = 0

    b = utils.empty(like=d_pos, shape=(N + P, C))
    b[:N] = d_val
    b[N:] = 0

    coeff = torch.linalg.lstsq(A, b).solution
    # [N + P, C]

    w = coeff[:N].to(device)  # [N, C]
    c = coeff[N:].to(device)  # [P, C]

    return poly_power, w, c


@beartype
def query(
    *,
    interior: bool,
    d_pos: typing.Optional[torch.Tensor] = None,  # [N, D]
    poly_power: typing.Optional[torch.Tensor] = None,  # [P, D]
    w: torch.Tensor,  # [N, C]
    c: torch.Tensor,  # [P, C]
    kernel: typing.Optional[
        typing.Callable[[torch.Tensor], torch.Tensor]] = None,

    q_pos: typing.Optional[torch.Tensor] = None,  # [..., D]
    q_kernel_dist: typing.Optional[torch.Tensor] = None,  # [..., N]
    q_poly_val: typing.Optional[torch.Tensor] = None,  # [..., P]
) -> tuple[
    torch.Tensor,  # q_val[..., C]
    torch.Tensor,  # q_kernel_dist[..., N]
    torch.Tensor,  # q_poly_val[..., P]
]:
    N, P, D, C = -1, -2, -3, -4

    N, P, D, C = utils.check_shapes(
        d_pos, (N, D),
        w, (N, C),
        c, (P, C),
        q_pos, (..., D),
    )

    if q_kernel_dist is None:
        assert d_pos is not None
        assert kernel is not None
        assert q_pos is not None

        q_kernel_dist = kernel(utils.vec_norm(
            q_pos[..., None, :] - d_pos))
        # [..., N]

        if interior:
            q_kernel_dist = q_kernel_dist / \
                q_kernel_dist.sum(-1, True)

    assert q_kernel_dist.shape == (*q_pos.shape[:-1], N)

    w_val = utils.einsum(
        "nc, ...n -> ...c",
        w,  # [N, C]
        q_kernel_dist,  # [..., N]
    )
    # [..., C]

    if q_poly_val is None:
        assert poly_power is not None
        assert q_pos is not None

        q_poly_val = make_poly_value(q_pos, poly_power)
        # [..., P]

    assert q_poly_val.shape == (*q_pos.shape[:-1], P)

    c_val = utils.einsum(
        "pc, ...p -> ...c",
        c,  # [P, C]
        q_poly_val,  # [..., P]
    )
    # [..., C]

    return w_val + c_val, q_kernel_dist, q_poly_val


@beartype
class RBFInterpolator(torch.nn.Module):
    def __init__(
        self,
        interior: bool,
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

        self.interior = interior
        self.data_pos = data_pos
        self.poly_power = poly_power
        self.w = w
        self.c = c
        self.kernel = kernel

    @staticmethod
    def from_data_point(
        interior: bool,
        d_pos: torch.Tensor,  # [N, D]
        d_val: torch.Tensor,  # [N, C]
        kernel: RadialFunc,
        degree: int = -1,  # DEG
        smoothness: float = 0.0,
    ) -> RBFInterpolator:
        if degree < 0:
            degree = kernel.min_degree

        assert kernel.min_degree <= degree

        poly_power, w, c = build(
            interior, d_pos, d_val, kernel, degree, smoothness)

        return RBFInterpolator(interior, d_pos, poly_power, w, c, kernel)

    @staticmethod
    def from_state_dict(
        state_dict: typing.Mapping[str, object],
        device: typing.Optional[torch.device] = None,
    ) -> RBFInterpolator:
        interior = state_dict["interior"]

        data_pos = utils.deserialize_tensor(
            state_dict["data_pos"], device=device)

        poly_power = utils.deserialize_tensor(
            state_dict["poly_power"], device=device)

        w = utils.deserialize_tensor(
            state_dict["w"], device=device)

        c = utils.deserialize_tensor(
            state_dict["c"], device=device)

        kernel = state_dict["kernel"]

        return RBFInterpolator(interior, data_pos, poly_power, w, c, kernel)

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("interior", self.interior),
            ("data_pos", utils.serialize_tensor(self.data_pos)),
            ("poly_power", utils.serialize_tensor(self.poly_power)),
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
        *,
        q_pos: typing.Optional[torch.Tensor] = None,  # [..., D]
        q_kernel_dist: typing.Optional[torch.Tensor] = None,  # [..., N]
        q_poly_val: typing.Optional[torch.Tensor] = None,  # [..., P]
    ) -> tuple[
        torch.Tensor,  # q_val[..., C]
        torch.Tensor,  # q_kernel_dist[..., N]
        torch.Tensor,  # q_poly_val[..., P]
    ]:
        return query(
            interior=self.interior,
            d_pos=self.data_pos,
            poly_power=self.poly_power,
            w=self.w,
            c=self.c,
            kernel=self.kernel,

            q_pos=q_pos,
            q_kernel_dist=q_kernel_dist,
            q_poly_val=q_poly_val,
        )
