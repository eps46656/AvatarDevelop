from __future__ import annotations

import collections
import enum
import typing

import einops
import torch
from beartype import beartype

from . import utils


@beartype
def build(
    data_pos: torch.Tensor,  # [N, D]
    data_val: torch.Tensor,  # [N, C]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
    degree: int,  # DEG
) -> tuple[
    torch.Tensor,  # w[C, N]
    torch.Tensor,  # c[C]
    torch.Tensor,  # p[C, DEG, D]
]:
    DEG = degree

    N, D, C = -1, -2, -3

    N, D, C = utils.check_shapes(
        data_pos, (N, D),
        data_val, (N, C),
    )

    assert 0 <= DEG

    K = 1 + D * DEG

    A = torch.empty(
        (C, N + K, N + K), dtype=data_pos.dtype, device=data_pos.device)
    # [N + K, N + K]

    A[:N, :N] = kernel(data_pos[: None, :] - data_pos[None, :, :])

    A[:N, N] = 1
    A[N, :N] = 1

    powers = torch.arange(
        1, DEG + 1, dtype=data_pos.dtype, device=data_pos.device)
    # [DEG]

    data_pos[:, None, :].pow(
        powers[None, :, None],  # [DEG] -> [1, DEG, 1]
        out=A[:N, N + 1:].view(N, DEG, D)
    )
    # ([N, D] -> [N, 1, D]) ** ([DEG] -> [1, DEG, 1])
    # ->
    # ([N, DEG * D] -> [N, DEG, D])

    A[N + 1:, :N] = A[:N, N + 1:].T

    A[N:, N:] = 0

    b = torch.empty(
        (C, N + K, 1), dtype=data_pos.dtype, device=data_pos.device)

    b[:, :N, 0] = data_val.transpose(0, 1)

    b[:, N:, 0] = 0

    # A @ [w, lambda_const, *lambda_1, ...] = b

    coeffs = torch.linalg.lstsq(
        A[None, ...].expand(C, N + K, N + K), b).solution
    # [C, (w, lambda_const, *lambda_1, ...]

    w = coeffs[:, :N]  # [C, N]
    c = coeffs[:, N]  # [C]
    p = coeffs[:, N + 1:].view(C, DEG, D)  # [C, DEG, D]

    return w, c, p


@beartype
def query(
    data_pos: torch.Tensor,  # [N, D]
    w: torch.Tensor,  # [C, N]
    c: torch.Tensor,  # [C]
    p: torch.Tensor,  # [C, DEG, D]
    kernel: typing.Callable[[torch.Tensor], torch.Tensor],
    point_pos: torch.Tensor,  # [..., D]
):
    N, C, D, DEG = -1, -2, -3, -4

    N, C, D, DEG = utils.check_shapes(
        w, (C, N),
        c, (C,),
        p, (C, DEG, D),
    )

    w_val = torch.einsum(
        "cn, ...n -> ...c",
        w,  # [C, N]
        kernel(point_pos[..., None, :] - data_pos),
        # [..., N]
    )
    # [..., C]

    powers = torch.arange(
        1, DEG + 1, dtype=point_pos.dtype, device=point_pos.device)
    # [DEG]

    p_val = torch.einsum(
        "crd, ...rd -> ...c",

        p,  # [C, DEG, D]

        point_pos[..., None, :].pow(powers[:, None]),
        # ([..., D] -> [..., 1, D]) ** ([DEG] -> [DEG, 1]) -> [..., DEG, D]
    )

    return w_val + c + p_val


@beartype
class RadialFuncBase:
    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def scale_variant(self) -> bool:
        raise NotImplementedError()

    @property
    def min_degree(self) -> int:
        raise NotImplementedError()

    def __call__(self, dir: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


@beartype
class LinearRadialFunc(RadialFuncBase):
    @property
    def name(self) -> str:
        return "linear"

    @property
    def scale_variant(self) -> bool:
        return False

    @property
    def min_degree(self) -> int:
        return 0

    def __call__(self, dir: torch.Tensor) -> torch.Tensor:
        return utils.vec_norm(dir)


@beartype
class InverseQuadraticKernel(RadialFuncBase):
    @property
    def name(self) -> str:
        return "inverse_quadratic"

    @property
    def scale_variant(self) -> bool:
        return True

    @property
    def min_degree(self) -> int:
        return 0

    def __call__(self, dir: torch.Tensor) -> torch.Tensor:
        return 1 / (utils.vec_sq_norm(dir) + 1)


@beartype
class CubicKernel(RadialFuncBase):
    @property
    def name(self) -> str:
        return "cubic"

    @property
    def scale_variant(self) -> bool:
        return False

    @property
    def min_degree(self) -> int:
        return 1

    def __call__(self, dir: torch.Tensor) -> torch.Tensor:
        return utils.vec_norm(dir)**3


@beartype
class QuinticKernel(RadialFuncBase):
    @property
    def name(self) -> str:
        return "quintic"

    @property
    def scale_variant(self) -> bool:
        return False

    @property
    def min_degree(self) -> int:
        return 2

    def __call__(self, dir: torch.Tensor) -> torch.Tensor:
        return utils.vec_norm(dir)**5


@beartype
class MultiquadricKernel(RadialFuncBase):
    @property
    def name(self) -> str:
        return "multiquadric"

    @property
    def scale_variant(self) -> bool:
        return True

    @property
    def min_degree(self) -> int:
        return 0

    def __call__(self, dir: torch.Tensor) -> torch.Tensor:
        return (utils.vec_sq_norm(dir) + 1).sqrt()


@beartype
class InverseMultiquadricKernel(RadialFuncBase):
    @property
    def name(self) -> str:
        return "inverse_multiquadric"

    @property
    def scale_variant(self) -> bool:
        return True

    @property
    def min_degree(self) -> int:
        return 0

    def __call__(self, dir: torch.Tensor) -> torch.Tensor:
        return (utils.vec_sq_norm(dir) + 1).rsqrt()


@beartype
class GaussianKernel(RadialFuncBase):
    @property
    def name(self) -> str:
        return "gaussian"

    @property
    def scale_variant(self) -> bool:
        return True

    @property
    def min_degree(self) -> int:
        return 0

    def __call__(self, dir: torch.Tensor) -> torch.Tensor:
        return torch.exp(-utils.vec_sq_norm(dir))


@beartype
class ThinPlateSplineKernel(RadialFuncBase):
    @property
    def name(self) -> str:
        return "thin_plate_spline"

    @property
    def scale_variant(self) -> bool:
        return False

    @property
    def min_degree(self) -> int:
        return 1

    def __call__(self, dir: torch.Tensor) -> torch.Tensor:
        sq_dist = utils.vec_sq_norm(dir).clamp(1e-7, None)
        return sq_dist * torch.log(sq_dist)


class RBFInterpolator(torch.nn.Module):
    def __init__(
        self,
        data_pos: torch.Tensor,  # [N, D]
        w,  # [C, N]
        c,  # [C]
        p,  # [C, DEG, D]
        kernel: RadialFuncBase,
    ):
        super().__init__()

        N, D, C, DEG = -1, -2, -3, -4

        N, D, C = utils.check_shapes(
            data_pos, (N, D),
            w, (C, N),
            c, (C),
            p, (C, DEG, D),
        )

        self.data_pos = data_pos
        self.w = w
        self.c = c
        self.p = p
        self.kernel = kernel

    @staticmethod
    def from_data_point(
        data_pos: torch.Tensor,  # [N, D]
        data_val: torch.Tensor,  # [N, C]
        kernel_type: KernelTypeEnum,
        degree: int = 0,  # DEG
    ) -> RBFInterpolator:
        kernel = {
            KernelTypeEnum.LINEAR: linear,
            KernelTypeEnum.CUBIC: cubic,
            KernelTypeEnum.THIN_PLATE_SPLINE: thin_plate_spline,
        }[kernel_type]

        w, c, p = build(data_pos, data_val, kernel, degree)

        return RBFInterpolator(data_pos, w, c, p, kernel)

    @staticmethod
    def from_state_dict(state_dict: dict[str, torch.Tensor]) -> RBFInterpolator:
        return RBFInterpolator(
            data_pos=state_dict["data_pos"],
            data_val=state_dict["data_val"],
            kernel=state_dict["kernel"],
            degree=state_dict["degree"],
        )

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return {
            "data_pos": utils.tensor_serialize(self.data_pos),
            "w": utils.tensor_serialize(self.w),
            "c": utils.tensor_serialize(self.c),
            "p": utils.tensor_serialize(self.p),
            "kernel_type": self.kernel_type,
        }

    def forward(
        self,
        point_pos: torch.Tensor,  # [..., D]
    ):
        pass
