import collections
import typing

import torch
from beartype import beartype

from .. import utils


@beartype
class RadialFunc:
    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def scale_variant(self) -> bool:
        raise NotImplementedError()

    @property
    def min_degree(self) -> int:
        raise NotImplementedError()

    def state_dict(self) -> collections.OrderedDict[str, object]:
        raise NotImplementedError()

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        raise NotImplementedError()

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


@beartype
class LinearRadialFunc(RadialFunc):
    @property
    def name(self) -> str:
        return "linear"

    @property
    def scale_variant(self) -> bool:
        return False

    @property
    def min_degree(self) -> int:
        return 0

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict()

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        assert state_dict["name"] == self.name
        return

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        return -dist


@beartype
class InverseQuadraticRadialFunc(RadialFunc):
    def __init__(self, eps: float):
        assert 0 < eps
        self.eps = eps

    @property
    def name(self) -> str:
        return "inverse_quadratic"

    @property
    def scale_variant(self) -> bool:
        return True

    @property
    def min_degree(self) -> int:
        return 0

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("name", self.name), ("eps", self.eps)])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        assert state_dict["name"] == self.name
        self.eps = state_dict["eps"]

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        k = self.eps**2
        return k / (dist.square() + k)


@beartype
class CubicRadialFunc(RadialFunc):
    @property
    def name(self) -> str:
        return "cubic"

    @property
    def scale_variant(self) -> bool:
        return False

    @property
    def min_degree(self) -> int:
        return 1

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict()

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        assert state_dict["name"] == self.name
        return

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        return dist.pow(3)


@beartype
class QuinticRadialFunc(RadialFunc):
    @property
    def name(self) -> str:
        return "quintic"

    @property
    def scale_variant(self) -> bool:
        return False

    @property
    def min_degree(self) -> int:
        return 2

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict()

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        assert state_dict["name"] == self.name
        return

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        return -dist.pow(5)


@beartype
class MultiquadricRadialFunc(RadialFunc):
    def __init__(self, eps: float):
        assert 0 < eps
        self.eps = eps

    @property
    def name(self) -> str:
        return "multiquadric"

    @property
    def scale_variant(self) -> bool:
        return True

    @property
    def min_degree(self) -> int:
        return 0

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("name", self.name), ("eps", self.eps)])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        assert state_dict["name"] == self.name
        self.eps = state_dict["eps"]

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        k = self.eps**2
        return -k * (dist.square() + k).sqrt()


@beartype
class InverseMultiquadricRadialFunc(RadialFunc):
    def __init__(self, eps: float):
        assert 0 < eps
        self.eps = eps

    @property
    def name(self) -> str:
        return "inverse_multiquadric"

    @property
    def scale_variant(self) -> bool:
        return True

    @property
    def min_degree(self) -> int:
        return 0

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("name", self.name), ("eps", self.eps)])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        assert state_dict["name"] == self.name
        self.eps = state_dict["eps"]

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        k = self.eps**2
        return k * (dist.square() + k).rsqrt()


@beartype
class GaussianRadialFunc(RadialFunc):
    def __init__(self, eps: float):
        assert 0 < eps
        self.eps = eps

    @property
    def name(self) -> str:
        return "gaussian"

    @property
    def scale_variant(self) -> bool:
        return True

    @property
    def min_degree(self) -> int:
        return 0

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict([
            ("name", self.name), ("eps", self.eps)])

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        assert state_dict["name"] == self.name
        self.eps = state_dict["eps"]

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(-dist.square() / self.eps**2)


@beartype
class ThinPlateSplineRadialFunc(RadialFunc):
    @property
    def name(self) -> str:
        return "thin_plate_spline"

    @property
    def scale_variant(self) -> bool:
        return False

    @property
    def min_degree(self) -> int:
        return 1

    def state_dict(self) -> collections.OrderedDict[str, object]:
        return collections.OrderedDict()

    def load_state_dict(self, state_dict: typing.Mapping[str, object]) -> None:
        assert state_dict["name"] == self.name
        return

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        sq_dist = dist.square().clamp(utils.EPS[dist.dtype], None)
        return 0.5 * sq_dist * torch.log(sq_dist)
