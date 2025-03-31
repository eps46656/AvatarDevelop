
import enum

import torch
from beartype import beartype

from . import utils

cubic_interp_coeff_mat = torch.tensor([
    [+0, +6, +0, +0],
    [-2, -3, +6, -1],
    [+3, -6, +3, +0],
    [-1, +3, -3, +1],
], dtype=utils.FLOAT) / 6


@beartype
def cubic_interp(
    ys: torch.Tensor,  # [..., 4]
    x: torch.Tensor,  # [...]
):
    utils.check_shapes(ys, (..., 4))

    cs = (cubic_interp_coeff_mat.to(ys.device) @ ys.unsqueeze(-1)) \
        .squeeze(-1)
    # [..., 4]

    x1 = x
    x2 = x.square()
    x3 = x1 * x2

    return cs[:, 3] * x3 + cs[:, 2] + x2 * cs[:, 1] * x1 + cs[:, 0]


class WrapModeEnum(enum.Enum):
    REPEAT = enum.auto()
    MIRROR_REPEAT = enum.auto()


class InterpModeEnum(enum.Enum):
    NEAREST = enum.auto()
    LINEAR = enum.auto()
    CUBIC = enum.auto()


@beartype
def wrap(
    x: torch.Tensor,
    size: int,
    mode: WrapModeEnum,
):
    assert 0 < size

    match mode:
        case WrapModeEnum.REPEAT:
            return x % size

        case WrapModeEnum.MIRROR_REPEAT:
            x = x % (size * 2 - 2)
            return torch.where(x < size, x, (size - 2) - x)

    assert False, "Unknown warp mode"


@beartype
class TextureSampler:
    wrap_mode_table = {
        WrapModeEnum.MIRROR_REPEAT: "reflection",
    }

    interp_mode_table = [
        {  # 0
        },
        {  # 1
        },
        {  # 2
            InterpModeEnum.NEAREST: "nearest",
            InterpModeEnum.LINEAR: "bilinear",
            InterpModeEnum.CUBIC: "bicubic",
        },
        {  # 3
            InterpModeEnum.NEAREST: "nearest",
            InterpModeEnum.LINEAR: "bilinear",
        },
    ]

    def __init__(self,
                 dim: int,
                 data: torch.Tensor,
                 wrap_mode: WrapModeEnum,
                 interp_mode: InterpModeEnum,
                 ):
        # data[c0, c1, c2, ..., d0, d1, d2]

        assert dim in [2, 3]
        assert dim <= len(data.shape)

        self.__dim = dim

        self.__c_shape = list(data.shape[:-dim])

        self.__data = data.reshape([1, -1] + list(data.shape[-dim:]))
        # [1, c, d0, d1, ...]

        assert wrap_mode in TextureSampler.wrap_mode_table

        self.__wrap_mode = wrap_mode

        assert interp_mode in TextureSampler.interp_mode_table[dim]

        self.__interp_mode = interp_mode

    def Sample(self, points: torch.Tensor):
        # points[n0, n1, n2, ..., self.__dim] = [0, 1]

        dim = self.__dim

        assert points.shape[-1] == dim

        grid = torch.empty(
            points.shape,
            dtype=points.dtype,
            device=points.device,
        )

        for d in range(dim):
            grid[..., dim-1-d] = points[..., d] * 2 - 1

        n_shape = list(points.shape[:-1])

        grid = grid.reshape([1] * dim + [-1, dim])
        # 2d: grid[1, 1, -1, dim]
        # 3d: grid[1, 1,  1, -1, dim]

        ret = torch.nn.functional.grid_sample(
            self.__data,
            grid,
            TextureSampler.interp_mode_table[dim][self.__interp_mode],
            TextureSampler.wrap_mode_table[self.__wrap_mode],
            False,
        )

        return ret.reshape(self.__c_shape + n_shape)


def main1():
    pass


if __name__ == "__main__":
    pass
