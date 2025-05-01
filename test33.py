import torch
import typing

from beartype import beartype


@beartype
def normalize_idx(idx: int, length: int) -> int:
    assert -length <= idx
    assert idx < length

    return idx % length


@beartype
def unbatch_expand(x: typing.Optional[torch.Tensor], dim: int):
    dim = normalize_idx(dim, x.dim())

    idx = [None for _ in range(x.dim())]

    is_first = True

    for i in range(x.dim()):
        if dim <= i or x.shape[i] == 0 or x.stride(i) != 0:
            is_first = False
            idx[i] = slice(None)
        else:
            idx[i] = 0 if is_first else slice(0, 1)

    return x[*idx]


def main1():
    x = torch.rand((3, 3))

    y = x.expand((5, 4, 3, 3))

    print(f"{y=}")

    print(f"{y.stride()=}")

    z = y[:2, 4:, :, :]

    print(f"{z.shape=}")

    z_ = unbatch_expand(z, -2)

    print(f"{z.shape=}")
    print(f"{z_.shape=}")


if __name__ == "__main__":
    main1()
