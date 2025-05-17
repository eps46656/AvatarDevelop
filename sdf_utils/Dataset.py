

from __future__ import annotations

import dataclasses

import torch
from beartype import beartype

from .. import dataset_utils, mesh_utils, utils


@beartype
@dataclasses.dataclass
class Sample:
    point_pos: torch.Tensor  # [..., 3]
    signed_dist: torch.Tensor  # [...]


@beartype
def generate_point(
    mean: torch.Tensor,  # [..., D]
    std: float,
    N: int,
) -> torch.Tensor:
    D = utils.check_shapes(mean, (..., -1))

    mean = mean[..., None, :]
    # [..., 1, D]

    return (mean + torch.normal(mean, std)).reshape(-1, D)


@beartype
class Dataset(dataset_utils.Dataset):
    def __init__(
        self,
        mean: tuple[float, float,  float],
        std: tuple[float, float,  float],
        epoch_size: int,
        mesh_graph: mesh_utils.MeshGraph,
        vert_pos: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ):
        assert 0 < epoch_size

        self.mean = mean
        self.std = std

        self.epoch_size = epoch_size

        self.mesh_data = mesh_utils.MeshData(mesh_graph, vert_pos)

        self.dtype = dtype
        self.device = device

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.epoch_size,))

    def __getitem__(self, idx: tuple[torch.Tensor]) -> Sample:
        N = idx[0].numel()

        point_pos = torch.empty(
            (N, 3), dtype=self.dtype, device=self.device)

        for d in range(3):
            torch.normal(self.mean[d], self.std[d], (N,), out=point_pos[:, d])

        signed_dists = self.mesh_data.calc_signed_dist(point_pos)  # [N]

        return Sample(
            point_pos=point_pos,
            signed_dist=signed_dists,
        )

    def to(self, *args, **kwargs) -> Dataset:
        self.mesh_data = self.mesh_data.to(*args, **kwargs)
        return self
