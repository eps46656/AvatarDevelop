

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
class Dataset(dataset_utils.Dataset):
    def __init__(
        self,
        mean: tuple[float, float,  float],
        std: tuple[float, float,  float],
        epoch_size: int,
        mesh_data: mesh_utils.MeshData,
        vert_pos: torch.Tensor,
    ):
        assert 0 < epoch_size

        self.mean = mean
        self.std = std

        self.epoch_size = epoch_size

        self.mesh_data = mesh_data
        self.vert_pos = vert_pos

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.epoch_size,))

    @property
    def device(self) -> torch.device:
        return self.vert_pos.device

    def __getitem__(self, idx: tuple[torch.Tensor]) -> Sample:
        N = idx[0].numel()

        point_pos = utils.empty_like(self.vert_pos, shape=(N, 3))

        for d in range(3):
            torch.normal(self.mean[d], self.std[d], (N,),
                         out=point_pos[:, d])

        signed_dists = self.mesh_data.calc_signed_dist(
            self.vert_pos,
            point_pos,
        )  # [N]

        return Sample(
            point_pos=point_pos,
            signed_dist=signed_dists,
        )

    def to(self, *args, **kwargs) -> Dataset:
        self.vert_pos = self.vert_pos.to(*args, **kwargs)
        return self
