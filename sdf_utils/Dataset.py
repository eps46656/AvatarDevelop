

from __future__ import annotations

import dataclasses
import math

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
) -> torch.Tensor:  # [..., N, D]
    D = utils.check_shapes(mean, (..., -1))

    return torch.normal(
        mean[..., None, :].expand(*mean.shape[:-1], N, D), std)


@beartype
class Dataset(dataset_utils.Dataset):
    def __init__(
        self,

        mesh_data: mesh_utils.MeshData,

        std: float,

        shape: torch.Size,
    ):
        assert 0 < std

        assert 0 < math.prod(shape)

        self.mesh_data = mesh_data

        self.std = std

        self.__shape = shape

        self.point_pos = None
        self.signed_dist = None

    @property
    def shape(self) -> torch.Size:
        return self.__shape

    def __getitem__(self, idx: tuple[torch.Tensor]) -> Sample:
        if self.point_pos is None:
            self.refresh()

        return Sample(
            point_pos=self.point_pos[idx],
            signed_dist=self.signed_dist[idx],
        )

    def to(self, *args, **kwargs) -> Dataset:
        self.mesh_data = self.mesh_data.to(*args, **kwargs)
        return self

    def refresh(self) -> None:
        points_cnt = math.prod(self.shape)

        V = self.mesh_data.verts_cnt

        assert 0 < V

        N = points_cnt // V + 1

        p = generate_point(self.mesh_data.vert_pos, self.std, N).view(-1, 3)
        # [P, 3]

        self.point_pos = p[
            torch.randperm(p.shape[0], dtype=torch.long, device=p.device)
            [:points_cnt]
        ]
        # [B, 3]

        self.signed_dist = self.mesh_data.calc_signed_dist(
            self.point_pos)
