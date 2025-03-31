import random
import typing

import torch
from beartype import beartype

from . import utils


@beartype
class Dataset:
    @property
    def shape() -> torch.Size:
        raise utils.UnimplementationError()

    def __getitem__(self, idx):
        raise utils.UnimplementationError()


@beartype
class DatasetLoader:
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = -1,
        batches_cnt: int = -1,
    ):
        self.dataset = dataset

        dataset_shape = self.dataset.shape
        dataset_size = dataset_shape.numel()

        assert (0 < batch_size) != (0 < batches_cnt)

        self.batches_cnt = batches_cnt if 0 < batches_cnt else \
            (dataset_size + batch_size - 1) // batch_size

    def load(
        self,
        *,
        random_batch_size: bool = True,
        random_sample: bool = True,
    ) -> typing.Iterable[tuple[tuple[torch.Tensor], object]]:
        dataset_shape = self.dataset.shape
        dataset_size = dataset_shape.numel()

        if random_sample:
            idxes = torch.randperm(dataset_size, dtype=torch.long)
        else:
            idxes = torch.arange(dataset_size, dtype=torch.long)

        batch_idxes = torch.unravel_index(idxes, dataset_shape)

        k = dataset_size // self.batches_cnt
        l = dataset_size % self.batches_cnt

        batch_sizes = [k] * (self.batches_cnt - l) + [k+1] * l

        if random_batch_size:
            random.shuffle(batch_sizes)

        i = 0

        for batch_size in batch_sizes:
            nxt_i = i + batch_size

            cur_batch_idxes = tuple(
                batch_idxes[d][i:nxt_i]
                for d in range(len(dataset_shape))
            )

            yield cur_batch_idxes, self.dataset[cur_batch_idxes]

            i = nxt_i
