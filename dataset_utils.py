from __future__ import annotations

import math
import random

import torch
from beartype import beartype

from . import utils


@beartype
class Dataset:
    @property
    def shape(self) -> torch.Size:
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


@beartype
class BatchIdxIterator:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        batch_size: int = 0,
        batches_cnt: int = 0,
        shuffle: bool,
    ):
        size = math.prod(shape)

        assert (0 < batch_size) != (0 < batches_cnt)

        if batches_cnt <= 0:
            batches_cnt = (size + batch_size - 1) // batch_size

        assert 0 < batches_cnt

        batches_cnt = min(batches_cnt, size)

        if shuffle:
            idxes = torch.randperm(size, dtype=torch.long)
        else:
            idxes = torch.arange(size, dtype=torch.long)

        self.batch_idxes = torch.unravel_index(idxes, shape)

        self.acc_batches_size = [
            i * size // batches_cnt for i in range(batches_cnt + 1)]

        self.batch_i = 0

    def __len__(self) -> int:
        return len(self.acc_batches_size) - 1

    def __iter__(self) -> BatchIdxIterator:
        return self

    def __next__(self) -> tuple[torch.Tensor, ...]:
        if len(self) <= self.batch_i:
            raise StopIteration()

        beg_i = self.acc_batches_size[self.batch_i]
        end_i = self.acc_batches_size[self.batch_i + 1]

        cur_batch_idxes = tuple(
            batch_idxes[beg_i:end_i] for batch_idxes in self.batch_idxes)

        self.batch_i += 1

        return cur_batch_idxes


@beartype
class DatasetIterator:
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 0,
        batches_cnt: int = 0,
        shuffle: bool,
    ):
        self.dataset = dataset

        self.batch_idx_iter = BatchIdxIterator(
            dataset.shape,
            batch_size=batch_size,
            batches_cnt=batches_cnt,
            shuffle=shuffle,
        )

    def __len__(self) -> int:
        return len(self.batch_idx_iter) - 1

    def __iter__(self) -> DatasetIterator:
        return self

    def __next__(self) -> tuple[tuple[torch.Tensor, ...], object]:
        batch_idx = next(self.batch_idx_iter)
        return batch_idx, self.dataset[batch_idx]


@beartype
def load(
    dataset: Dataset,
    *,
    batch_size: int = 0,
    batches_cnt: int = 0,
    shuffle: bool = True,
):
    return DatasetIterator(
        dataset,
        batch_size=batch_size,
        batches_cnt=batches_cnt,
        shuffle=shuffle,
    )
