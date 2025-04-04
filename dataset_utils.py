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
class DatasetIterator:
    def __init__(
        self,
        dataset: Dataset,
        batches_cnt: int,
        random_batch_size: bool,
        random_sample: bool,
    ):
        self.dataset = dataset

        dataset_shape = self.dataset.shape
        dataset_size = dataset_shape.numel()

        assert 0 < batches_cnt

        batches_cnt = min(batches_cnt, dataset_size)

        if random_sample:
            idxes = torch.randperm(dataset_size, dtype=torch.long)
        else:
            idxes = torch.arange(dataset_size, dtype=torch.long)

        self.batch_idxes = torch.unravel_index(idxes, dataset_shape)

        k = dataset_size // batches_cnt
        l = dataset_size % batches_cnt

        batch_sizes = [k] * (batches_cnt - l) + [k+1] * l

        if random_batch_size:
            random.shuffle(batch_sizes)

        acc_batches_size = [0 for _ in range(batches_cnt + 1)]

        for i in range(1, batches_cnt + 1):
            acc_batches_size[i] = acc_batches_size[i-1] + batch_sizes[i-1]

        self.acc_batches_size = acc_batches_size

        self.batch_i = 0

    def __len__(self) -> int:
        return len(self.acc_batches_size) - 1

    def __iter__(self):
        return self

    def __next__(self) -> tuple[tuple[torch.Tensor], object]:
        if len(self) <= self.batch_i:
            raise StopIteration()

        beg_i = self.acc_batches_size[self.batch_i]
        end_i = self.acc_batches_size[self.batch_i + 1]

        cur_batch_idxes = tuple(
            batch_idxes[beg_i:end_i] for batch_idxes in self.batch_idxes)

        self.batch_i += 1

        return cur_batch_idxes, self.dataset[cur_batch_idxes]


@beartype
def load(
    dataset: Dataset,
    *,
    batch_size: int = 0,
    batches_cnt: int = 0,
    random_batch_size: bool = True,
    random_sample: bool = True,
):
    assert (0 < batch_size) != (0 < batches_cnt)

    if batches_cnt <= 0:
        batches_cnt = (dataset.shape.numel() + batch_size - 1) // batch_size

    return DatasetIterator(
        dataset,
        batches_cnt,
        random_batch_size,
        random_sample,
    )
