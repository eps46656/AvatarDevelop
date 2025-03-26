import random

import torch

from . import utils


class Dataset:
    def __len__(self):
        raise utils.UnimplementationError()

    def Get(self, idx: int):
        raise utils.UnimplementationError()

    def BatchGet(self, idxes: torch.Tensor):
        raise utils.UnimplementationError()


class DatasetLoader:
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int = -1,
        batches_cnt: int = -1,
    ):
        self.dataset = dataset

        dataset_size = len(self.dataset)
        assert 0 <= dataset_size

        assert (0 < batch_size) != (0 < batches_cnt)

        self.batches_cnt = batches_cnt if 0 < batches_cnt else \
            (dataset_size + batch_size - 1) // batch_size

    def __iter__(self):
        dataset_size = len(self.dataset)
        assert 0 <= dataset_size

        idxes = torch.randperm(dataset_size)

        k = dataset_size // self.batches_cnt
        l = dataset_size % self.batches_cnt

        batch_sizes = [k] * (self.batches_cnt - l) + [k+1] * l
        random.shuffle(batch_sizes)

        i = 0

        for batch_size in batch_sizes:
            nxt_i = i + batch_size

            cur_idxes = idxes[i:nxt_i]

            yield cur_idxes, self.dataset.BatchGet(cur_idxes)

            i = nxt_i
