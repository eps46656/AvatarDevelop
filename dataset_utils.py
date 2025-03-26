import random

import torch

from . import utils


class Dataset:
    def GetBatchShape() -> torch.Size:
        raise utils.UnimplementationError()

    def BatchGet(self, batch_idxes: tuple[torch.Tensor, ...]):
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

        dataset_shape = self.dataset.GetBatchShape()
        dataset_size = dataset_shape.numel()

        assert (0 < batch_size) != (0 < batches_cnt)

        self.batches_cnt = batches_cnt if 0 < batches_cnt else \
            (dataset_size + batch_size - 1) // batch_size

    def __iter__(self):
        dataset_shape = self.dataset.GetBatchShape()
        dataset_size = dataset_shape.numel()

        idxes = torch.randperm(dataset_size)

        batch_idxes = torch.unravel_index(
            idxes, dataset_shape)

        k = dataset_size // self.batches_cnt
        l = dataset_size % self.batches_cnt

        batch_sizes = [k] * (self.batches_cnt - l) + [k+1] * l
        random.shuffle(batch_sizes)

        i = 0

        for batch_size in batch_sizes:
            nxt_i = i + batch_size

            cur_batch_idxes = tuple(
                batch_idxes[d][i:nxt_i]
                for d in range(dataset_shape.dim())
            )

            yield cur_batch_idxes, self.dataset.BatchGet(cur_batch_idxes)

            i = nxt_i
