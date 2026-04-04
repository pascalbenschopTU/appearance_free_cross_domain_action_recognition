"""Dataset samplers."""

import torch
from torch.utils.data import Sampler

class ResumableShuffleSampler(Sampler[int]):
    def __init__(self, dataset_len: int, seed: int, epoch: int, start_index: int, drop_last: bool, batch_size: int):
        self.dataset_len = dataset_len
        self.seed = seed
        self.epoch = epoch
        self.start_index = start_index
        self.drop_last = drop_last
        self.batch_size = batch_size

        if drop_last:
            self.epoch_size = (dataset_len // batch_size) * batch_size
        else:
            self.epoch_size = dataset_len

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + 1000003 * self.epoch)

        perm = torch.randperm(self.dataset_len, generator=g).tolist()

        # enforce consistent epoch size when drop_last=True
        perm = perm[:self.epoch_size]

        # resume from start_index (in samples)
        start = min(self.start_index, len(perm))
        for idx in perm[start:]:
            yield idx

    def __len__(self):
        return self.epoch_size - min(self.start_index, self.epoch_size)

__all__ = ["ResumableShuffleSampler"]
