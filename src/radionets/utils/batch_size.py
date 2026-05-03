from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Generator

OOM_EXCEPTIONS = (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError, MemoryError)
MIN_BATCH_SIZE = 1


class AdaptiveBatchSize:
    def __init__(
        self, *tensors: torch.Tensor, initial_batch_size: int | None = None
    ) -> None:
        self.tensors = tensors
        self.n_samples = tensors[0].shape[0]
        self.batch_size = initial_batch_size if initial_batch_size else self.n_samples

    def __enter__(self) -> AdaptiveBatchSize:
        self._cuda_gc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cuda_gc()

    def _get_batches(self) -> Generator:
        for i in range(0, self.n_samples, self.batch_size):
            yield tuple(t[i : i + self.batch_size] for t in self.tensors)

    def __iter__(self) -> Generator:
        while True:
            try:
                yield from self._get_batches()
                return
            except OOM_EXCEPTIONS:  # pragma: no cover
                if self.batch_size >= MIN_BATCH_SIZE:
                    raise

                self.batch_size = min(MIN_BATCH_SIZE, self.batch_size // 2)
                self._cuda_gc()

    def _cuda_gc(self):
        """Garbage collector for CUDA."""
        gc.collect()

        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.empty_cache()
