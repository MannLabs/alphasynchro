#!python
'''Module to represent data as a 2D sparse index (CSR).'''


# builtin
import dataclasses

# external
import numpy as np

# local
import alphasynchro.performance.compiling
import alphasynchro.performance.multithreading


@alphasynchro.performance.compiling.njit_dataclass
class SparseIndex:

    indptr: np.ndarray = dataclasses.field(repr=False)
    values: np.ndarray = dataclasses.field(repr=False)
    size: int = dataclasses.field(init=False, repr=False)
    shape: tuple[int, int] = dataclasses.field(init=False)

    def __post_init__(self):
        size = len(self.indptr) - 1
        object.__setattr__(self, "size", size)
        shape = tuple([self.size, self.indptr[-1]])
        object.__setattr__(self, "shape", shape)

    def __len__(self):
        return self.size

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_values(self, index: int) -> np.ndarray:
        start, end = self.get_boundaries(index)
        return self.values[start: end]

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_size(self, index: int) -> int:
        start, end = self.get_boundaries(index)
        return end - start

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_boundaries(self, index: int) -> tuple[int, int]:
        start = 0
        end = 0
        if self.is_valid(index):
            start = self.indptr[index]
            end = self.indptr[index + 1]
        return start, end

    @alphasynchro.performance.compiling.njit(nogil=True)
    def is_valid(self, index: int) -> bool:
        return 0 <= index < self.size

    @alphasynchro.performance.compiling.njit(nogil=True)
    def is_empty(self, index: int) -> bool:
        start, end = self.get_boundaries(index)
        return start == end

    def filter(self, indices: np.ndarray[int]):
        new_indptr = np.zeros(len(indices) + 1, dtype=self.indptr.dtype)
        new_indptr[1:] = self.indptr[indices + 1] - self.indptr[indices]
        new_indptr = np.cumsum(new_indptr)
        new_values = np.empty(new_indptr[-1], dtype=self.values.dtype)
        alphasynchro.performance.multithreading.parallel(
            self._set_new_values_after_filtering,
        )(
            range(len(indices)),
            indices,
            new_indptr,
            new_values,
        )
        return type(self)(indptr=new_indptr, values=new_values)

    @alphasynchro.performance.compiling.njit(nogil=True)
    def _set_new_values_after_filtering(
        self,
        index: int,
        indices: np.ndarray,
        new_indptr: np.ndarray,
        new_values: np.ndarray,
    ) -> None:
        old_index = indices[index]
        values = self.get_values(old_index)
        start = new_indptr[index]
        end = new_indptr[index + 1]
        new_values[start: end] = values

    def filter_values(
        self,
        valid_indices: np.ndarray[bool],
    ):
        new_values = self.values[valid_indices]
        counts = np.zeros_like(self.indptr)
        alphasynchro.performance.multithreading.parallel(
            self._filter_counts,
        )(
            range(len(self)),
            valid_indices,
            counts[1:],
        )
        new_indptr = np.cumsum(counts)
        return type(self)(
            indptr=new_indptr,
            values=new_values,
        )

    @alphasynchro.performance.compiling.njit(nogil=True)
    def _filter_counts(
        self,
        index: int,
        valid_indices: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        start, end = self.get_boundaries(index)
        count = np.sum(valid_indices[start: end])
        counts[index] = count
