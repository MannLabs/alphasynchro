#!python
'''Module to represent transitions of individuek frame groups.'''


# builtin
import dataclasses

# local
import alphasynchro.data.sparse_indices
import alphasynchro.stats.distributions
import alphasynchro.data.dataframe
import alphasynchro.performance.compiling
import alphasynchro.io.hdf
import alphasynchro.ms.peaks.precursors

# external
import numpy as np


@alphasynchro.performance.compiling.njit_dataclass
class Transitions(alphasynchro.data.sparse_indices.SparseIndex):

    weights: np.ndarray = dataclasses.field(repr=False)
    precursor_indices: np.ndarray = dataclasses.field(repr=False)

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_weights(self, index: int) -> np.ndarray:
        start, end = self.get_boundaries(index)
        return self.weights[start: end]

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_precursor_index(self, index: int) -> int:
        precursor_index = -1
        if self.is_valid(index):
            precursor_index = self.precursor_indices[index]
        return precursor_index

    def filter(self, indices: np.ndarray):
        new_indptr = np.zeros(len(indices) + 1, dtype=self.indptr.dtype)
        new_indptr[1:] = self.indptr[indices + 1] - self.indptr[indices]
        new_indptr = np.cumsum(new_indptr)
        new_values = np.empty(new_indptr[-1], dtype=self.values.dtype)
        new_weights = np.empty(new_indptr[-1], dtype=self.weights.dtype)
        new_precursor_indices = self.precursor_indices[indices]
        alphasynchro.performance.multithreading.parallel(
            self._set_new_values_after_filtering,
        )(
            range(len(indices)),
            indices,
            new_indptr,
            new_values,
            new_weights,
        )
        return type(self)(
            indptr=new_indptr,
            values=new_values,
            weights=new_weights,
            precursor_indices=new_precursor_indices,
        )

    @alphasynchro.performance.compiling.njit(nogil=True)
    def _set_new_values_after_filtering(
        self,
        index: int,
        indices: np.ndarray,
        new_indptr: np.ndarray,
        new_values: np.ndarray,
        new_weights: np.ndarray,
    ) -> None:
        old_index = indices[index]
        values = self.get_values(old_index)
        weights = self.get_weights(old_index)
        start = new_indptr[index]
        end = new_indptr[index + 1]
        new_values[start: end] = values
        new_weights[start: end] = weights

    def filter_weights(
        self,
        valid_indices: np.ndarray,
    ):
        new_weights = self.weights[valid_indices]
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
            weights=new_weights,
            precursor_indices=self.precursor_indices,
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

    @classmethod
    def from_analysis_hdf(cls, analysis_file):
        transitions = cls(
            indptr=analysis_file.transitions.indptr,
            values=analysis_file.transitions.values,
            weights=analysis_file.transitions.weights,
            precursor_indices=analysis_file.transitions.precursor_indices,
        )
        return transitions


@alphasynchro.performance.compiling.njit
def get_best_uniqueness_mask(valid_fragments, weights):
    mask = np.zeros(np.max(valid_fragments) + 1, dtype=np.int64)
    for i, fragment in enumerate(valid_fragments):
        if mask[fragment] == 0:
            mask[fragment] = i
            continue
        if weights[i] < weights[mask[fragment]]:
            mask[fragment] = i
    return mask[mask>0]
