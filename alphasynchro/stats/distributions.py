#!python
'''Module to define Partial/Cumulative distributions with(out) offsets and/or summed values.'''


# builtin
import dataclasses

# local
import alphasynchro.data.sparse_indices
import alphasynchro.performance.compiling
import alphasynchro.performance.multithreading

# external
import numpy as np
import scipy.signal


@alphasynchro.performance.compiling.njit_dataclass
class PDF(alphasynchro.data.sparse_indices.SparseIndex):

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_cdf(self, index: int) -> np.ndarray:
        distribution = self.get_pdf(index)
        cdf = np.cumsum(distribution)
        if cdf[-1] > 0:
            cdf /= cdf[-1]
        return cdf

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_pdf(self, index: int) -> np.ndarray:
        return self.get_values(index)

    def to_cdf(self):
        new_values = np.empty(self.indptr[-1], dtype=self.values.dtype)
        alphasynchro.performance.multithreading.parallel(
            self._convert_to_cdf,
        )(
            range(len(self)),
            new_values,
        )
        return CDF(
            indptr=self.indptr.copy(),
            values=new_values,
        )

    @alphasynchro.performance.compiling.njit(nogil=True)
    def _convert_to_cdf(
        self,
        index: int,
        new_values: np.ndarray
    ) -> None:
        cdf = self.get_cdf(index)
        start, end = self.get_boundaries(index)
        new_values[start: end] = cdf


@alphasynchro.performance.compiling.njit_dataclass
class CDF(alphasynchro.data.sparse_indices.SparseIndex):

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_cdf(self, index: int) -> np.ndarray:
        return self.get_values(index)

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_pdf(
        self,
        index: int,
    ) -> np.ndarray:
        cdf = self.get_cdf(index)
        distribution = np.empty_like(cdf)
        if len(cdf) > 0:
            distribution[0] = cdf[0]
            distribution[1:] = np.diff(cdf)
        return distribution

    def to_pdf(self):
        new_values = np.empty(self.indptr[-1], dtype=self.values.dtype)
        alphasynchro.performance.multithreading.parallel(
            self._convert_to_pdf,
        )(
            range(len(self)),
            new_values,
        )
        return PDF(
            indptr=self.indptr.copy(),
            values=new_values,
        )

    @alphasynchro.performance.compiling.njit(nogil=True)
    def _convert_to_pdf(
        self,
        index: int,
        new_values: np.ndarray
    ) -> None:
        pdf = self.get_pdf(index)
        start, end = self.get_boundaries(index)
        new_values[start: end] = pdf

    def smooth(self, index: int, smooth_array: np.ndarray) -> np.ndarray:
        distribution = self.get_pdf(index)
        smooth_distribution = scipy.signal.convolve(
            distribution,
            smooth_array,
            mode='same'
        )
        return smooth_distribution

@alphasynchro.performance.compiling.njit_dataclass
class CDFWithOffset(CDF):

    start_offsets: np.ndarray = dataclasses.field(repr=False)

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_start_offset(self, index: int) -> int:
        offset = 0
        if self.is_valid(index):
            offset = self.start_offsets[index]
        return offset

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_end_offset(self, index: int) -> int:
        offset = self.get_start_offset(index)
        offset += self.get_size(index)
        return offset

    def filter(self, indices: np.ndarray):
        new_indptr = np.zeros(len(indices) + 1, dtype=self.indptr.dtype)
        new_indptr[1:] = self.indptr[indices + 1] - self.indptr[indices]
        new_indptr = np.cumsum(new_indptr)
        new_start_offsets = self.start_offsets[indices]
        new_values = np.empty(new_indptr[-1], dtype=self.values.dtype)
        alphasynchro.performance.multithreading.parallel(
            self._set_new_values_after_filtering,
        )(
            range(len(indices)),
            indices,
            new_indptr,
            new_values,
        )
        return type(self)(
            indptr=new_indptr,
            values=new_values,
            start_offsets=new_start_offsets,
        )


@alphasynchro.performance.compiling.njit_dataclass
class CDFWithSummedValues(CDF):

    summed_values: np.ndarray = dataclasses.field(repr=False)

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_summed_value(
        self,
        index: int
    ):
        value = 0
        if self.is_valid(index):
            value = self.summed_values[index]
        return value

    def filter(self, indices: np.ndarray):
        new_indptr = np.zeros(len(indices) + 1, dtype=self.indptr.dtype)
        new_indptr[1:] = self.indptr[indices + 1] - self.indptr[indices]
        new_indptr = np.cumsum(new_indptr)
        new_values = np.empty(new_indptr[-1], dtype=self.values.dtype)
        summed_values = self.summed_values[indices]
        alphasynchro.performance.multithreading.parallel(
            self._set_new_values_after_filtering,
        )(
            range(len(indices)),
            indices,
            new_indptr,
            new_values,
        )
        return type(self)(
            indptr=new_indptr,
            values=new_values,
            summed_values=summed_values
        )


@alphasynchro.performance.compiling.njit_dataclass
class CDFWithOffsetAndSummedValues(CDFWithOffset, CDFWithSummedValues):

    def filter(self, indices: np.ndarray):
        new_indptr = np.zeros(len(indices) + 1, dtype=self.indptr.dtype)
        new_indptr[1:] = self.indptr[indices + 1] - self.indptr[indices]
        new_indptr = np.cumsum(new_indptr)
        new_start_offsets = self.start_offsets[indices]
        new_values = np.empty(new_indptr[-1], dtype=self.values.dtype)
        new_summed_values = self.summed_values[indices]
        alphasynchro.performance.multithreading.parallel(
            self._set_new_values_after_filtering,
        )(
            range(len(indices)),
            indices,
            new_indptr,
            new_values,
        )
        return type(self)(
            indptr=new_indptr,
            values=new_values,
            start_offsets=new_start_offsets,
            summed_values=new_summed_values
        )
