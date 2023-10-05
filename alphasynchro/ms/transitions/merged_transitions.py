#!python
'''Module to merge transitions from mutliple frame groups.'''


# external
import numpy as np

# local
import alphasynchro.performance.compiling
import alphasynchro.ms.peaks.fragments
import alphasynchro.algorithms.calibration
import alphasynchro.stats.distributions
import alphasynchro.data.sparse_indices
import alphasynchro.ms.transitions.frame_transitions


@alphasynchro.performance.compiling.njit_dataclass
class MergedFrames(alphasynchro.data.sparse_indices.SparseIndex):

    rt_weights: np.ndarray
    im_weights: np.ndarray
    fragments: alphasynchro.ms.peaks.fragments.Fragments

    @classmethod
    def from_transition_dicts(
        cls,
        *,
        im_transition_dict: dict[int: alphasynchro.ms.transitions.frame_transitions.Transitions],
        rt_transition_dict: dict[int: alphasynchro.ms.transitions.frame_transitions.Transitions],
        fragments: alphasynchro.ms.peaks.fragments.Fragments,
    ):
        merged_transitions_indptr = np.zeros_like(im_transition_dict[1].indptr)
        for _, transitions in im_transition_dict.items():
            merged_transitions_indptr += transitions.indptr
        merged_transitions_values = np.empty(merged_transitions_indptr[-1], dtype=np.int64)
        merged_rt_weights = np.empty(merged_transitions_indptr[-1])
        merged_im_weights = np.empty(merged_transitions_indptr[-1])
        merged_transitions_indptr_copy = np.copy(merged_transitions_indptr)
        for frame, im_transitions in im_transition_dict.items():
            rt_transitions = rt_transition_dict[frame]
            for precursor_index in range(im_transitions.size):
                if im_transitions.is_empty(precursor_index):
                    continue
                start = merged_transitions_indptr_copy[precursor_index]
                size = im_transitions.get_size(precursor_index)
                merged_transitions_values[start: start + size] = im_transitions.get_values(precursor_index)
                merged_im_weights[start: start + size] = im_transitions.get_weights(precursor_index)
                merged_rt_weights[start: start + size] = rt_transitions.get_weights(precursor_index)
                merged_transitions_indptr_copy[precursor_index] += size
        return cls(
            indptr=merged_transitions_indptr,
            values=merged_transitions_values,
            rt_weights=merged_im_weights,
            im_weights=merged_rt_weights,
            fragments=fragments,
        )

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_rt_weights(self, index: int) -> np.ndarray:
        start, end = self.get_boundaries(index)
        return self.rt_weights[start: end]

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_im_weights(self, index: int) -> np.ndarray:
        start, end = self.get_boundaries(index)
        return self.im_weights[start: end]

    def sort_mz(self):
        new_values = np.empty_like(self.values)
        new_rt_weights = np.empty_like(self.rt_weights)
        new_im_weights = np.empty_like(self.im_weights)
        alphasynchro.performance.multithreading.parallel(
            self._sort_by_mz,
        )(
            range(self.size),
            new_values,
            new_rt_weights,
            new_im_weights,
        )
        return type(self)(
            indptr=self.indptr,
            values=new_values,
            rt_weights=new_rt_weights,
            im_weights=new_im_weights,
            fragments=self.fragments,
        )

    @alphasynchro.performance.compiling.njit(nogil=True)
    def _sort_by_mz(
        self,
        precursor_index: int,
        new_values: np.ndarray,
        new_rt_weights: np.ndarray,
        new_im_weights: np.ndarray,
    ) -> None:
        values = self.get_values(precursor_index)
        mz_values = self.fragments.aggregate_data.mz_weighted_average[values]
        order = np.argsort(mz_values)
        start, end = self.get_boundaries(precursor_index)
        new_values[start: end] = values[order]
        new_rt_weights[start: end] = self.get_rt_weights(precursor_index)[order]
        new_im_weights[start: end] = self.get_im_weights(precursor_index)[order]

    def count_all_merged_peaks(
        self,
        ppm_tolerance: float = 50,
    ) -> np.ndarray:
        buffer_array = np.zeros(len(self) + 1, dtype=np.int64)
        alphasynchro.performance.multithreading.parallel(
            self._count_fragments,
        )(
            range(self.size),
            buffer_array[1:],
            ppm_tolerance,
        )
        buffer_array = np.cumsum(buffer_array)
        return buffer_array

    @alphasynchro.performance.compiling.njit(nogil=True)
    def _count_fragments(
        self,
        precursor_index: int,
        buffer_array: np.ndarray,
        ppm_tolerance: float,
    ) -> None:
        count = self.count_fragments(precursor_index, ppm_tolerance)
        buffer_array[precursor_index] = count

    @alphasynchro.performance.compiling.njit(nogil=True)
    def count_fragments(
        self,
        precursor_index: int,
        ppm_tolerance: float,
    ) -> int:
        count = 0
        for _ in self.generate_groups(precursor_index, ppm_tolerance):
            count += 1
        return count

    @alphasynchro.performance.compiling.njit(nogil=True)
    def generate_groups(
        self,
        precursor_index: int,
        ppm_tolerance: float,
    ) -> (tuple[int, int]):
        if self.is_empty(precursor_index):
            return
        values = self.get_values(precursor_index)
        last_mz = self.fragments.aggregate_data.mz_weighted_average[values[0]]
        previous_index, last_index = self.get_boundaries(precursor_index)
        for index, fragment_index in enumerate(values[1:], previous_index + 1):
            mz_value = self.fragments.aggregate_data.mz_weighted_average[fragment_index]
            if (mz_value - last_mz) / mz_value * 10**6 >= ppm_tolerance:
                yield (previous_index, index)
                last_mz = mz_value
                previous_index = index
        yield (previous_index, last_index)

    def index_peaks(
        self,
        indptr: np.ndarray = None,
        ppm_tolerance: float = 50,
    ) -> alphasynchro.data.sparse_indices.SparseIndex:
        if indptr is None:
            indptr = self.count_all_merged_peaks(ppm_tolerance)
        buffer_array = np.zeros(indptr[-1] + 1, dtype=np.int64)
        alphasynchro.performance.multithreading.parallel(
            self._set_merged_fragments,
        )(
            range(self.size),
            buffer_array,
            np.copy(indptr),
            ppm_tolerance,
        )
        buffer_array[-1] = self.shape[1]
        return alphasynchro.data.sparse_indices.SparseIndex(
            indptr=buffer_array,
            values=self.values,
        )

    @alphasynchro.performance.compiling.njit(nogil=True)
    def _set_merged_fragments(
        self,
        precursor_index: int,
        buffer_array: np.ndarray,
        indptr: np.ndarray,
        ppm_tolerance: float,
    ) -> None:
        for (start_index, _) in self.generate_groups(
            precursor_index,
            ppm_tolerance,
        ):
            offset = indptr[precursor_index]
            buffer_array[offset] = start_index
            indptr[precursor_index] += 1
