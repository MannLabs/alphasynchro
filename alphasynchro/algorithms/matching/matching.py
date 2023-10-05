#!python


# external
import numpy as np

# local
import alphasynchro.ms.peaks.indexed.mz_peaks
import alphasynchro.performance.compiling
import alphasynchro.ms.dimensions.push_matching
import alphasynchro.ms.dimensions.mz_matching


@alphasynchro.performance.compiling.njit_dataclass
class Matcher:

    indexed_precursors: alphasynchro.ms.peaks.indexed.mz_peaks.PushIndexedMzs
    indexed_fragments: alphasynchro.ms.peaks.indexed.mz_peaks.PushIndexedMzs
    scan_tolerance: int = 10
    cycle_tolerance: int = 3

    @alphasynchro.performance.compiling.njit(nogil=True)
    def count_matches(
        self,
        push_index: int,
    ) -> int:
        count = 0
        for _ in self.generate_matches(push_index):
            count += 1
        return count


@alphasynchro.performance.compiling.njit_dataclass
class UnfragmentedMatcher(Matcher):

    ppm_tolerance: float = 50

    @alphasynchro.performance.compiling.njit(nogil=True)
    def generate_matches(
        self,
        push_index: int,
    ) -> (tuple[int, int]):
        if self.indexed_precursors.is_empty(push_index):
            return
        mz_values1 = self.indexed_precursors.get_values(push_index)
        (
            precursor_start_offset,
            _
        ) = self.indexed_precursors.get_boundaries(push_index)
        for frame_offset in range(1, self.indexed_precursors.axis_shape[1]):
            for other_push_index in alphasynchro.ms.dimensions.push_matching.generate_neighbor_push_indices(
                push_index + frame_offset * self.indexed_precursors.axis_shape[2],
                self.indexed_precursors.axis_shape,
                self.scan_tolerance,
                self.cycle_tolerance,
            ):
                mz_values2 = self.indexed_fragments.get_values(other_push_index)
                (
                    fragment_start_offset,
                    _
                ) = self.indexed_fragments.get_boundaries(other_push_index)
                for precursor_index, fragment_index in alphasynchro.ms.dimensions.mz_matching.match_mz_arrays(
                    mz_values1,
                    mz_values2,
                    self.ppm_tolerance
                ):
                    yield (
                        precursor_start_offset + precursor_index,
                        fragment_start_offset + fragment_index
                    )


@alphasynchro.performance.compiling.njit_dataclass
class FragmentedMatcher(Matcher):

    frame: int

    @alphasynchro.performance.compiling.njit(nogil=True)
    def generate_matches(
        self,
        push_index: int,
    ) -> (tuple[int, int]):
        if self.indexed_precursors.is_empty(push_index):
            return
        precursor_indices = self.indexed_precursors.get_values(push_index)
        for other_push_index in alphasynchro.ms.dimensions.push_matching.generate_neighbor_push_indices(
            push_index + self.frame * self.indexed_precursors.axis_shape[2],
            self.indexed_precursors.axis_shape,
            self.scan_tolerance,
            self.cycle_tolerance,
        ):
            (
                fragment_start_offset,
                fragment_end_offset
            ) = self.indexed_fragments.get_boundaries(other_push_index)
            for precursor_index in precursor_indices:
                for fragment_index in range(fragment_start_offset, fragment_end_offset):
                    yield (precursor_index, fragment_index)


@alphasynchro.performance.compiling.njit_dataclass
class MultiThreader:

    def count_all(
        self,
    ) -> np.ndarray[float]:
        match_counts = np.empty(len(self.indexed_precursors), dtype=np.int64)
        alphasynchro.performance.multithreading.parallel(
            self._count_from_buffers
        )(
            range(len(match_counts)),
            match_counts,
        )
        return match_counts

    @alphasynchro.performance.compiling.njit(nogil=True)
    def _count_from_buffers(
        self,
        push_index: int,
        match_counts: np.ndarray[float],
    ) -> None:
        match_count = self.count_matches(push_index)
        match_counts[push_index] = match_count

    def match_all(
        self,
        match_counts: np.ndarray = None,
    ) -> np.ndarray[float]:
        if match_counts is None:
            match_counts = self.count_all()
        match_indptr = np.zeros(
            len(self.indexed_precursors) + 1,
            dtype=np.int64
        )
        match_indptr[1:] = np.cumsum(match_counts)
        matches = np.empty(
            (match_indptr[-1], 2),
            dtype=np.int64
        )
        alphasynchro.performance.multithreading.parallel(
            self._set_match_from_buffers
        )(
            range(len(match_indptr)),
            matches,
            match_indptr,
        )
        return matches

    @alphasynchro.performance.compiling.njit(nogil=True)
    def _set_match_from_buffers(
        self,
        push_index: int,
        matches: np.ndarray,
        match_indptr: np.ndarray
    ) -> None:
        for precursor_index, fragment_index in self.generate_matches(push_index):
            match_count = match_indptr[push_index]
            matches[match_count] = precursor_index, fragment_index
            match_indptr[push_index] += 1


@alphasynchro.performance.compiling.njit_dataclass
class UnfragmentedMatcherMultithreaded(UnfragmentedMatcher, MultiThreader):

    pass


@alphasynchro.performance.compiling.njit_dataclass
class FragmentedMatcherMultithreaded(FragmentedMatcher, MultiThreader):

    pass
