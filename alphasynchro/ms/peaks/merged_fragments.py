#!python
'''Module to represent merged fragments from multiple frame groups.'''


# local
import alphasynchro.data.sparse_indices
import alphasynchro.stats.distributions
import alphasynchro.data.dataframe
import alphasynchro.performance.compiling
import alphasynchro.io.hdf
import alphasynchro.ms.peaks.fragments
import alphasynchro.ms.transitions.merged_transitions
import alphasynchro.performance.multithreading

# external
import numpy as np


@alphasynchro.performance.compiling.njit_dataclass
class MergedFragments:

    fragment_pointers: alphasynchro.data.sparse_indices.SparseIndex
    frame_intensities: alphasynchro.stats.distributions.CDF
    rt_weights: alphasynchro.data.sparse_indices.SparseIndex
    im_weights: alphasynchro.data.sparse_indices.SparseIndex
    aggregate_data: alphasynchro.data.dataframe.DataFrame

    @classmethod
    def from_pointers(
        cls,
        fragment_pointers: alphasynchro.data.sparse_indices.SparseIndex,
        fragments: alphasynchro.ms.peaks.fragments.Fragments,
        merged_transition_index: alphasynchro.ms.transitions.merged_transitions.MergedFrames,
        fragment_frame_count: int,
    ):
        rt_weights = alphasynchro.data.sparse_indices.SparseIndex(
            indptr=fragment_pointers.indptr,
            values=merged_transition_index.rt_weights,
        )
        im_weights = alphasynchro.data.sparse_indices.SparseIndex(
            indptr=fragment_pointers.indptr,
            values=merged_transition_index.im_weights,
        )
        frame_intensities = FrameIntensitiesCalculator(
            fragments=fragments,
            fragment_frame_count=fragment_frame_count,
            fragment_pointers=fragment_pointers,
        ).calculate_all()
        aggregate_data = StatsAggregateCalculator(
            fragments=fragments,
            fragment_pointers=fragment_pointers,
        ).calculate_all()
        return cls(
            fragment_pointers=fragment_pointers,
            frame_intensities=frame_intensities,
            rt_weights=rt_weights,
            im_weights=im_weights,
            aggregate_data=aggregate_data,
        )

    @classmethod
    def from_analysis_hdf(cls, analysis_file):
        merged_fragments = cls(
            fragment_pointers=alphasynchro.data.sparse_indices.SparseIndex(
                indptr=analysis_file.merged_fragments.fragment_pointers.indptr,
                values=analysis_file.merged_fragments.fragment_pointers.values,
            ),
            frame_intensities=alphasynchro.stats.distributions.CDF(
                indptr=analysis_file.merged_fragments.frame_intensities.indptr,
                values=analysis_file.merged_fragments.frame_intensities.values,
            ),
            rt_weights=alphasynchro.data.sparse_indices.SparseIndex(
                indptr=analysis_file.merged_fragments.rt_weights.indptr,
                values=analysis_file.merged_fragments.rt_weights.values,
            ),
            im_weights=alphasynchro.data.sparse_indices.SparseIndex(
                indptr=analysis_file.merged_fragments.im_weights.indptr,
                values=analysis_file.merged_fragments.im_weights.values,
            ),
            aggregate_data=alphasynchro.data.dataframe.DataFrame(
                **{
                    array: analysis_file.merged_fragments.aggregate_data.__getattribute__(
                        array
                    ) for array in analysis_file.merged_fragments.aggregate_data.arrays
                }
            ),
        )
        return merged_fragments

    def __len__(self):
        return len(self.aggregate_data)


@alphasynchro.performance.compiling.njit_dataclass
class FrameIntensitiesCalculator:

    fragments: alphasynchro.ms.peaks.fragments.Fragments
    fragment_pointers: alphasynchro.data.sparse_indices.SparseIndex
    fragment_frame_count: int

    def calculate_all(
        self,
    ) -> alphasynchro.data.dataframe.DataFrame:
        indptr = np.arange(len(self.fragment_pointers) + 1) * self.fragment_frame_count
        values = np.zeros(len(self.fragment_pointers) * self.fragment_frame_count)
        alphasynchro.performance.multithreading.parallel(
            self._calculate_frame_intensities,
        )(
            range(self.fragment_pointers.size),
            values,
        )
        return alphasynchro.stats.distributions.PDF(
            indptr=indptr,
            values=values,
        ).to_cdf()

    @alphasynchro.performance.compiling.njit
    def _calculate_frame_intensities(
        self,
        merged_fragment_index,
        values,
    ):
        offset = merged_fragment_index * self.fragment_frame_count - 1
        fragment_indices = self.fragment_pointers.get_values(merged_fragment_index)
        for fragment_index in fragment_indices:
            frame = self.fragments.aggregate_data.frame_group[fragment_index]
            intensity = self.fragments.aggregate_data.summed_intensity[fragment_index]
            values[offset + frame] += intensity


@alphasynchro.performance.compiling.njit_dataclass
class StatsAggregateCalculator:

    fragments: alphasynchro.ms.peaks.fragments.Fragments
    fragment_pointers: alphasynchro.data.sparse_indices.SparseIndex

    def calculate_all(
        self,
    ) -> alphasynchro.data.dataframe.DataFrame:
        mz_weighted_average = np.empty(len(self.fragment_pointers), dtype=np.float32)
        im_weighted_average = np.empty(len(self.fragment_pointers), dtype=np.float32)
        rt_weighted_average = np.empty(len(self.fragment_pointers), dtype=np.float32)
        summed_intensity = np.empty(len(self.fragment_pointers), dtype=np.float32)
        alphasynchro.performance.multithreading.parallel(
            self._calculate_aggregate_data,
        )(
            range(self.fragment_pointers.size),
            mz_weighted_average,
            im_weighted_average,
            rt_weighted_average,
            summed_intensity,
        )
        return alphasynchro.data.dataframe.DataFrame(
            mz_weighted_average=mz_weighted_average,
            im_weighted_average=im_weighted_average,
            rt_weighted_average=rt_weighted_average,
            summed_intensity=summed_intensity,
        )

    @alphasynchro.performance.compiling.njit
    def _calculate_aggregate_data(
        self,
        merged_fragment_index,
        mz_weighted_average,
        im_weighted_average,
        rt_weighted_average,
        summed_intensity,
    ):
        fragment_indices = self.fragment_pointers.get_values(merged_fragment_index)
        intensities = self.fragments.aggregate_data.summed_intensity[fragment_indices]
        total_intensity = np.sum(intensities)
        mz_weighted_average[merged_fragment_index] = np.sum(
            self.fragments.aggregate_data.mz_weighted_average[fragment_indices] * intensities
        ) / total_intensity
        im_weighted_average[merged_fragment_index] = np.sum(
            self.fragments.aggregate_data.im_weighted_average[fragment_indices] * intensities
        ) / total_intensity
        rt_weighted_average[merged_fragment_index] = np.sum(
            self.fragments.aggregate_data.rt_weighted_average[fragment_indices] * intensities
        ) / total_intensity
        summed_intensity[merged_fragment_index] = total_intensity
