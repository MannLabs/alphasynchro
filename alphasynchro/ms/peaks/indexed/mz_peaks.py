#!python
'''Module to index peaks by their push index and set mz value as values.'''


# external
import numpy as np

# local
import alphasynchro.performance.compiling
import alphasynchro.ms.peaks.peaks
import alphasynchro.ms.peaks.indexed.indexed_peaks


@alphasynchro.performance.compiling.njit_dataclass
class PushIndexedMzs(alphasynchro.ms.peaks.indexed.indexed_peaks.IndexedPeaks):

    @classmethod
    def from_data_space(
        cls,
        *,
        peaks: alphasynchro.ms.peaks.peaks.Peaks,
        cycle_shape: tuple,
        tof_indptr: np.ndarray,
    ):
        apices = peaks.aggregate_data.apex_pointer
        push_apices = cls.convert_apices_to_push_apices(
            apices,
            tof_indptr,
        )
        push_indptr = cls.index_push_apices(
            push_apices,
            tof_indptr
        )
        return cls(
            indptr=push_indptr,
            values=peaks.aggregate_data.mz_weighted_average, # TODO, these are not perfectly sorted
            axis_shape=cls.get_axis_shape(
                cycle_shape,
                tof_indptr,
            ),
        )
