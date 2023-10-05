#!python
'''Module to index peaks by their push index and set scan index as values.'''


# external
import numpy as np

# local
import alphasynchro.performance.compiling
import alphasynchro.ms.peaks.peaks
import alphasynchro.ms.peaks.indexed.indexed_peaks


@alphasynchro.performance.compiling.njit_dataclass
class PushIndexedImPeaks(alphasynchro.ms.peaks.indexed.indexed_peaks.IndexedPeaks):

    @classmethod
    def from_data_space(
        cls,
        *,
        peaks: alphasynchro.ms.peaks.peaks.Peaks,
        im_apices: np.ndarray,
        cycle_shape: tuple,
        tof_indptr: np.ndarray,
    ):
        selection = np.flatnonzero(im_apices != -1)
        original_apices = peaks.aggregate_data.apex_pointer[selection]
        original_push_apices = cls.convert_apices_to_push_apices(
            original_apices,
            tof_indptr,
        )
        rt_apices = original_push_apices // cycle_shape[2]
        push_apices = rt_apices * cycle_shape[2] + im_apices[selection]
        push_indptr = cls.index_push_apices(
            push_apices,
            tof_indptr,
        )
        return cls(
            indptr=push_indptr,
            values=selection,
            axis_shape=cls.get_axis_shape(
                cycle_shape,
                tof_indptr,
            ),
        )
