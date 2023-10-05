#!python
'''Module to index peaks by their push index.'''


# external
import numpy as np

# local
import alphasynchro.performance.compiling
import alphasynchro.ms.peaks.peaks
import alphasynchro.data.sparse_indices


@alphasynchro.performance.compiling.njit_dataclass
class IndexedPeaks(alphasynchro.data.sparse_indices.SparseIndex):

    axis_shape: tuple[int, int, int]

    @classmethod
    def convert_apices_to_push_apices(
        cls,
        apices: np.ndarray,
        tof_indptr: np.ndarray,
    ) -> np.ndarray:
        push_apices = np.searchsorted(
            tof_indptr,
            apices,
            "right",
        ) - 1
        return push_apices

    @classmethod
    def index_push_apices(
        cls,
        push_apices: np.ndarray,
        tof_indptr: np.ndarray,
    ) -> np.ndarray:
        push_indptr = np.bincount(push_apices, minlength=len(tof_indptr))
        push_indptr[1:] = np.cumsum(push_indptr[:-1])
        push_indptr[0] = 0
        return push_indptr

    @classmethod
    def get_axis_shape(
        cls,
        cycle_shape: tuple,
        tof_indptr: np.ndarray,
    ) -> np.ndarray:
        axis_shape = (
            int(
                np.ceil(
                    len(tof_indptr) / np.prod(
                        cycle_shape[1:3], dtype=np.int64
                    )
                )
            ),
            *cycle_shape[1:3]
        )
        return axis_shape
