#external
import numpy as np
import pytest

#local
import alphasynchro.io.hdf
import alphasynchro.ms.peaks.indexed.mz_peaks


def test_from_data_space():
    tof_indptr = np.array([0, 4, 4, 5])
    cycle_shape = (1, 3, 4, 5)
    peaks = alphasynchro.ms.peaks.peaks.Peaks.from_clusters_hdf(
        "./unit_tests/test_clusters.hdf",
        indices=np.array([0], dtype=np.int64)
    )
    push_indexed_mzs = alphasynchro.ms.peaks.indexed.mz_peaks.PushIndexedMzs.from_data_space(
        peaks=peaks,
        cycle_shape=cycle_shape,
        tof_indptr=tof_indptr,
    )
    axis_shape = np.array([1, 3, 4])
    indptr = np.array([0, 1, 1, 1])
    assert np.array_equal(peaks.aggregate_data.mz_weighted_average, push_indexed_mzs.values)
    assert np.array_equal(axis_shape, push_indexed_mzs.axis_shape)
    assert np.array_equal(indptr, push_indexed_mzs.indptr)
