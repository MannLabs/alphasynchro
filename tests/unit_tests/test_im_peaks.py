#external
import numpy as np

#local
import alphasynchro.io.hdf
import alphasynchro.ms.peaks.indexed.im_peaks


def test_from_data_space():
    tof_indptr = np.array([0, 4, 4, 5])
    cycle_shape = (1, 3, 4, 5)
    peaks = alphasynchro.ms.peaks.peaks.Peaks.from_clusters_hdf(
        "./unit_tests/test_clusters.hdf",
        indices=np.array([0], dtype=np.int64)
    )
    im_apices = np.array([0])
    push_indexed_ims = alphasynchro.ms.peaks.indexed.im_peaks.PushIndexedImPeaks.from_data_space(
        peaks=peaks,
        cycle_shape=cycle_shape,
        tof_indptr=tof_indptr,
        im_apices=im_apices,
    )
    axis_shape = np.array([1, 3, 4])
    indptr = np.array([0, 1, 1, 1])
    values = np.array([0])
    assert np.array_equal(push_indexed_ims.axis_shape, axis_shape)
    assert np.array_equal(push_indexed_ims.indptr, indptr)
    assert np.array_equal(push_indexed_ims.values, values)
