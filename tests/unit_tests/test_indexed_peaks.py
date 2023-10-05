#external
import numpy as np
import pytest

#local
import alphasynchro.io.hdf
import alphasynchro.ms.peaks.indexed.indexed_peaks


@pytest.fixture(scope="module")
def tof_indptr():
    tof_indptr = np.array([0, 4, 4, 5])
    return tof_indptr


def test_convert_apices_to_push_apices(tof_indptr):
    apices = np.arange(5)
    push_apices = alphasynchro.ms.peaks.indexed.indexed_peaks.IndexedPeaks.convert_apices_to_push_apices(
        apices,
        tof_indptr,
    )
    expected = np.array([0, 0, 0, 0, 2])
    assert np.array_equal(push_apices, expected)


def test_index_push_apices(tof_indptr):
    push_apices = np.array([0, 1, 0, 2])
    push_indptr = alphasynchro.ms.peaks.indexed.indexed_peaks.IndexedPeaks.index_push_apices(
        push_apices,
        tof_indptr,
    )
    expected = np.array([0, 2, 3, 4])
    assert np.array_equal(push_indptr, expected)


def test_get_axis_shape(tof_indptr):
    cycle_shape = (1, 3, 4, 5)
    axis_shape = alphasynchro.ms.peaks.indexed.indexed_peaks.IndexedPeaks.get_axis_shape(
        cycle_shape,
        tof_indptr,
    )
    expected = np.array([1, 3, 4])
    assert np.array_equal(axis_shape, expected)
