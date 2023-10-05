#external
import numpy as np
import pytest

#local
import alphasynchro.io.hdf
import alphasynchro.ms.peaks.precursors


@pytest.fixture(scope="module")
def test_peaks():
    test_peaks = alphasynchro.ms.peaks.precursors.Precursors.from_clusters_hdf(
        "./unit_tests/test_clusters.hdf",
    )
    return test_peaks


@pytest.mark.parametrize(
    "input, expected",
    [
        ("im_projection.indptr", np.array([0, 2])),
        ("rt_projection.values", np.array([.1, .2, 1])),
    ]
)
def test_select_peaks(test_peaks, input, expected):
    current_object = test_peaks
    for attr in input.split("."):
        current_object = current_object.__getattribute__(attr)
    output = current_object
    assert np.array_equal(output, expected)
