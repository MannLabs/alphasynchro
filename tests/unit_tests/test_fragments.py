#external
import numpy as np
import pytest

#local
import alphasynchro.io.hdf
import alphasynchro.ms.peaks.fragments


@pytest.fixture(scope="module")
def test_peaks():
    test_peaks = alphasynchro.ms.peaks.fragments.Fragments.from_clusters_hdf(
        "./unit_tests/test_clusters.hdf",
    )
    return test_peaks


@pytest.mark.parametrize(
    "input, expected",
    [
        ("im_projection.indptr", np.array([0, 3, 5])),
        ("rt_projection.values", np.array([1, .9, 1])),
    ]
)
def test_select_peaks(test_peaks, input, expected):
    current_object = test_peaks
    for attr in input.split("."):
        current_object = current_object.__getattribute__(attr)
    output = current_object
    assert np.array_equal(output, expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, 2),
        (1, 2),
        (2, 1),
        (3, 1),
        (4, 0),
    ]
)
def test_min_fragments(input, expected):
    test_peaks2 = alphasynchro.ms.peaks.fragments.Fragments.from_clusters_hdf(
        "./unit_tests/test_clusters.hdf",
        min_fragment_size=input
    )
    assert len(test_peaks2)==expected
