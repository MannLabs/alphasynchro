#external
import numpy as np
import pytest
import scipy.signal

#local
import alphasynchro.stats.distributions
import alphasynchro.stats.apex_finder


@pytest.fixture(scope="module")
def cdf():
    indptr = np.array([0, 10, 13, 15])
    values = np.arange(15, dtype=np.float64)
    values[:10] /= 10
    values[10] = .5
    values[11] = .5
    values[12] = 1
    values[13:] = 0
    start_offsets = np.array([0, 4, 2], dtype=np.int64)
    summed_values = np.array([1, 2, 0])
    cdf_with_offset_and_summed_values = alphasynchro.stats.distributions.CDFWithOffsetAndSummedValues(
        indptr=indptr,
        values=values,
        start_offsets=start_offsets,
        summed_values=summed_values,
    )
    return cdf_with_offset_and_summed_values


@pytest.fixture(scope="module")
def apex_finder(cdf):
    apex_finder = alphasynchro.stats.apex_finder.SmoothApexFinder(
        cdf=cdf,
    )
    return apex_finder


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, -1),
        (0, 5),
        (1, 5),
        (2, -1),
        (3, -1),
    ]
)
def test_calculate_apex(apex_finder, input, expected):
    output = apex_finder.calculate(input)
    assert output == expected


def test_calculate_all(apex_finder):
    output = apex_finder.calculate_all()
    expected = np.array([5, 5, -1])
    assert np.array_equal(output, expected)
