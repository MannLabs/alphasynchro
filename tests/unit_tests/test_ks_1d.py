#external
import numpy as np
import pytest

#local
import alphasynchro.data.sparse_indices
import alphasynchro.stats.distributions
import alphasynchro.stats.ks_1d


@pytest.fixture(scope="module")
def ks_tester():
    indptr = np.array([0, 2, 3, 6, 8])
    values = np.array([.5, 1.0, 1.0, .2, .5, 1., .3, 1.0])
    start_offsets = np.array([1, 1, 0, 2], dtype=np.int64)
    cdf_with_offset = alphasynchro.stats.distributions.CDFWithOffset(
        indptr=indptr,
        values=values,
        start_offsets=start_offsets,
    )
    ks_tester = alphasynchro.stats.ks_1d.KSTester1DMultithreaded(
        cdf_with_offset=cdf_with_offset,
    )
    return ks_tester


@pytest.mark.parametrize(
    "input, expected",
    [
        ((0, 0), 0),
        ((1, 0), .5),
        ((2, 0), .2),
        ((3, 0), .7),
        ((0, 1), .5),
        ((0, 2), .2),
        ((0, 3), .7),
        ((2, 3), .7),
        ((3, 2), .7),
    ]
)
def test_calculate(ks_tester, input, expected):
    output = ks_tester.calculate(*input)
    assert output == expected


def test_calculate_all(ks_tester):
    input_data = np.array(
        [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (2, 3),
            (3, 2),
        ]
    )
    expected = np.array([0, .5, .2, .7, .5, .2, .7, .7, .7])
    output = ks_tester.calculate_all(input_data)
    assert np.array_equal(output, expected)


@pytest.fixture(scope="module")
def paired_ks_tester(ks_tester):
    paired_ks_tester = alphasynchro.stats.ks_1d.KSTester1DPairedMultithreaded(
        cdf_with_offset=ks_tester.cdf_with_offset,
        secondary_cdf_with_offset=ks_tester.cdf_with_offset,
    )
    return paired_ks_tester


def test_calculate_all_paired(paired_ks_tester):
    input_data = np.array(
        [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (2, 3),
            (3, 2),
        ]
    )
    expected = np.array([0, .5, .2, .7, .5, .2, .7, .7, .7])
    output = paired_ks_tester.calculate_all(input_data)
    assert np.array_equal(output, expected)


def test_calculate_all_no_offsets():
    input_data = np.array(
        [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (2, 3),
            (3, 2),
        ]
    )
    indptr = np.array([0, 2, 3, 6, 8])
    values = np.array([.5, 1.0, 1.0, .2, .5, 1., .3, 1.0])
    cdf_with_offset = alphasynchro.stats.distributions.CDF(
        indptr=indptr,
        values=values,
    )
    paired_ks_tester = alphasynchro.stats.ks_1d.KSTester1DNoOffsetPairedMultithreaded(
        cdf_with_offset=cdf_with_offset,
        secondary_cdf_with_offset=cdf_with_offset,
    )
    expected = np.array([0.0, 0.5, 0.5, 0.2, 0.5, 0.5, 0.2, 0.5, 0.5])
    output = paired_ks_tester.calculate_all(input_data)
    assert np.array_equal(output, expected)
