#external
import numpy as np
import pandas as pd
import pytest

#local
import alphasynchro.algorithms.matching.matching
import alphasynchro.ms.peaks.indexed.mz_peaks

@pytest.fixture(scope="module")
def unfragmented_matcher():
    indexed_precursors = alphasynchro.ms.peaks.indexed.mz_peaks.PushIndexedMzs(
        indptr=np.array([0, 2, 2, 2, 3, 3, 3], dtype=np.int64),
        values=np.array([1000, 1002, 900.0001]),
        axis_shape=(2, 3, 1),
    )
    indexed_fragments = alphasynchro.ms.peaks.indexed.mz_peaks.PushIndexedMzs(
        indptr=np.array([0, 0, 2, 3, 3, 3, 3], dtype=np.int64),
        values=np.array([900, 999.99, 1000.0001,]),
        axis_shape=(2, 3, 1),
    )
    unfragmented_matcher = alphasynchro.algorithms.matching.matching.UnfragmentedMatcherMultithreaded(
        indexed_precursors=indexed_precursors,
        indexed_fragments=indexed_fragments,
    )
    return unfragmented_matcher


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, 2),
        (1, 0),
        (2, 0),
        (3, 1),
    ]
)
def test_count(unfragmented_matcher, input, expected):
    output = unfragmented_matcher.count_matches(input)
    assert output == expected


def test_count_all(unfragmented_matcher):
    expected = np.array([2, 0, 0, 1, 0, 0])
    output = unfragmented_matcher.count_all()
    assert np.array_equal(output, expected)


def test_match_all(unfragmented_matcher):
    expected = np.array(
        [
            [0, 1],
            [0, 2],
            [2, 0]
        ]
    )
    output = unfragmented_matcher.match_all()
    print(expected, output)
    assert np.array_equal(output, expected)
