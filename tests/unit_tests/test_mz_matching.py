#external
import numpy as np
import pytest

#local
import alphasynchro.ms.dimensions.mz_matching


@pytest.fixture(scope="module")
def mz_values1():
    mz_values1 = np.array([100, 200, 500, 1000])
    return mz_values1


@pytest.fixture(scope="module")
def mz_values2():
    mz_values2 = np.array([100, 200 * (1 + 49 * 10**-6), 1000 * (1 + 51 * 10**-6), 1000 * (1 + 99 * 10**-6)])
    return mz_values2


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, [(0, 0)]),
        (30, [(0, 0)]),
        (50, [(0, 0), (1, 1)]),
        (52, [(0, 0), (1, 1), (3, 2)]),
        (100, [(0, 0), (1, 1), (3, 2), (3, 3)]),
    ]
)
def test_match_mz_arrays(mz_values1, mz_values2, input, expected):
    output = list(
        alphasynchro.ms.dimensions.mz_matching.match_mz_arrays(
            mz_values1,
            mz_values2,
            ppm_tolerance=input,
        )
    )
    assert output == expected
