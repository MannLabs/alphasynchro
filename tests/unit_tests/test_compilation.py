# external
import numpy as np
import pytest
import numba
import pandas as pd

# local
import alphasynchro.performance.compiling


class PDDummy(pd.DataFrame):

    def __init__(self):
        super().__init__(
            {
                "vals": np.arange(4)**2,
            }
        )

    def __hash__(self):
        return 0


@alphasynchro.performance.compiling.njit_dataclass
class Dummy:

    arr: np.ndarray = np.arange(4)**2
    pd_df: PDDummy = PDDummy()

    def normal_func(self, index):
        return self.arr[index]

    @numba.njit
    def sub_njit_func(self, index: int) -> int: # pragma: no cover
        return (self.arr[index] + self.pd_df.vals[index]) / 2

    @numba.njit
    def njit_func(self, index: int) -> int: # pragma: no cover
        return self.sub_njit_func(index)



@pytest.fixture(scope="module")
def dummy():
    dummy = Dummy()
    return dummy


def test_arr(dummy):
    assert np.array_equal(dummy.arr, np.arange(4)**2)


def test_eq(dummy):
    dummy2 = Dummy()
    assert id(dummy) != id(dummy2)
    assert hash(dummy) == hash(dummy2)
    assert dummy == dummy2


def test_neq(dummy):
    dummy2 = Dummy(arr=np.arange(4)**3)
    assert id(dummy) != id(dummy2)
    assert hash(dummy) != hash(dummy2)
    assert dummy != dummy2


@pytest.mark.parametrize(
    "input, expected",
    [
        ("normal_func", False),
        ("sub_njit_func", True),
        ("njit_func", True),
    ]
)
def test_is_njit_func(dummy, input, expected):
    output = eval(f"dummy.{input}")
    assert isinstance(
        output,
        numba.core.registry.CPUDispatcher
    ) is expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, 0),
        (1, 1),
        (2, 4),
        (3, 9),
    ]
)
def test_normal_func(dummy, input, expected):
    output = dummy.normal_func(input)
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, 0),
        (1, 1),
        (2, 4),
        (3, 9),
    ]
)
def test_sub_njit_func(dummy, input, expected):
    output = dummy.sub_njit_func(input)
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, 0),
        (1, 1),
        (2, 4),
        (3, 9),
    ]
)
def test_njit_func(dummy, input, expected):
    output = dummy.njit_func(input)
    assert output == expected
