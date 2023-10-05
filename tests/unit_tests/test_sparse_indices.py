#external
import numpy as np
import pytest

#local
import alphasynchro.data.sparse_indices


@pytest.fixture(scope="module")
def sparse_index():
    indptr = np.array([0, 2, 2, 5])
    values = np.array([1, 2, 3, 4, 5])
    sparse_index = alphasynchro.data.sparse_indices.SparseIndex(
        indptr=indptr,
        values=values
    )
    return sparse_index


def test_size(sparse_index):
    assert sparse_index.size == 3


def test_shape(sparse_index):
    assert sparse_index.shape == (3, 5)


def test_len(sparse_index):
    assert len(sparse_index) == 3


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, True),
        (1, True),
        (2, True),
        (-1, False),
        (3, False),
    ]
)
def test_is_valid(sparse_index, input, expected):
    output = sparse_index.is_valid(input)
    assert output is expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, False),
        (1, True),
        (2, False),
        (-1, True),
        (3, True),
    ]
)
def test_is_empty(sparse_index, input, expected):
    output = sparse_index.is_empty(input)
    assert output is expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, 2),
        (1, 0),
        (2, 3),
        (-1, 0),
        (3, 0),
    ]
)
def test_get_size(sparse_index, input, expected):
    output = sparse_index.get_size(input)
    assert output== expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, (0, 2)),
        (1, (2, 2)),
        (2, (2, 5)),
        (-1, (0, 0)),
        (3, (0, 0)),
    ]
)
def test_get_boundaries(sparse_index, input, expected):
    output = sparse_index.get_boundaries(input)
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, np.array([1, 2])),
        (1, np.array([])),
        (2, np.array([3, 4, 5])),
        (-1, np.array([])),
        (3, np.array([])),
    ]
)
def test_get_values(sparse_index, input, expected):
    output = sparse_index.get_values(input)
    assert np.array_equal(output, expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            np.array([0]),
            alphasynchro.data.sparse_indices.SparseIndex(
                indptr=np.array([0, 2]),
                values=np.array([1, 2]),
            )
        ),
        (
            np.array([1]),
            alphasynchro.data.sparse_indices.SparseIndex(
                indptr=np.array([0, 0]),
                values=np.array([]),
            )
        ),
        (
            np.array([2,0]),
            alphasynchro.data.sparse_indices.SparseIndex(
                    indptr=np.array([0, 3, 5]),
                    values=np.array([3, 4, 5, 1, 2]),
            )
        ),
    ]
)
def test_filter(sparse_index, input, expected):
    output = sparse_index.filter(input)
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            np.array([False, True, False, False, True]),
            alphasynchro.data.sparse_indices.SparseIndex(
                indptr=np.array([0, 1, 1, 2]),
                values=np.array([2, 5]),
            )
        ),
    ]
)
def test_filter_values(sparse_index, input, expected):
    output = sparse_index.filter_values(input)
    assert output == expected
