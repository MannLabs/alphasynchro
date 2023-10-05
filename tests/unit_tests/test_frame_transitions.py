#external
import numpy as np
import pytest

#local
import alphasynchro.ms.peaks.fragments
import alphasynchro.ms.transitions.frame_transitions


@pytest.fixture(scope="module")
def transitions():
    test_transitions = alphasynchro.ms.transitions.frame_transitions.Transitions(
        indptr=np.array([0, 2, 5, 6]),
        values=np.array([4, 5, 0, 1, 2, 3]),
        weights=np.array([.1, .9, .1, .2, .7, .5]),
        precursor_indices=np.arange(3),
    )
    return test_transitions


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, np.array([])),
        (0, np.array([.1, .9])),
        (1, np.array([.1, .2, .7])),
        (2, np.array([.5])),
        (4, np.array([])),
    ]
)
def test_weights(transitions, input, expected):
    output = transitions.get_weights(input)
    assert np.array_equal(output, expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, -1),
        (0, 0),
        (1, 1),
        (2, 2),
        (4, -1),
    ]
)
def test_precursor_index(transitions, input, expected):
    output = transitions.get_precursor_index(input)
    assert np.array_equal(output, expected)


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            np.array([0]),
            alphasynchro.ms.transitions.frame_transitions.Transitions(
                indptr=np.array([0, 2]),
                values=np.array([4, 5]),
                weights=np.array([.1, .9]),
                precursor_indices=np.array([0]),
            )
        ),
        (
            np.array([1]),
            alphasynchro.ms.transitions.frame_transitions.Transitions(
                indptr=np.array([0, 3]),
                values=np.array([0, 1, 2]),
                weights=np.array([.1, .2, .7]),
                precursor_indices=np.array([1]),
            )
        ),
        (
            np.array([2,0]),
            alphasynchro.ms.transitions.frame_transitions.Transitions(
                indptr=np.array([0, 1, 3]),
                values=np.array([3, 4, 5]),
                weights=np.array([.5, .1, .9]),
                precursor_indices=np.array([2, 0]),
            )
        ),
    ]
)
def test_filter(transitions, input, expected):
    output = transitions.filter(input)
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            np.arange(4),
            alphasynchro.ms.transitions.frame_transitions.Transitions(
                indptr=np.array([0, 2, 4, 4]),
                values=np.array([4, 5, 0, 1]),
                weights=np.array([.1, .9, .1, .2]),
                precursor_indices=np.arange(3),
            )
        ),
        (
            np.array([1, 5]),
            alphasynchro.ms.transitions.frame_transitions.Transitions(
                indptr=np.array([0, 1, 1, 2]),
                values=np.array([5, 3]),
                weights=np.array([.9, .5]),
                precursor_indices=np.arange(3),
            )
        ),
    ]
)
def test_filter_weights(transitions, input, expected):
    output = transitions.filter_weights(input)
    assert output == expected
