#external
import numpy as np
import pytest

#local
import alphasynchro.ms.dimensions.push_matching


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, []),
        (105, []),
        (0, [0, 1, 21, 22]),
        (1, [0, 1, 2, 21, 22, 23]),
        (10, [9, 10, 11, 30, 31, 32]),
        (50, [28, 29, 30, 49, 50, 51, 70, 71, 72]),
        (63, [42, 43, 63, 64, 84, 85]),
    ]
)
def test_push_index(input, expected):
    push_index = input
    shape = (5, 3, 7)
    cycle_tolerance = 1
    scan_tolerance = 1
    output = list(
        alphasynchro.ms.dimensions.push_matching.generate_neighbor_push_indices(
            push_index,
            shape,
            scan_tolerance,
            cycle_tolerance,
        )
    )
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ((1, 1, 1), []),
        ((100, 10, 10), [10, 11, 110, 111]),
        ((100, 10, 100), [9, 10, 11, 1009, 1010, 1011]),
    ]
)
def test_shape(input, expected):
    push_index = 10
    shape = input
    cycle_tolerance = 1
    scan_tolerance = 1
    output = list(
        alphasynchro.ms.dimensions.push_matching.generate_neighbor_push_indices(
            push_index,
            shape,
            scan_tolerance,
            cycle_tolerance,
        )
    )
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, [9, 10, 11]),
        (1, [9, 10, 11, 30, 31, 32]),
        (2, [9, 10, 11, 30, 31, 32, 51, 52, 53]),
    ]
)
def test_cycle_tolerance(input, expected):
    push_index = 10
    shape = (5, 3, 7)
    cycle_tolerance = input
    scan_tolerance = 1
    output = list(
        alphasynchro.ms.dimensions.push_matching.generate_neighbor_push_indices(
            push_index,
            shape,
            scan_tolerance,
            cycle_tolerance,
        )
    )
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (0, [10, 31]),
        (1, [9, 10, 11, 30, 31, 32]),
        (2, [8, 9, 10, 11, 12, 29, 30, 31, 32, 33]),
    ]
)
def test_scan_tolerance(input, expected):
    push_index = 10
    shape = (5, 3, 7)
    cycle_tolerance = 1
    scan_tolerance = input
    output = list(
        alphasynchro.ms.dimensions.push_matching.generate_neighbor_push_indices(
            push_index,
            shape,
            scan_tolerance,
            cycle_tolerance,
        )
    )
    assert output == expected
