#external
import numpy as np
import pandas as pd
import pytest

#local
import alphasynchro.algorithms.calibration


@pytest.fixture(scope="module")
def calibrator():
    calibrator = alphasynchro.algorithms.calibration.TransmissionCalibrator(
        indexed_efficiency=np.array(
            [
                [1, .5, .2],
                [.9, .4, .1]
            ]
        ),
    )
    return calibrator


@pytest.mark.parametrize(
    "input, expected",
    [
        (.09, .9),
        (.19, .4),
        (.20, .1),
        (.30, 0.),
        (.31, 0.),
        (31, 0.),
        (-.09, 1.),
        (-.19, .5),
        (-.20, .2),
        (-.30, 0.),
        (-.31, 0.),
        (-31, 0.),
        (0, .9),
    ]
)
def test_get_efficiency(calibrator, input, expected):
    output = calibrator.get_efficiency(input)
    assert np.array_equal(output, expected)
