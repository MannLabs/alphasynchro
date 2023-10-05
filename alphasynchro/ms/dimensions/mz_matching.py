#!python
'''Module to match indices of mz arrays.'''


# external
import numpy as np

# local
import alphasynchro.performance.compiling


@alphasynchro.performance.compiling.njit(nogil=True)
def match_mz_arrays(
    mz_values1: np.ndarray,
    mz_values2: np.ndarray,
    ppm_tolerance: float = 50.0
) -> (tuple[int, int]):
    start2 = 0
    for index1, mz_value1 in enumerate(mz_values1):
        for index2, mz_value2 in enumerate(mz_values2[start2:], start2):
            ppm_difference = (mz_value2 - mz_value1) * 2 / (mz_value1 + mz_value2) * 10**6
            if ppm_difference > ppm_tolerance:
                break
            if ppm_difference < -ppm_tolerance:
                start2 += 1
            else:
                yield index1, index2
        else:
            break
