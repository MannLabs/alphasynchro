#!python
'''Module to find apices of (smoothed) CDFs.'''


# external
import numpy as np

# local
import alphasynchro.performance.compiling
import alphasynchro.stats.distributions


@alphasynchro.performance.compiling.njit_dataclass
class SmoothApexFinder:

    cdf: alphasynchro.stats.distributions.CDFWithOffsetAndSummedValues
    smooth_sigma: int = 5

    def __post_init__(self):
        bins = np.arange(-3 * self.smooth_sigma, 3 * self.smooth_sigma + 1)
        normalization_constant = 1 / (self.smooth_sigma * np.sqrt(2 * np.pi))
        exponents = -bins**2 / (2 * self.smooth_sigma**2)
        smooth_array = normalization_constant * np.exp(exponents)
        object.__setattr__(self, "smooth_array", smooth_array)

    def calculate_all(self) -> np.ndarray:
        apices = np.empty(self.cdf.shape[0], dtype=np.int64)
        for index in range(self.cdf.shape[0]):
            apices[index] = self.calculate(index)
        return apices

    def calculate(self, index: int) -> np.ndarray:
        im_apex = -1
        summed_value = self.cdf.get_summed_value(index)
        if summed_value > 0:
            smooth_distribution = self.cdf.smooth(index, self.smooth_array)
            im_apex = np.argmax(smooth_distribution) + self.cdf.start_offsets[index]
        return im_apex
