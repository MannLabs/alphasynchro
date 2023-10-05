#!python
'''Module to calculate Kolmogorov-Smirnov distances from various CDFs.'''


# local
import alphasynchro.performance.compiling
import alphasynchro.performance.multithreading
import alphasynchro.stats.distributions

# external
import numpy as np


@alphasynchro.performance.compiling.njit_dataclass
class KSTester1D:

    cdf_with_offset: alphasynchro.stats.distributions.CDFWithOffset
    threshold: float = 1.0

    @alphasynchro.performance.compiling.njit(nogil=True)
    def calculate(self, index1: int, index2: int) -> float:
        max_diff, cdf1, cdf2 = self.trim_left_edge_of_cdfs(index1, index2)
        cdf1, cdf2 = self.trim_right_edge_of_cdfs(cdf1, cdf2)
        max_diff = self.check_overlapping_part_of_cdfs(cdf1, cdf2, max_diff)
        return max_diff

    @alphasynchro.performance.compiling.njit(nogil=True)
    def trim_left_edge_of_cdfs(
        self,
        index1: int,
        index2: int,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        cdf1 = self.cdf_with_offset.get_cdf(index1)
        cdf2 = self.cdf_with_offset.get_cdf(index2)
        start_offset1 = self.cdf_with_offset.get_start_offset(index1)
        start_offset2 = self.cdf_with_offset.get_start_offset(index2)
        if start_offset1 < start_offset2:
            max_diff = cdf1[start_offset2 - start_offset1 - 1]
            cdf1 = cdf1[start_offset2 - start_offset1:]
        elif start_offset2 < start_offset1:
            max_diff = cdf2[start_offset1 - start_offset2 - 1]
            cdf2 = cdf2[start_offset1 - start_offset2:]
        else:
            max_diff = 0
        return  max_diff, cdf1, cdf2

    @alphasynchro.performance.compiling.njit(nogil=True)
    def trim_right_edge_of_cdfs(
        self,
        cdf1: np.ndarray,
        cdf2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(cdf1) > len(cdf2):
            cdf1 = cdf1[:len(cdf2)]
        else:
            cdf2 = cdf2[:len(cdf1)]
        return cdf1, cdf2

    @alphasynchro.performance.compiling.njit(nogil=True)
    def check_overlapping_part_of_cdfs(
        self,
        cdf1: np.ndarray,
        cdf2: np.ndarray,
        max_diff = float
    ) -> float:
        for v1, v2 in zip(cdf1, cdf2):
            diff = v1 - v2
            if diff < 0:
                diff = -diff
            if diff > self.threshold:
                max_diff = self.threshold
                break
            if diff > max_diff:
                max_diff = diff
        return max_diff


@alphasynchro.performance.compiling.njit_dataclass
class KSTester1DMultithreadedInterface:

    def calculate_all(
        self,
        paired_indices: np.ndarray[int, int],
    ) -> np.ndarray[float]:
        ks_values = np.empty(len(paired_indices))
        alphasynchro.performance.multithreading.parallel(
            self.calculate_from_buffers
        )(
            range(len(ks_values)),
            ks_values,
            paired_indices,
        )
        return ks_values

    @alphasynchro.performance.compiling.njit(nogil=True)
    def calculate_from_buffers(
        self,
        index: int,
        ks_values: np.ndarray[float],
        paired_indices: np.ndarray[int, int],
    ) -> None:
        index1, index2 = paired_indices[index]
        ks_value = self.calculate(
            index1,
            index2,
        )
        ks_values[index] = ks_value


@alphasynchro.performance.compiling.njit_dataclass
class KSTester1DMultithreaded(KSTester1D, KSTester1DMultithreadedInterface):

    pass


@alphasynchro.performance.compiling.njit_dataclass
class KSTester1DPaired(KSTester1D):

    secondary_cdf_with_offset: alphasynchro.stats.distributions.CDFWithOffset

    @alphasynchro.performance.compiling.njit(nogil=True)
    def trim_left_edge_of_cdfs(
        self,
        index1: int,
        index2: int,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        cdf1 = self.cdf_with_offset.get_cdf(index1)
        cdf2 = self.secondary_cdf_with_offset.get_cdf(index2)
        start_offset1 = self.cdf_with_offset.get_start_offset(index1)
        start_offset2 = self.secondary_cdf_with_offset.get_start_offset(index2)
        if start_offset1 < start_offset2:
            max_diff = cdf1[start_offset2 - start_offset1 - 1]
            cdf1 = cdf1[start_offset2 - start_offset1:]
        elif start_offset2 < start_offset1:
            max_diff = cdf2[start_offset1 - start_offset2 - 1]
            cdf2 = cdf2[start_offset1 - start_offset2:]
        else:
            max_diff = 0
        return  max_diff, cdf1, cdf2


@alphasynchro.performance.compiling.njit_dataclass
class KSTester1DPairedMultithreaded(
    KSTester1DPaired,
    KSTester1DMultithreadedInterface
):

    pass


@alphasynchro.performance.compiling.njit_dataclass
class KSTester1DNoOffsetPaired(KSTester1D):

    cdf_with_offset: alphasynchro.stats.distributions.CDF
    secondary_cdf_with_offset: alphasynchro.stats.distributions.CDF
    threshold: float = 1.0

    @alphasynchro.performance.compiling.njit(nogil=True)
    def calculate(self, index1: int, index2: int) -> float:
        max_diff = 0
        cdf1 = self.cdf_with_offset.get_cdf(index1)
        cdf2 = self.secondary_cdf_with_offset.get_cdf(index2)
        max_diff = self.check_overlapping_part_of_cdfs(cdf1, cdf2, max_diff)
        return max_diff


@alphasynchro.performance.compiling.njit_dataclass
class KSTester1DNoOffsetPairedMultithreaded(
    KSTester1DNoOffsetPaired,
    KSTester1DMultithreadedInterface
):

    pass
