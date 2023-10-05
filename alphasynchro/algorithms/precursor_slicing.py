#!python


# external
import numpy as np

# local
import alphasynchro.performance.compiling
import alphasynchro.ms.peaks.precursors
import alphasynchro.algorithms.calibration
import alphasynchro.stats.distributions


@alphasynchro.performance.compiling.njit_dataclass
class SlicedIMDistribution:

    precursors: alphasynchro.ms.peaks.precursors.Precursors
    calibration: alphasynchro.algorithms.calibration.TransmissionCalibrator
    cycle_center: np.ndarray

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_transmitted_projection_cdf(
        self,
        index: int,
        frame: int,
    ) -> np.ndarray:
        cdf = self.multiply(index, frame)
        cdf = np.cumsum(cdf)
        cdf = self.normalize_cdf(cdf)
        return cdf

    @alphasynchro.performance.compiling.njit(nogil=True)
    def normalize_cdf(
        self,
        cdf: np.ndarray,
    ) -> np.ndarray:
        if cdf[-1] > 0:
            cdf /= cdf[-1]
        return cdf

    @alphasynchro.performance.compiling.njit(nogil=True)
    def multiply(
        self,
        index: int,
        frame: int,
    ) -> np.ndarray:
        distribution = self.precursors.im_projection.get_pdf(index)
        efficiency = self.get_efficiency(index, frame)
        return efficiency * distribution

    @alphasynchro.performance.compiling.njit(nogil=True)
    def get_efficiency(
        self,
        index: int,
        frame: int,
    ) -> np.ndarray:
        precursor_mz = self.precursors.aggregate_data.mz_weighted_average[index]
        start_offset = self.precursors.im_projection.get_start_offset(index)
        end_offset = self.precursors.im_projection.get_end_offset(index)
        result_buffer = np.copy(self.cycle_center[frame, start_offset: end_offset])
        for buffer_index, mz in enumerate(result_buffer):
            mz_difference = mz - precursor_mz
            result_buffer[buffer_index] = self.calibration.get_efficiency(
                mz_difference
            )
        return result_buffer


@alphasynchro.performance.compiling.njit_dataclass
class SlicedIMDistributionMultithreaded(SlicedIMDistribution):

    def calculate_all_transmitted_cdf_for_frame(
        self,
        frame: int
    ) -> np.ndarray[float]:
        new_values = np.empty(len(self.precursors.im_projection.values))
        summed_values = np.empty(len(self.precursors))
        alphasynchro.performance.multithreading.parallel(
            self.calculate_from_buffers
        )(
            range(len(self.precursors)),
            new_values,
            summed_values,
            frame,
        )
        im_projection = alphasynchro.stats.distributions.CDFWithOffsetAndSummedValues(
            indptr=self.precursors.im_projection.indptr,
            values=new_values,
            start_offsets=self.precursors.im_projection.start_offsets,
            summed_values=summed_values,
        )
        return im_projection

    @alphasynchro.performance.compiling.njit(nogil=True)
    def calculate_from_buffers(
        self,
        index: int,
        cdf_values: np.ndarray,
        summed_values: np.ndarray,
        frame: int,
    ) -> None:
        cdf = self.multiply(index, frame)
        cdf = np.cumsum(cdf)
        summed_values[index] = cdf[-1]
        cdf = self.normalize_cdf(cdf)
        start, end = self.precursors.im_projection.get_boundaries(index)
        cdf_values[start: end] = self.normalize_cdf(cdf)
