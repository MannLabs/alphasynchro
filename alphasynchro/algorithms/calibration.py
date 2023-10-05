#!python


# external
import numpy as np
import scipy.signal

# local
import alphasynchro.ms.peaks.indexed.mz_peaks
import alphasynchro.performance.compiling
import alphasynchro.ms.dimensions.push_matching
import alphasynchro.ms.dimensions.mz_matching
import alphasynchro.algorithms.matching.matching


@alphasynchro.performance.compiling.njit_dataclass
class TransmissionCalibrator:

    # mz_offsets: np.ndarray
    # efficiency: np.ndarray
    unfragmented_pairs: np.ndarray = np.array([], dtype=np.int64)
    indexed_efficiency: np.ndarray
    decimals: int = 1

    @classmethod
    def from_unfragmented_pairs(
        cls,
        indexed_fragments,
        indexed_precursors,
        monoisotopic_precursors,
        fragments,
        cycle,
        smooth_factor=10,
        max_mz=100,
        decimals=1,
        most_intense_count=10000,
    ):
        um = alphasynchro.algorithms.matching.matching.UnfragmentedMatcherMultithreaded(
            indexed_precursors=indexed_precursors,
            indexed_fragments=indexed_fragments,
        )
        unfragmented_pairs = um.match_all()
        unfragmented_pairs = unfragmented_pairs[
            np.argpartition(
                fragments.aggregate_data.summed_intensity[unfragmented_pairs[:,1]],
                -most_intense_count
            )[-most_intense_count:]
        ]
        cycle_center = (np.sum(cycle, axis=-1) / 2)[0]
        eff_dict = {}
        for prec_index, frag_index in unfragmented_pairs:
            im_projection = fragments.im_projection.get_pdf(frag_index)
            im_projection /= np.max(im_projection)
            frame = fragments.aggregate_data.frame_group[frag_index]
            start_offset = fragments.im_projection.get_start_offset(frag_index)
            end_offset = fragments.im_projection.get_end_offset(frag_index)
            mz_centers = np.copy(cycle_center[frame, start_offset: end_offset])
            mz = monoisotopic_precursors.aggregate_data.mz_weighted_average[prec_index]
            mz_centers -= mz

            prec_im_projection = monoisotopic_precursors.im_projection.get_pdf(prec_index)
            prec_im_projection /= np.max(prec_im_projection)
            prec_start_offset = monoisotopic_precursors.im_projection.get_start_offset(prec_index)
            prec_end_offset = monoisotopic_precursors.im_projection.get_end_offset(prec_index)

            smooth = scipy.signal.convolve(im_projection, np.ones(smooth_factor)/smooth_factor, mode='same')
            prec_smooth = scipy.signal.convolve(prec_im_projection, np.ones(smooth_factor)/smooth_factor, mode='same')

            efficiency = np.zeros(prec_end_offset - prec_start_offset)
            efficiency_slice = efficiency[:]
            target_slice = smooth
            if prec_start_offset < start_offset:
                efficiency_slice = efficiency_slice[start_offset - prec_start_offset:]
            else:
                target_slice = target_slice[prec_start_offset - start_offset:]
            if len(efficiency_slice) > len(target_slice):
                efficiency_slice = efficiency_slice[:len(target_slice)]
            else:
                target_slice = target_slice[:len(efficiency_slice)]

            efficiency_slice[:] = target_slice[:]
            efficiency /= prec_smooth

            mz_centers = np.copy(cycle_center[frame, prec_start_offset: prec_end_offset])
            mz = monoisotopic_precursors.aggregate_data.mz_weighted_average[prec_index]
            mz_centers -= mz

            for mz, eff in zip(np.round(mz_centers, decimals), efficiency):
                if mz not in eff_dict:
                    eff_dict[mz] = []
                eff_dict[mz].append(eff)
        eff_dict = {
            mz: np.median(np.array(eff)[np.isfinite(eff)]) for mz, eff in sorted(eff_dict.items())
        }
        # plt.plot(*zip(*eff_dict2.items()))
        indexed_efficiency = np.zeros((2, int(max_mz * smooth_factor)))
        for mz, eff in eff_dict.items():
            sign = int(mz > 0)
            index = int(abs(mz) * 10**decimals)
            if index < indexed_efficiency.shape[1]:
                indexed_efficiency[sign, index] = eff
        indexed_efficiency[:, 0] = (np.max(indexed_efficiency), np.max(indexed_efficiency))
        return cls(
            # mz_offsets=eff_dict.keys(),
            # efficiency=eff_dict.values(),
            unfragmented_pairs=unfragmented_pairs,
            indexed_efficiency=indexed_efficiency,
            decimals=decimals,
        )

    @alphasynchro.performance.compiling.njit
    def get_efficiency(
        self,
        mz_distance,
    ):
        sign = int(mz_distance >= 0)
        index = int(abs(mz_distance) * 10**self.decimals)
        efficiency = 0.
        if index < self.indexed_efficiency.shape[1]:
            efficiency = self.indexed_efficiency[sign, index]
        return efficiency
