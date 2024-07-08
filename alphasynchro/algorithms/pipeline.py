#!python


# builtin
import logging
from typing import Any

# local
import alphasynchro.ms.peaks.precursors
import alphasynchro.ms.peaks.fragments
import alphasynchro.io.hdf
import alphasynchro.io.writing.mgf
import alphasynchro.algorithms.calibration
import alphasynchro.ms.peaks.indexed.mz_peaks
import alphasynchro.algorithms.precursor_slicing
import alphasynchro.ms.transitions.frame_transitions
import alphasynchro.algorithms.matching.matching
import alphasynchro.data.sparse_indices
import alphasynchro.stats.apex_finder
import alphasynchro.ms.peaks.indexed.im_peaks
import alphasynchro.stats.ks_1d
import alphasynchro.ms.peaks.merged_fragments
import alphasynchro.stats.distributions
import alphasynchro.io.writing.mgf
import alphasynchro.data.dataframe

# external
import numpy as np


class Pipeline:

    def __init__(
        self,
        analysis_file_name: str,
        overwrite: bool = False,
    ):
        analysis_file = alphasynchro.io.hdf.HDFObject.from_file(
            analysis_file_name,
            new=overwrite,
        )
        object.__setattr__(self, "analysis_file", analysis_file)

    def run(
        self,
        cluster_file_name: str,
        min_fragment_size: int = 0,
        smooth_factor: int = 10,
        max_mz: float = 100,
        decimals: int = 1,
        most_intense_count: int = 10000,
        max_rt_weight: float = .5,
        max_im_weight: float = .5,
        max_frame_weight: float = 1,
        min_peaks: int = 0,
        unique_transitions_only: bool = False,
        diapasef: bool = False,
    ) -> None:
        self.load_data_space(cluster_file_name)
        self.load_peaks(cluster_file_name, min_fragment_size)
        self.calibrate(
            smooth_factor=smooth_factor,
            max_mz=max_mz,
            decimals=decimals,
            most_intense_count=most_intense_count,
        )
        self.create_ms2_spectra(
            min_peaks=min_peaks,
            max_rt_weight=max_rt_weight,
            max_im_weight=max_im_weight,
            max_frame_weight=max_frame_weight,
            unique_transitions_only=unique_transitions_only,
            diapasef=diapasef,
        )

    def load_data_space(
        self,
        cluster_file_name: str,
    ) -> None:
        logging.info("Loading acquisition data...")
        hdf_cluster_object = alphasynchro.io.hdf.HDFObject.from_file(
            cluster_file_name
        )
        self.sample_name = hdf_cluster_object.sample_name
        self.cycle = hdf_cluster_object.acquisition.cycle
        self.tof_indptr = hdf_cluster_object.acquisition.tof_indptr

    def load_peaks(
        self,
        cluster_file_name,
        min_fragment_size: int = 0,
    ) -> None:
        logging.info("Loading peaks...")
        logging.info("Loading precursors...")
        self.monoisotopic_precursors = alphasynchro.ms.peaks.precursors.Precursors.from_clusters_hdf(
            cluster_file_name,
        )
        logging.info("Loading fragments...")
        self.fragments = alphasynchro.ms.peaks.fragments.Fragments.from_clusters_hdf(
            cluster_file_name,
            min_fragment_size=min_fragment_size,
        )
        logging.info("Indexing fragments...")
        self.indexed_fragments = alphasynchro.ms.peaks.indexed.mz_peaks.PushIndexedMzs.from_data_space(
            peaks=self.fragments,
            cycle_shape=self.cycle.shape,
            tof_indptr=self.tof_indptr,
        )
        logging.info("Finished loading peaks")

    def calibrate(
        self,
        smooth_factor=10,
        max_mz=100,
        decimals=1,
        most_intense_count=10000,
    ) -> None:
        logging.info("Calibrating quadrupole...")
        logging.info("Indexing precursors...")
        indexed_precursors = alphasynchro.ms.peaks.indexed.mz_peaks.PushIndexedMzs.from_data_space(
            peaks=self.monoisotopic_precursors,
            cycle_shape=self.cycle.shape,
            tof_indptr=self.tof_indptr,
        )
        self.transmission_calibrator = alphasynchro.algorithms.calibration.TransmissionCalibrator.from_unfragmented_pairs(
            self.indexed_fragments,
            indexed_precursors,
            self.monoisotopic_precursors,
            self.fragments,
            self.cycle,
            smooth_factor=smooth_factor,
            max_mz=max_mz,
            decimals=decimals,
            most_intense_count=most_intense_count,
        )
        logging.info("Finished calibrating quadrupole")

    def create_ms2_spectra(
        self,
        min_peaks: int = 0,
        max_rt_weight: float = .5,
        max_im_weight: float = .5,
        max_frame_weight: float = .5,
        unique_transitions_only: bool = False,
        diapasef: bool = False,
    ) -> None:
        logging.info("Calculating transitions and creating MS2 spectra...")
        cycle_center = (np.sum(self.cycle, axis=-1) / 2)[0]
        slicer = alphasynchro.algorithms.precursor_slicing.SlicedIMDistributionMultithreaded(
            precursors=self.monoisotopic_precursors,
            calibration=self.transmission_calibrator,
            cycle_center=cycle_center,
            cycle=self.cycle,
            diapasef=diapasef,
        )
        rt_transitions_dict = {}
        im_transitions_dict = {}
        summed_precursor_intensities_dict = {}
        logging.info("Calculating frame transitions...")
        for frame_index in range(1, self.cycle.shape[1]):
            (
                summed_precursor_intensities_dict[frame_index],
                im_transitions_dict[frame_index],
                rt_transitions_dict[frame_index],
            ) = self.calculate_transitions_of_frame(
                frame_index=frame_index,
                unique_transitions_only=unique_transitions_only,
                max_im_weight=max_im_weight,
                max_rt_weight=max_rt_weight,
                min_peaks=min_peaks,
                slicer=slicer,
            )
        logging.info("Merging transitions...")
        merged_transition_index = alphasynchro.ms.transitions.merged_transitions.MergedFrames.from_transition_dicts(
            im_transition_dict=im_transitions_dict,
            rt_transition_dict=rt_transitions_dict,
            fragments=self.fragments,
        )
        del im_transitions_dict
        del rt_transitions_dict
        merged_transition_index = merged_transition_index.sort_mz()
        merged_peak_transitions = merged_transition_index.count_all_merged_peaks()
        fragment_pointers = merged_transition_index.index_peaks(merged_peak_transitions)
        logging.info("Merging sliced fragments...")
        merged_fragments = alphasynchro.ms.peaks.merged_fragments.MergedFragments.from_pointers(
            fragment_pointers,
            self.fragments,
            merged_transition_index,
            self.cycle.shape[1] - 1,
        )
        self.merged_fragments = merged_fragments
        logging.info("Calculating sliced intensity profiles...")
        size = self.cycle.shape[1] - 1
        indptr = np.arange(len(self.monoisotopic_precursors) + 1) * size
        values = np.empty(len(self.monoisotopic_precursors) * size)
        for frame in range(size):
            values[frame::size] = summed_precursor_intensities_dict[frame + 1]
        slice_profile = alphasynchro.stats.distributions.PDF(
            indptr=indptr,
            values=values,
        ).to_cdf()
        del summed_precursor_intensities_dict
        logging.info("Calculating ks-stats for intensity profiles...")
        ks_tester = alphasynchro.stats.ks_1d.KSTester1DNoOffsetPairedMultithreaded(
            cdf_with_offset=slice_profile,
            secondary_cdf_with_offset=self.merged_fragments.frame_intensities,
        )
        precursor_indices = np.repeat(
            np.arange(slice_profile.shape[0]),
            np.diff(merged_peak_transitions),
        )
        fragment_indices = np.arange(merged_peak_transitions[-1])
        paired_indices = np.vstack(
            [
                precursor_indices,
                fragment_indices
            ]
        ).T
        frame_weights = ks_tester.calculate_all(paired_indices)
        logging.info("Filtering final transitions...")
        transitions = alphasynchro.ms.transitions.frame_transitions.Transitions(
            indptr=merged_peak_transitions,
            values=np.arange(merged_peak_transitions[-1]),
            weights=frame_weights,
            precursor_indices=np.arange(len(merged_peak_transitions) - 1),
        )
        valid_fragments = frame_weights <= max_frame_weight
        transitions = transitions.filter_weights(valid_fragments)
        valid_precursors = np.flatnonzero(np.diff(transitions.indptr) >= 5)
        self.transitions = transitions.filter(valid_precursors)
        logging.info("Finished calculating transitions and creating MS2 spectra")

    def calculate_transitions_of_frame(
        self,
        frame_index: int,
        unique_transitions_only: bool,
        max_im_weight: float,
        max_rt_weight: float,
        min_peaks: int,
        slicer,
    ):
        logging.info(f"Calculating transitions for frame {frame_index}...")
        logging.info("Calculating transmission efficiency...")
        transmitted_precursor_im_profiles = slicer.calculate_all_transmitted_cdf_for_frame(frame_index)
        summed_precursor_intensities = transmitted_precursor_im_profiles.summed_values
        logging.info("Calculating im apices...")
        apex_finder = alphasynchro.stats.apex_finder.SmoothApexFinder(
            cdf=transmitted_precursor_im_profiles
        )
        im_apices = apex_finder.calculate_all()
        logging.info("Indexing precusors...")
        indexed_precursors_for_frame = alphasynchro.ms.peaks.indexed.im_peaks.PushIndexedImPeaks.from_data_space(
            peaks=self.monoisotopic_precursors,
            im_apices=im_apices,
            cycle_shape=self.cycle.shape,
            tof_indptr=self.tof_indptr,
        )
        logging.info("Matching precursors with fragments...")
        matcher = alphasynchro.algorithms.matching.matching.FragmentedMatcherMultithreaded(
            indexed_precursors=indexed_precursors_for_frame,
            indexed_fragments=self.indexed_fragments,
            frame=frame_index,
        )
        precursor_fragment_pairs = matcher.match_all()
        order = np.argsort(precursor_fragment_pairs[:, 0 ])
        precursor_fragment_pairs = precursor_fragment_pairs[order]
        logging.info("Calculating ks-stats for IM...")
        paired_ks_tester_im = alphasynchro.stats.ks_1d.KSTester1DPairedMultithreaded(
            cdf_with_offset=transmitted_precursor_im_profiles,
            secondary_cdf_with_offset=self.fragments.im_projection,
        )
        im_weights = paired_ks_tester_im.calculate_all(precursor_fragment_pairs)
        logging.info("Calculating ks-stats for RT...")
        paired_ks_tester_rt = alphasynchro.stats.ks_1d.KSTester1DPairedMultithreaded(
            cdf_with_offset=self.monoisotopic_precursors.rt_projection,
            secondary_cdf_with_offset=self.fragments.rt_projection,
        )
        rt_weights = paired_ks_tester_rt.calculate_all(precursor_fragment_pairs)
        logging.info("Filtering transitions...")
        indptr = np.zeros(len(self.monoisotopic_precursors) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(
            np.bincount(
                precursor_fragment_pairs[:, 0],
                minlength=len(self.monoisotopic_precursors)
            )
        )
        transition_dummy = alphasynchro.data.sparse_indices.SparseIndex(
            indptr=indptr,
            values=np.arange(precursor_fragment_pairs.shape[0]),
        )
        if unique_transitions_only:
            valid_indices = np.zeros(len(im_weights), dtype=np.bool_)
            # unique_indices = alphasynchro.ms.transitions.frame_transitions.get_best_uniqueness_mask(
            #     precursor_fragment_pairs[:, 1], im_weights+rt_weights
            # )
            unique_indices = alphasynchro.ms.transitions.frame_transitions.get_best_uniqueness_mask(
                precursor_fragment_pairs[:, 1], im_weights
            )
            valid_indices[unique_indices] = True
            unique_indices = alphasynchro.ms.transitions.frame_transitions.get_best_uniqueness_mask(
                precursor_fragment_pairs[:, 1], rt_weights
            )
            valid_indices[unique_indices] = True
        else:
            valid_indices = (im_weights <= max_im_weight) & (rt_weights <= max_rt_weight)
        transition_dummy = transition_dummy.filter_values(valid_indices)
        valid_precursors = np.flatnonzero(np.diff(transition_dummy.indptr) >= min_peaks)
        transition_dummy = transition_dummy.filter(valid_precursors)
        valid_fragments = precursor_fragment_pairs[transition_dummy.values, 1]
        im_transitions = alphasynchro.ms.transitions.frame_transitions.Transitions(
            indptr=transition_dummy.indptr,
            values=valid_fragments,
            weights=im_weights[transition_dummy.values],
            precursor_indices=valid_precursors,
        )
        rt_transitions = alphasynchro.ms.transitions.frame_transitions.Transitions(
            indptr=transition_dummy.indptr,
            values=valid_fragments,
            weights=im_weights[transition_dummy.values],
            precursor_indices=valid_precursors,
        )
        return (
            summed_precursor_intensities,
            im_transitions,
            rt_transitions,
        )

    def write_ms2_spectra(
        self,
        output_file_name,
    ) -> None:
        logging.info("Writing MS2 spectra to file...")
        if not hasattr(self, "monoisotopic_precursors"):
            object.__setattr__(
                self,
                "monoisotopic_precursors",
                alphasynchro.ms.peaks.precursors.Precursors.from_analysis_hdf(
                    self.analysis_file
                )
            )
        if not hasattr(self, "merged_fragments"):
            object.__setattr__(
                self,
                "merged_fragments",
                alphasynchro.ms.peaks.merged_fragments.MergedFragments.from_analysis_hdf(
                    self.analysis_file
                )
            )
        if not hasattr(self, "transitions"):
            object.__setattr__(
                self,
                "transitions",
                alphasynchro.ms.transitions.frame_transitions.Transitions.from_analysis_hdf(
                    self.analysis_file
                )
            )
        alphasynchro.io.writing.mgf.Writer(
            file_name=output_file_name,
            precursors=self.monoisotopic_precursors,
            fragments=self.merged_fragments,
            transitions=self.transitions,
        ).write_to_file()
        logging.info("Finished witing MS2 spectra to file...")

    def __setattr__(self, __name: str, __value: Any) -> None:
        logging.info(f"Storing {__name}...")
        value = self.analysis_file.recursive_store(__name, __value)
        object.__setattr__(self, __name, value)
