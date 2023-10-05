#!python
'''Module to represent (monoisotopic) precursors as a njit dataclass.'''


# local
import alphasynchro.performance.compiling
import alphasynchro.ms.peaks.peaks
import alphasynchro.io.hdf

# external
import numpy as np


@alphasynchro.performance.compiling.njit_dataclass
class Precursors(alphasynchro.ms.peaks.peaks.Peaks):

    @classmethod
    def from_clusters_hdf(cls, file_name: str):
        hdf_cluster_object = alphasynchro.io.hdf.HDFObject.from_file(
            file_name
        )
        precursor_indices = hdf_cluster_object.ms1.precursors.cluster_pointers[
            hdf_cluster_object.ms1.monoisotopic_precursors.as_dataframe.precursor_pointers
        ]
        return super().from_clusters_hdf(file_name, precursor_indices)

    @classmethod
    def load_aggregate_data(
        cls,
        hdf_cluster_object: alphasynchro.io.hdf.HDFObject,
        indices: np.ndarray
    ) -> alphasynchro.data.dataframe.DataFrame:
        array_dict = {
            array_name: hdf_cluster_object.clustering.as_dataframe.__getattribute__(
                array_name
            )[indices] for array_name in hdf_cluster_object.clustering.as_dataframe.arrays if array_name != "frame_group"
        }
        array_dict["charge"] = hdf_cluster_object.ms1.monoisotopic_precursors.as_dataframe.charge
        return alphasynchro.data.dataframe.DataFrame(**array_dict)

    @classmethod
    def from_analysis_hdf(cls, analys_hdf_file):
        return super().from_analysis_hdf_subgroup(analys_hdf_file.monoisotopic_precursors)
