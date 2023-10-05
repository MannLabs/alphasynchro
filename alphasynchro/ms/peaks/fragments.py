#!python
'''Module to represent fragments as a njit dataclass.'''


# local
import alphasynchro.performance.compiling
import alphasynchro.ms.peaks.peaks
import alphasynchro.io.hdf


@alphasynchro.performance.compiling.njit_dataclass
class Fragments(alphasynchro.ms.peaks.peaks.Peaks):

    @classmethod
    def from_clusters_hdf(
        cls,
        file_name: str,
        min_fragment_size: int = 0
    ):
        hdf_cluster_object = alphasynchro.io.hdf.HDFObject.from_file(
            file_name
        )
        fragment_indices = hdf_cluster_object.ms2.fragments.cluster_pointers
        if min_fragment_size != 0:
            fragment_indices = fragment_indices[
                hdf_cluster_object.clustering.as_dataframe.number_of_ions[
                    fragment_indices
                ] >= min_fragment_size
            ]
        return super().from_clusters_hdf(file_name, fragment_indices)

    @classmethod
    def from_analysis_hdf(cls, analys_hdf_file):
        return super().from_analysis_hdf_subgroup(analys_hdf_file.fragments)
