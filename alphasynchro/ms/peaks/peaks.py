#!python
'''Module to represent MS1/MS2 peaks as a njit dataclass.'''


# local
import alphasynchro.data.sparse_indices
import alphasynchro.stats.distributions
import alphasynchro.data.dataframe
import alphasynchro.performance.compiling
import alphasynchro.io.hdf

# external
import numpy as np


@alphasynchro.performance.compiling.njit_dataclass
class Peaks:

    raw_pointers: alphasynchro.data.sparse_indices.SparseIndex
    rt_projection: alphasynchro.stats.distributions.CDFWithOffset
    im_projection: alphasynchro.stats.distributions.CDFWithOffset
    aggregate_data: alphasynchro.data.dataframe.DataFrame

    @classmethod
    def from_clusters_hdf(
        cls,
        file_name: str,
        indices: np.ndarray = ...
    ):
        hdf_cluster_object = alphasynchro.io.hdf.HDFObject.from_file(file_name)
        raw_pointers = cls.load_raw_pointers(hdf_cluster_object, indices)
        rt_projection = cls.load_rt_projection(hdf_cluster_object, indices)
        im_projection = cls.load_im_projection(hdf_cluster_object, indices)
        aggregate_data = cls.load_aggregate_data(hdf_cluster_object, indices)
        return cls(
            rt_projection=rt_projection,
            im_projection=im_projection,
            raw_pointers=raw_pointers,
            aggregate_data=aggregate_data,
        )

    @classmethod
    def load_raw_pointers(
        cls,
        hdf_cluster_object: alphasynchro.io.hdf.HDFObject,
        indices: np.ndarray
    ) -> alphasynchro.data.sparse_indices.SparseIndex:
        return alphasynchro.data.sparse_indices.SparseIndex(
            indptr=hdf_cluster_object.clustering.raw_pointers.indptr,
            values=hdf_cluster_object.clustering.raw_pointers.indices,
        ).filter(indices)

    @classmethod
    def load_rt_projection(
        cls,
        hdf_cluster_object: alphasynchro.io.hdf.HDFObject,
        indices: np.ndarray
    ) -> alphasynchro.stats.distributions.CDFWithOffset:
        return alphasynchro.stats.distributions.CDFWithOffset(
            indptr=hdf_cluster_object.clustering.rt_projection.indptr,
            values=hdf_cluster_object.clustering.rt_projection.summed_intensity_values,
            start_offsets=hdf_cluster_object.clustering.rt_projection.start_index
        ).filter(indices)

    @classmethod
    def load_im_projection(
        cls,
        hdf_cluster_object: alphasynchro.io.hdf.HDFObject,
        indices: np.ndarray
    ) -> alphasynchro.stats.distributions.CDFWithOffset:
        return alphasynchro.stats.distributions.CDFWithOffset(
            indptr=hdf_cluster_object.clustering.im_projection.indptr,
            values=hdf_cluster_object.clustering.im_projection.summed_intensity_values,
            start_offsets=hdf_cluster_object.clustering.im_projection.start_index,
        ).filter(indices)

    @classmethod
    def load_aggregate_data(
        cls,
        hdf_cluster_object: alphasynchro.io.hdf.HDFObject,
        indices: np.ndarray
    ) -> alphasynchro.data.dataframe.DataFrame:
        return alphasynchro.data.dataframe.DataFrame(
            **{
                array: hdf_cluster_object.clustering.as_dataframe.__getattribute__(
                    array
                )[indices] for array in hdf_cluster_object.clustering.as_dataframe.arrays
            }
        )

    def __len__(self):
        return len(self.aggregate_data)

    @classmethod
    def from_analysis_hdf_subgroup(cls, hdf_subgroup):
        peaks = cls(
            raw_pointers=alphasynchro.data.sparse_indices.SparseIndex(
                indptr=hdf_subgroup.raw_pointers.indptr,
                values=hdf_subgroup.raw_pointers.values,
            ),
            rt_projection=alphasynchro.stats.distributions.CDFWithOffset(
                indptr=hdf_subgroup.rt_projection.indptr,
                values=hdf_subgroup.rt_projection.values,
                start_offsets=hdf_subgroup.rt_projection.start_offsets,
            ),
            im_projection=alphasynchro.stats.distributions.CDFWithOffset(
                indptr=hdf_subgroup.im_projection.indptr,
                values=hdf_subgroup.im_projection.values,
                start_offsets=hdf_subgroup.im_projection.start_offsets,
            ),
            aggregate_data=alphasynchro.data.dataframe.DataFrame(
                **{
                    array: hdf_subgroup.aggregate_data.__getattribute__(
                        array
                    ) for array in hdf_subgroup.aggregate_data.arrays
                }
            ),
        )
        return peaks
