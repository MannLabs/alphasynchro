#local
import alphasynchro.io.hdf

#external
import numpy as np


def create_test_cluster_hdf_object():
    dummy_hdf_cluster = alphasynchro.io.hdf.HDFObject.from_file(
        "./unit_tests/test_clusters.hdf",
        new=True
    )

    dummy_hdf_cluster.set_attr("sample_name", "test_sample")

    acquisition_hdf_object = dummy_hdf_cluster.set_group("acquisition")
    acquisition_hdf_object.set_mmap(
        "cycle",
        np.arange(24).reshape((1,2,3,4))
    )
    acquisition_hdf_object.set_mmap(
        "tof_indptr",
        np.arange(24)
    )

    clusters_hdf_object = dummy_hdf_cluster.set_group("clustering")

    im_projection_hdf_object = clusters_hdf_object.set_group("im_projection")
    im_projection_hdf_object.set_mmap(
        "indptr",
        np.array([0, 2, 5, 7], dtype=np.int64)
    )
    im_projection_hdf_object.set_mmap(
        "start_index",
        np.array([1, 0, 0], dtype=np.int64)
    )
    im_projection_hdf_object.set_mmap(
        "summed_intensity_values",
        np.array([.5, 1,.3, .7, 1, .4, 1])
    )

    rt_projection_hdf_object = clusters_hdf_object.set_group("rt_projection")
    rt_projection_hdf_object.set_mmap(
        "indptr",
        np.array([0, 3, 4, 6], dtype=np.int64)
    )
    rt_projection_hdf_object.set_mmap(
        "start_index",
        np.array([2, 0, 2], dtype=np.int64)
    )
    rt_projection_hdf_object.set_mmap(
        "summed_intensity_values",
        np.array([.1, .2, 1, 1, .9, 1])
    )

    raw_pointers_hdf_object = clusters_hdf_object.set_group("raw_pointers")
    raw_pointers_hdf_object.set_mmap(
        "indptr",
        np.array([0, 1, 4, 5], dtype=np.int64)
    )
    raw_pointers_hdf_object.set_mmap(
        "indices",
        np.array([1, 0, 2, 3, 4], dtype=np.int64)
    )

    as_dataframe_hdf_object = clusters_hdf_object.set_group("as_dataframe")
    as_dataframe_hdf_object.set_mmap(
        "apex_pointer",
        np.array([1, 2, 3], dtype=np.int64)
    )
    as_dataframe_hdf_object.set_mmap(
        "frame_group",
        np.array([0, 1, 3], dtype=np.int64)
    )
    as_dataframe_hdf_object.set_mmap(
        "im_weighted_average",
        np.array([.7, .701, .8])
    )
    as_dataframe_hdf_object.set_mmap(
        "rt_weighted_average",
        np.array([10.1, 9.9, 100])
    )
    as_dataframe_hdf_object.set_mmap(
        "mz_weighted_average",
        np.array([500.1, 500.1, 700.1])
    )
    as_dataframe_hdf_object.set_mmap(
        "number_of_ions",
        np.array([1, 3, 1], dtype=np.int64)
    )
    as_dataframe_hdf_object.set_mmap(
        "summed_intensity",
        np.array([200, 850, 150])
    )

    ms1_hdf_object = dummy_hdf_cluster.set_group("ms1")
    precursors_hdf_object = ms1_hdf_object.set_group("precursors")
    precursors_hdf_object.set_mmap(
        "cluster_pointers",
        np.array([0], dtype=np.int64)
    )
    monoisotopic_precursors_hdf_object = ms1_hdf_object.set_group("monoisotopic_precursors")
    as_dataframe_hdf_object = monoisotopic_precursors_hdf_object.set_group("as_dataframe")
    as_dataframe_hdf_object.set_mmap(
        "charge",
        np.array([2], dtype=np.int64)
    )
    as_dataframe_hdf_object.set_mmap(
        "precursor_pointers",
        np.array([0], dtype=np.int64)
    )

    ms2_hdf_object = dummy_hdf_cluster.set_group("ms2")
    fragments_hdf_object = ms2_hdf_object.set_group("fragments")
    fragments_hdf_object.set_mmap(
        "cluster_pointers",
        np.array([1, 2], dtype=np.int64)
    )
