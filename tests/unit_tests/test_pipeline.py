#external
import os
import numpy as np
import pytest

#local
import alphasynchro.algorithms.pipeline


TEST_FILE_NAME = "sandbox_folder/analysis.hdf"


if os.path.exists(TEST_FILE_NAME):
    os.remove(TEST_FILE_NAME)


def create_pipeline(overwrite=True):
    pipeline = alphasynchro.algorithms.pipeline.Pipeline(
        TEST_FILE_NAME,
        overwrite=overwrite
    )
    return pipeline


def test_analysis_file():
    pipeline = create_pipeline()
    assert len(pipeline.analysis_file.arrays) == 0
    pipeline.array = np.arange(3)
    assert len(pipeline.analysis_file.arrays) == 1
    file_object = alphasynchro.io.hdf.HDFObject.from_file(TEST_FILE_NAME)
    assert np.array_equal(pipeline.array, file_object.array)
    assert pipeline.analysis_file == file_object


def test_load_data_space():
    pipeline = create_pipeline()
    pipeline.load_data_space("./unit_tests/test_clusters.hdf")
    cycle = np.arange(24).reshape((1,2,3,4))
    assert np.array_equal(pipeline.cycle, cycle)
    tof_indptr = np.arange(24)
    assert np.array_equal(pipeline.tof_indptr, tof_indptr)


def test_load_peaks():
    pipeline = create_pipeline()
    pipeline.cycle = np.arange(24).reshape((1,2,3,4))
    pipeline.tof_indptr = np.arange(24)
    assert not hasattr(pipeline, "fragments")
    assert not hasattr(pipeline, "monoisotopic_precursors")
    pipeline.load_peaks("./unit_tests/test_clusters.hdf")
    assert hasattr(pipeline, "fragments")
    assert hasattr(pipeline, "monoisotopic_precursors")
