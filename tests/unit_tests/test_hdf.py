# builtin
import os

#external
import numpy as np
import pytest

#local
import alphasynchro.io.hdf


TEST_FILE_NAME = "sandbox_folder/sandbox.hdf"
GROUP_NAME = "/test/deeper/l2/"

if os.path.exists(TEST_FILE_NAME):
    os.remove(TEST_FILE_NAME)


def test_read_and_write_hdf_attr():
    expected = 1
    output = alphasynchro.io.hdf.write_attr(
        file_name=TEST_FILE_NAME,
        group_name=GROUP_NAME,
        attr_name="attr",
        attr_value=expected,
    )
    assert output == expected


def test_read_and_write_hdf_mmap():
    expected = np.arange(10)
    output = alphasynchro.io.hdf.write_mmap(
        file_name=TEST_FILE_NAME,
        group_name=GROUP_NAME,
        mmap_name="mmap",
        mmap_value=expected,
    )
    assert np.array_equal(output, expected)


def test_overwrite_hdf_mmap():
    expected = np.arange(10)
    _ = alphasynchro.io.hdf.write_mmap(
        file_name=TEST_FILE_NAME,
        group_name=GROUP_NAME,
        mmap_name="mmap",
        mmap_value=expected,
    )
    output = alphasynchro.io.hdf.write_mmap(
        file_name=TEST_FILE_NAME,
        group_name=GROUP_NAME,
        mmap_name="mmap",
        mmap_value=expected,
    )
    assert np.array_equal(output, expected)


def test_hdf_object_reading():
    expected = np.arange(10)
    alphasynchro.io.hdf.write_mmap(
        file_name=TEST_FILE_NAME,
        group_name=GROUP_NAME,
        mmap_name="mmap",
        mmap_value=expected,
    )
    output = alphasynchro.io.hdf.HDFObject.from_file(TEST_FILE_NAME)
    assert np.array_equal(output.test.deeper.l2.mmap, expected)


def test_hdf_object_mmap_writing():
    output = alphasynchro.io.hdf.HDFObject.from_file(
        TEST_FILE_NAME,
        new=True
    )
    sub_folder = output.set_group("sub_folder")
    expected = np.arange(10)
    assert "arr" not in sub_folder.arrays
    sub_folder.set_mmap("arr", expected)
    assert np.array_equal(output.sub_folder.arr, expected)
    assert "arr" in sub_folder.arrays


def test_hdf_object_attr_writing():
    output = alphasynchro.io.hdf.HDFObject.from_file(
        TEST_FILE_NAME,
        new=True
    )
    sub_folder = output.set_group("sub_folder")
    expected = "test_result"
    assert "test" not in sub_folder.attrs
    sub_folder.set_attr("test", expected)
    assert np.array_equal(output.sub_folder.test, expected)
    assert "test" in sub_folder.attrs


@pytest.mark.parametrize(
    "input",
    [
        "./temp_hdf.hdf",
        None,
    ]
)
def test_temporary(input):
    with alphasynchro.io.hdf.temporary(input) as temp_hdf:
        sub_folder = temp_hdf.set_group("sub_folder")
        expected = "test_result"
        assert "test" not in sub_folder.attrs
        sub_folder.set_attr("test", expected)
        assert np.array_equal(temp_hdf.sub_folder.test, expected)
        assert "test" in sub_folder.attrs


def test_recursive_store():
    import alphasynchro.ms.peaks.peaks
    test_peaks = alphasynchro.ms.peaks.peaks.Peaks.from_clusters_hdf(
        "./unit_tests/test_clusters.hdf",
        indices=np.array([0], dtype=np.int64)
    )
    with alphasynchro.io.hdf.temporary() as temp_hdf:
        result = temp_hdf.recursive_store("peaks", test_peaks)
        assert result == test_peaks
