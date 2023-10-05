#external
import numpy as np
import pytest
import os
import filecmp

#local
import alphasynchro.ms.peaks.precursors
import alphasynchro.ms.peaks.fragments
import alphasynchro.ms.transitions.frame_transitions
import alphasynchro.io.writing.hdf
import alphasynchro.io.writing.mgf
import alphasynchro.io.writing.prolucid


@pytest.fixture(scope="module")
def precursors():
    test_peaks = alphasynchro.ms.peaks.precursors.Precursors.from_clusters_hdf(
        "./unit_tests/test_clusters.hdf",
    )
    return test_peaks


@pytest.fixture(scope="module")
def fragments():
    test_peaks = alphasynchro.ms.peaks.fragments.Fragments.from_clusters_hdf(
        "./unit_tests/test_clusters.hdf",
    )
    return test_peaks


@pytest.fixture(scope="module")
def transitions():
    test_transitions = alphasynchro.ms.transitions.frame_transitions.Transitions(
        indptr=np.array([0, 2]),
        values=np.array([0, 1]),
        weights=np.array([.5, .7]),
        precursor_indices=np.arange(1),
    )
    return test_transitions


def test_prolucid(precursors, fragments, transitions):
    file_name = "./unit_tests/sandbox_output.prolucid"
    writer = alphasynchro.io.writing.prolucid.Writer(
        file_name=file_name,
        precursors=precursors,
        fragments=fragments,
        transitions=transitions,
    )
    writer.write_to_file()
    expected = read_file_lines("./unit_tests/test_output.prolucid", skip_lines=1)
    output = read_file_lines(file_name, skip_lines=1)
    assert str(output) == str(expected)
    os.remove(file_name)


def test_mgf(precursors, fragments, transitions):
    file_name = "./unit_tests/sandbox_output.mgf"
    writer = alphasynchro.io.writing.mgf.Writer(
        file_name=file_name,
        precursors=precursors,
        fragments=fragments,
        transitions=transitions,
    )
    writer.write_to_file()
    assert filecmp.cmp(file_name, "./unit_tests/test_output.mgf")
    os.remove(file_name)


def test_hdf(precursors, fragments, transitions):
    file_name = "./unit_tests/sandbox_output.hdf"
    writer = alphasynchro.io.writing.hdf.Writer(
        file_name=file_name,
        precursors=precursors,
        fragments=fragments,
        transitions=transitions,
    )
    writer.write_to_file()
    assert filecmp.cmp(file_name, "./unit_tests/test_output.hdf")
    os.remove(file_name)


def read_file_lines(file_name, skip_lines=0):
    with open(file_name) as raw_file:
        lines = raw_file.readlines()
    return lines[skip_lines:]
