# external
import logging
import os
import re

# local
import alphasynchro.io.logging


TEST_FILE_NAME = "sandbox_folder/log.txt"


if os.path.exists(TEST_FILE_NAME):
    os.remove(TEST_FILE_NAME)


def test_set_logger():
    assert not os.path.exists(TEST_FILE_NAME)
    alphasynchro.io.logging.set_logger(
        log_file_name=TEST_FILE_NAME
    )
    assert os.path.exists(TEST_FILE_NAME)
    lines = [
        "test1",
        "test2",
    ]
    for line in lines:
        logging.info(line)
    timestamp_regex = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    with open(TEST_FILE_NAME) as infile:
        for i, line in enumerate(infile):
            timestamp, info = line.split(">")
            assert re.match(timestamp_regex, timestamp)
            assert info.strip() == lines[i]
    assert i == len(lines) - 1



def test_show_platform_info():
    alphasynchro.io.logging.show_platform_info()


def test_show_python_info():
    alphasynchro.io.logging.show_python_info()
