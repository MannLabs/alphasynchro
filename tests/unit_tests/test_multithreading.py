# builtin
import time
import multiprocessing

# external
import numpy as np
import numba
import pytest

# local
import alphasynchro.performance.multithreading

CPU_COUNT = multiprocessing.cpu_count()


@numba.njit(nogil=True)
def func(step_index, idxs, arr): # pragma: no cover
    result = 0
    for i in idxs[::step_index]:
        result += np.sin(arr[i]) * np.cos(arr[i])
    return result


@alphasynchro.performance.multithreading.parallel(
    include_progress_callback=False
)
@numba.njit(nogil=True)
def parallel_func(step_index, idxs, arr, output_buffer): # pragma: no cover
    result = func(step_index, idxs, arr)
    output_buffer[step_index] = result


def run_and_time_func(threads):
    max_size = 10**6 + 1000
    idxs = np.arange(max_size)
    arr = np.arange(max_size, dtype=np.float64) / max_size
    output_buffer = np.empty(max_size)
    alphasynchro.performance.multithreading.set_threads(threads)
    s1 = time.time()
    parallel_func(
        range(1, max_size),
        idxs,
        arr,
        output_buffer,
    )
    s2 = time.time()
    elapsed_time = s2 - s1
    return output_buffer, elapsed_time


def test_is_faster_and_correct():
    _ = run_and_time_func(0)
    output_buffer1, elapsed_time1 = run_and_time_func(1)
    output_buffer2, elapsed_time2 = run_and_time_func(2)
    assert elapsed_time2 < elapsed_time1
    assert np.array_equal(output_buffer1, output_buffer2)


# def run_and_time_func_without_range(threads):
#     max_size = 10**6
#     idxs = np.arange(max_size)
#     arr = np.arange(max_size, dtype=np.float64) / max_size
#     output_buffer = np.empty(max_size)
#     alphasynchro.performance.multithreading.set_threads(threads)
#     s1 = time.time()
#     parallel_func(
#         np.arange(1, max_size, 2),
#         idxs,
#         arr,
#         output_buffer,
#     )
#     s2 = time.time()
#     elapsed_time = s2 - s1
#     return output_buffer, elapsed_time


# def test_is_faster_and_correct_without_range():
#     _ = run_and_time_func_without_range(0)
#     output_buffer1, elapsed_time1 = run_and_time_func_without_range(1)
#     output_buffer2, elapsed_time2 = run_and_time_func_without_range(2)
#     assert elapsed_time2 < elapsed_time1
#     assert np.allclose(output_buffer1, output_buffer2, atol=1e-10)


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, CPU_COUNT - 1),
        (0, CPU_COUNT),
        (1, 1),
        (2, 2),
        (CPU_COUNT + 1, CPU_COUNT),
    ]
)
def test_set_thread(input, expected):
    output = alphasynchro.performance.multithreading.set_threads(input)
    global_threads_new = alphasynchro.performance.multithreading.MAX_THREADS
    assert output == expected
    assert global_threads_new == output


@pytest.mark.parametrize(
    "input, expected",
    [
        (-1, CPU_COUNT - 1),
        (0, CPU_COUNT),
        (1, 1),
        (2, 2),
        (CPU_COUNT + 1, CPU_COUNT),
    ]
)
def test_set_thread_non_global(input, expected):
    global_threads = alphasynchro.performance.multithreading.MAX_THREADS
    output = alphasynchro.performance.multithreading.set_threads(
        input,
        False
    )
    global_threads_new = alphasynchro.performance.multithreading.MAX_THREADS
    assert output == expected
    assert global_threads == global_threads_new
