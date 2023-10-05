#!python
'''Module to multithread functions of njitted dataclasses.'''


# builtin
import multiprocessing
import multiprocessing.pool
import functools
import threading

# external
import tqdm
import numba
import numpy as np


MAX_THREADS = multiprocessing.cpu_count() - 1
MAX_GRANULARITY = 10**6
TRIMMED_GRANULARITY = 10**3
PROGRESS_SPEED_LIMIT = 0.01


def set_threads(threads: int, set_global: bool = True) -> int:
    max_cpu_count = multiprocessing.cpu_count()
    if threads > max_cpu_count:
        threads = max_cpu_count
    else:
        while threads <= 0:
            threads += max_cpu_count
    if set_global:
        global MAX_THREADS
        MAX_THREADS = threads
    return threads


def parallel(
    _func: callable = None,
    *,
    thread_count: int = None,
    include_progress_callback: bool = True,
) -> None:
    def parallel_compiled_func_inner(func):
        numba_func = func

        @numba.njit(nogil=True)
        def numba_func_parallel(
            iterable,
            thread_id,
            progress_counter,
            start,
            stop,
            step,
            *args,
        ):
            if len(iterable) == 0:
                for i in range(start, stop, step):
                    numba_func(i, *args)
                    progress_counter[thread_id] += 1
            else:
                for i in iterable:
                    numba_func(i, *args)
                    progress_counter[thread_id] += 1

        def wrapper(iterable, *args):
            current_thread_count = _set_current_thread_count(thread_count)
            threads = []
            progress_counter = np.zeros(current_thread_count, dtype=np.int64)
            for thread_id in range(current_thread_count):
                thread = _launch_thread(
                    iterable,
                    thread_id,
                    current_thread_count,
                    numba_func_parallel,
                    progress_counter,
                    args,
                )
                threads.append(thread)
            if include_progress_callback:
                _track_progress(iterable, progress_counter)
            for thread in threads:
                thread.join()
                del thread
        return functools.wraps(func)(wrapper)
    if _func is None:
        return parallel_compiled_func_inner
    else:
        return parallel_compiled_func_inner(_func)


def _set_current_thread_count(thread_count: int) -> int:
    if thread_count is None:
        current_thread_count = MAX_THREADS
    else:
        current_thread_count = set_threads(
            thread_count,
            set_global=False
        )
    return current_thread_count


def _launch_thread(
    iterable,
    thread_id,
    current_thread_count,
    numba_func_parallel,
    progress_counter,
    args,
) -> threading.Thread:
    local_iterable = iterable[thread_id::current_thread_count]
    if isinstance(local_iterable, range):
        start = local_iterable.start
        stop = local_iterable.stop
        step = local_iterable.step
        local_iterable = np.array([], dtype=np.int64)
    else:
        start = -1
        stop = -1
        step = -1
    thread = threading.Thread(
        target=numba_func_parallel,
        args=(
            local_iterable,
            thread_id,
            progress_counter,
            start,
            stop,
            step,
            *args
        ),
        daemon=True
    )
    thread.start()
    return thread


def _track_progress(iterable, progress_counter) -> None:
    import time
    if len(iterable) > MAX_GRANULARITY:
        granularity = TRIMMED_GRANULARITY
    else:
        granularity = len(iterable)
    progress_bar = 0
    progress_count = np.sum(progress_counter)
    for _ in tqdm.tqdm(range(granularity)):
        while progress_bar >= progress_count:
            time.sleep(PROGRESS_SPEED_LIMIT)
            progress_count = granularity * np.sum(progress_counter) / len(iterable)
        progress_bar += 1
