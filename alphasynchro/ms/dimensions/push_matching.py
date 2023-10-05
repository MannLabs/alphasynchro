#!python
'''Module to generate push indices that are within certain thresholds.'''


# local
import alphasynchro.performance.compiling


@alphasynchro.performance.compiling.njit(nogil=True)
def generate_neighbor_push_indices(
    push_index: int,
    shape: tuple[int, int, int], #  (cycles, frames, scans)
    scan_tolerance: int,
    cycle_tolerance: int,
):
    max_push_index = shape[0] * shape[1] * shape[2]
    if not (0 <= push_index < max_push_index):
        return
    pushes_per_cycle = shape[1] * shape[2]
    scan_index = push_index % shape[2]
    frame_index = (push_index // shape[2]) % shape[1]
    cycle_index = push_index // pushes_per_cycle
    for new_cycle_index in range(
        cycle_index - cycle_tolerance,
        cycle_index + cycle_tolerance + 1
    ):
        if not (0 <= new_cycle_index < shape[0]):
            continue
        for new_scan_index in range(
            scan_index - scan_tolerance,
            scan_index + scan_tolerance + 1
        ):
            if not (0 <= new_scan_index < shape[2]):
                continue
            new_push_index = new_cycle_index * pushes_per_cycle + frame_index * shape[2] + new_scan_index
            yield new_push_index
