from faasmctl.util.flush import flush_workers
from faasmctl.util.config import get_faasm_worker_ips
from faasmctl.util.planner import reset as reset_planner
from invoke import task
from time import sleep
import threading

from tasks.polybench.util import POLYBENCH_FUNCS, POLYBENCH_USER
from tasks.util.faasm import (
    get_faasm_exec_time_from_json,
    post_async_msg_and_get_result_json,
)
from faasmctl.util.planner import (
    get_in_fligh_apps as planner_get_in_fligh_apps,
    set_next_evicted_host as planner_set_next_evicted_host,
    wait_for_workers as planner_wait_for_workers,
)
from tasks.util.planner import (
    get_num_available_slots_from_in_flight_apps,
    get_num_idle_cpus_from_in_flight_apps,
    get_num_xvm_links_from_in_flight_apps,
)

# A helper to select the polybench benchmarks to run
def _get_poly_benchmarks(bench):
    if bench:
        if bench not in POLYBENCH_FUNCS:
            raise RuntimeError(
                f"Unrecognised benchmark: {bench}. Must be one in: {POLYBENCH_FUNCS}"
            )
        poly_benchmarks = [bench]
    else:
        poly_benchmarks = POLYBENCH_FUNCS
    return poly_benchmarks

@task(default=True)
def polyfunc(ctx, bench=None, repeats=3, poll_interval=1):
    """
    Run the PolyBench/C microbenchmark tasks concurrently on Granny.
    
    Instead of serially executing all tasks, this function creates a pool
    of tasks (each corresponding to one run of a PolyBench function) and uses
    planner monitoring (via get_num_idle_cpus_from_in_flight_apps) to schedule
    a new task only when an idle core is available. The tasks are dispatched
    round robin, and each task is executed in its own thread.
    """
    # Discover workers and reset the planner
    worker_ips = get_faasm_worker_ips()
    num_vms = len(worker_ips)
    reset_planner(num_vms)
    # Clear the host state
    flush_workers()

    # Build the task pool: one entry per (benchmark, repeat)
    poly_benchmarks = _get_poly_benchmarks(bench)
    tasks_to_run = []
    for poly_bench in poly_benchmarks:
        for run_index in range(repeats):
            tasks_to_run.append((poly_bench, run_index))
    
    active_threads = []

    # Function to send a single PolyBench task
    def run_task(poly_bench, run_index):
        print(f"Starting function {poly_bench}, run {run_index}")
        msg = {
            "user": POLYBENCH_USER,
            "function": poly_bench,
        }
        # This call is blocking and returns when the task completes.
        result_json = post_async_msg_and_get_result_json(msg)
        actual_time = get_faasm_exec_time_from_json(result_json)
        for res in result_json:
            print(f"Function: {poly_bench}, run: {run_index}, start: {res['start_ts']}, end: {res['finish_ts']}")
        print(f"Function: {poly_bench}, run: {run_index}, Actual time: {actual_time} seconds")

    # Main scheduling loop: keep scheduling until all tasks are dispatched
    while tasks_to_run or any(t.is_alive() for t in active_threads):
        # Clean up finished threads from our active list
        active_threads = [t for t in active_threads if t.is_alive()]

        # Get current resource metrics from the planner
        in_flight_apps = planner_get_in_fligh_apps()
        idle_info = get_num_idle_cpus_from_in_flight_apps(num_vms, 8, in_flight_apps)
        # idle_info is (num_idle_vms, num_idle_cpus); we use the total idle cores.
        idle_cores = idle_info[1] if isinstance(idle_info, tuple) else idle_info

        # While there are free cores and pending tasks, schedule new tasks.
        while idle_cores > 0 and tasks_to_run:
            poly_bench, run_index = tasks_to_run.pop(0)
            t = threading.Thread(target=run_task, args=(poly_bench, run_index))
            t.start()
            active_threads.append(t)
            idle_cores -= 1

        # Wait a short while before polling again.
        sleep(poll_interval)

    # Ensure all threads have finished before exiting the task.
    for t in active_threads:
        t.join()