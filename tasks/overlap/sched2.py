from faasmctl.util.flush import flush_workers
from faasmctl.util.config import get_faasm_worker_ips
from faasmctl.util.planner import reset as reset_planner
from invoke import task
from time import sleep
import threading

POLYBENCH_USER = "polybench"
POLYBENCH_FUNCS_A = [
    "poly_deriche",
]
POLYBENCH_FUNCS_B = [
    "poly_doitgen"
]

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
def _get_poly_benchmarks(bench, repeats):
    tasks_a = []
    tasks_b = []
    if bench:
        if bench in POLYBENCH_FUNCS_A:
            tasks_a = [(bench, run_index) for run_index in range(repeats)]
        elif bench in POLYBENCH_FUNCS_B:
            tasks_b = [(bench, run_index) for run_index in range(repeats)]
        else:
            raise RuntimeError(
                f"Unrecognised benchmark: {bench}. Must be one in: {POLYBENCH_FUNCS_A + POLYBENCH_FUNCS_B}"
            )
    else:
        for poly_bench in POLYBENCH_FUNCS_A:
            for run_index in range(repeats):
                tasks_a.append((poly_bench, run_index))
        for poly_bench in POLYBENCH_FUNCS_B:
            for run_index in range(repeats):
                tasks_b.append((poly_bench, run_index))
    return tasks_a, tasks_b

@task(default=True)
def polyfunc2(ctx, num_cpus_per_vm, bench=None, repeats=3, poll_interval=50):
    """
    Run the PolyBench/C microbenchmark tasks concurrently on Granny.
    
    Instead of serially executing all tasks, this function creates a pool
    of tasks (each corresponding to one run of a PolyBench function) and uses
    planner monitoring (via get_num_idle_cpus_from_in_flight_apps) to schedule
    a new task only when an idle core is available. The tasks are dispatched
    alternately from two groups (A and B) to maintain a balance, and each task
    is executed in its own thread.
    """
    # Discover workers and reset the planner
    worker_ips = get_faasm_worker_ips()
    num_vms = len(worker_ips)
    reset_planner(num_vms)
    # Clear the host state
    flush_workers()

    # Build the task pool: one entry per (benchmark, repeat) for each group A and B
    tasks_a, tasks_b = _get_poly_benchmarks(bench, repeats)
    
    active_threads = []
    toggle = True  # Toggle to alternate between A and B

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
    while (tasks_a or tasks_b) or any(t.is_alive() for t in active_threads):
        # Clean up finished threads from our active list
        active_threads = [t for t in active_threads if t.is_alive()]

        # Get current resource metrics from the planner
        in_flight_apps = planner_get_in_fligh_apps()
        idle_info = get_num_idle_cpus_from_in_flight_apps(num_vms, num_cpus_per_vm, in_flight_apps)
        # idle_info is (num_idle_vms, num_idle_cpus); we use the total idle cores.
        idle_cores = idle_info[1] if isinstance(idle_info, tuple) else idle_info

        # While there are free cores and pending tasks, schedule new tasks.
        while idle_cores > 0 and (tasks_a or tasks_b):
            if toggle:
                if tasks_a:
                    poly_bench, run_index = tasks_a.pop(0)
                    t = threading.Thread(target=run_task, args=(poly_bench, run_index))
                    t.start()
                    active_threads.append(t)
                    idle_cores -= 1
                elif tasks_b:
                    poly_bench, run_index = tasks_b.pop(0)
                    t = threading.Thread(target=run_task, args=(poly_bench, run_index))
                    t.start()
                    active_threads.append(t)
                    idle_cores -= 1
            else:
                if tasks_b:
                    poly_bench, run_index = tasks_b.pop(0)
                    t = threading.Thread(target=run_task, args=(poly_bench, run_index))
                    t.start()
                    active_threads.append(t)
                    idle_cores -= 1
                elif tasks_a:
                    poly_bench, run_index = tasks_a.pop(0)
                    t = threading.Thread(target=run_task, args=(poly_bench, run_index))
                    t.start()
                    active_threads.append(t)
                    idle_cores -= 1
            toggle = not toggle

        # Wait a short while before polling again.
        sleep(poll_interval / 1000.0)

    # Ensure all threads have finished before exiting the task.
    for t in active_threads:
        t.join()
