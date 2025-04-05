from faasmctl.util.flush import flush_workers
from faasmctl.util.config import get_faasm_worker_ips
from faasmctl.util.planner import reset as reset_planner
from invoke import task
import time
import json
import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap
import os

from faasmctl.cpp_client import create_client

from tasks.overlap.hrperf_api import (hrperf_start, hrperf_pause)

POLYBENCH_USER = "polybench"
POLYBENCH_FUNC_NAMES = [
    "poly_2mm",
    "poly_3mm",
    "poly_adi",
    "poly_atax",
    "poly_bicg",
    "poly_cholesky",
    "poly_correlation",
    "poly_covariance",
    "poly_deriche",
    "poly_doitgen",
    "poly_durbin",
    "poly_fdtd-2d",
    "poly_floyd-warshall",
    "poly_gramschmidt",
    "poly_heat-3d",
    "poly_jacobi-1d",
    "poly_jacobi-2d",
    "poly_lu",
    "poly_ludcmp",
    "poly_mvt",
    "poly_nussinov",
    "poly_seidel-2d",
    "poly_trisolv",
]
NUM_THREADS = [1, 20, 25, 30, 35, 40]

def get_faasm_exec_time_from_metrics(metrics):
    '''
    The "metrics" is returned from a ber status reply,
    so it may involve more than one message, so we have more than one "metric" dict
    '''
    if not metrics:
        return 0
    
    # Find the first metric with valid duration
    for metric in metrics:
        if "durationMs" in metric and metric["durationMs"] is not None and metric["durationMs"] > 0:
            return metric["durationMs"] / 1000.0  # convert to seconds
    
    # Fallback: manually calculate from timestamps
    for metric in metrics:
        if "startTimestamp" in metric and "finishTimestamp" in metric:
            if metric["startTimestamp"] > 0 and metric["finishTimestamp"] > 0:
                return (metric["finishTimestamp"] - metric["startTimestamp"]) / 1000.0  # convert to seconds
    
    return 0

@task(default=True)
def prof_polybench(ctx, num_cpus, async_pool_interval=50):

    worker_ips = get_faasm_worker_ips()
    num_vms = len(worker_ips)
    num_cpus = int(num_cpus)

    assert num_vms == 1, "We only use a single worker for profiling."
    assert num_cpus >= max(NUM_THREADS), "Num CPUs must be at least the maximum nthreads."

    reset_planner(num_vms)
    flush_workers()
    print("Cluster ready.")

    # Yeah, our cpp client
    client = create_client()

    unique_funcs = set()
    unique_funcs.update(POLYBENCH_FUNC_NAMES)

    print("== Prewarming functions ==")

    prewarm_runtimes = {}

    for func_name in unique_funcs:
        print(f"Prewarming func: {func_name}")
        res = client.invoke_function(
            user=POLYBENCH_USER,
            function=func_name,
            input_data="",
            async_execution=False
        )

        if "metrics" in res and res["metrics"]:
            prewarm_runtimes[func_name] = get_faasm_exec_time_from_metrics(res["metrics"])
            print(f"Runtime: {prewarm_runtimes[func_name]}")
        else:
            print(f"Failed to prewarm {func_name}")
            return
    print("== Prewarming done ==")

    time.sleep(2)

    # Now we can start the actual profiling
    print("== Profiling ==")

    hrperf_start()

    for func_name in unique_funcs:
        print(f"Profiling func: {func_name}")

        for nthreads in NUM_THREADS:
            invoked_apps_ids = []
            expected_num_messages = 0
            for i in range(nthreads):
                res = client.invoke_function(
                    user=POLYBENCH_USER,
                    function=func_name,
                    input_data="",
                    async_execution=True
                )

                if res["appId"] is None or res["appId"] < 100000:
                    print(f"Failed to invoke {func_name} with {nthreads} threads")
                    hrperf_pause()
                    return
                
                # TODO: I actually don't know what's this field
                expected_num_messages = res["expectedNumMessages"]
                invoked_apps_ids.append(res["appId"])
            
            # Wait for all apps to finish
            results = client.wait_for_completion(set(invoked_apps_ids),
                                                 expected_num_messages,
                                                 poll_interval_secs=async_pool_interval / 1000.0)

            # Process results and find the earliest start and latest finish
            start_times = []
            finish_times = []
            individual_durations = []
            for app_id, res in results.items():
                if "metrics" in res and res["metrics"]:
                    for metric in res["metrics"]:
                        if "startTimestamp" in metric and "finishTimestamp" in metric:
                            start_times.append(metric["startTimestamp"])
                            finish_times.append(metric["finishTimestamp"])
                            individual_durations.append(metric["finishTimestamp"] - metric["startTimestamp"])
            earliestStart = min(start_times)
            latestFinish = max(finish_times)

            # NOTE: This involves the delay of spining up bunch of executors
            total_duration_sec = (latestFinish - earliestStart) / 1000.0
            avg_individual_duration_sec = statistics.mean(individual_durations) / 1000.0
            
            print(f"Threads: {nthreads}, Start: {earliestStart}, Finish: {latestFinish}, " 
                  f"Duration: {total_duration_sec:.3f}s, Avg Individual: {avg_individual_duration_sec:.3f}s")
            time.sleep(1)
    
    hrperf_pause()
    print("== Profiling done ==")