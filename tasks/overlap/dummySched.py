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

# Import the C++ client
from faasmctl.cpp_client import create_client

POLYBENCH_USER = "polybench"
# Updated function definitions with integrated repeat counts and memory bandwidth requirements
POLYBENCH_FUNCS_A = [
    {"name": "poly_deriche", "repeats": 500, "mem_bandwidth": 2830.1941},
]
POLYBENCH_FUNCS_B = [
    {"name": "poly_doitgen", "repeats": 150, "mem_bandwidth": 108.8122},
]

# A helper to select the polybench benchmarks to run
def _get_poly_benchmarks(bench):
    tasks_a = []
    tasks_b = []
    if bench:
        found = False
        # Check in group A
        for func_config in POLYBENCH_FUNCS_A:
            if func_config["name"] == bench:
                tasks_a = [(bench, run_index, func_config["mem_bandwidth"]) for run_index in range(func_config["repeats"])]
                found = True
                break
        
        # Check in group B if not found in A
        if not found:
            for func_config in POLYBENCH_FUNCS_B:
                if func_config["name"] == bench:
                    tasks_b = [(bench, run_index, func_config["mem_bandwidth"]) for run_index in range(func_config["repeats"])]
                    found = True
                    break
        
        if not found:
            raise RuntimeError(
                f"Unrecognised benchmark: {bench}. Must be one in: {[f['name'] for f in POLYBENCH_FUNCS_A + POLYBENCH_FUNCS_B]}"
            )
    else:
        # If no specific benchmark is provided, add all benchmarks with their repeats
        for func_config in POLYBENCH_FUNCS_A:
            for run_index in range(func_config["repeats"]):
                tasks_a.append((func_config["name"], run_index, func_config["mem_bandwidth"]))
        
        for func_config in POLYBENCH_FUNCS_B:
            for run_index in range(func_config["repeats"]):
                tasks_b.append((func_config["name"], run_index, func_config["mem_bandwidth"]))
    
    return tasks_a, tasks_b

# Extract execution time from metrics
def get_faasm_exec_time_from_metrics(metrics):
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

# Generate Gantt chart from execution data
def generate_gantt_chart(execution_data, strategy, num_cpus_per_vm):
    # First, collect all the task data for plotting
    tasks = []
    
    # Find the earliest start timestamp to use as a reference point
    min_start_timestamp = float('inf')
    for func_name, metrics in execution_data.items():
        for i, app_id in enumerate(metrics["app_ids"]):
            if i < len(metrics["start_timestamps"]) and metrics["start_timestamps"][i] < min_start_timestamp:
                min_start_timestamp = metrics["start_timestamps"][i]
    
    # If we don't have valid timestamps, use 0
    if min_start_timestamp == float('inf'):
        min_start_timestamp = 0
    
    # Prepare data for plotting
    for func_name, metrics in execution_data.items():
        for i, app_id in enumerate(metrics["app_ids"]):
            if i < len(metrics["start_timestamps"]) and i < len(metrics["finish_timestamps"]):
                # Calculate relative times
                start_time = (metrics["start_timestamps"][i] - min_start_timestamp) / 1000.0  # in seconds
                finish_time = (metrics["finish_timestamps"][i] - min_start_timestamp) / 1000.0  # in seconds
                
                # Skip invalid entries
                if start_time >= finish_time:
                    continue
                
                # Get the host if available
                host = metrics["hosts"][i] if i < len(metrics["hosts"]) else "unknown"
                
                # Normalize host name for display
                short_host = host.split('.')[-1] if '.' in host else host
                
                tasks.append({
                    'Task': f"{func_name}-{i}",
                    'Start': start_time,
                    'Finish': finish_time,
                    'Duration': finish_time - start_time,
                    'Function': func_name,
                    'Host': short_host,
                    'AppId': app_id
                })
    
    # Sort tasks by start time
    tasks.sort(key=lambda x: x['Start'])
    
    # Create figure and axis
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Set up colors for different functions
    function_names = list(set([task['Function'] for task in tasks]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(function_names)))
    function_colors = {func: colors[i] for i, func in enumerate(function_names)}
    
    # Plot each task as a horizontal bar
    y_ticks = []
    y_labels = []
    
    for i, task in enumerate(tasks):
        ax.barh(i, task['Duration'], left=task['Start'], color=function_colors[task['Function']], alpha=0.8)
        
        y_ticks.append(i)
        y_labels.append(f"{task['Function']}-{i}")
    
    # Calculate the makespan (max finish time)
    makespan = max([task['Finish'] for task in tasks]) if tasks else 0
    
    # Add a vertical line at the makespan
    ax.axvline(x=makespan, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(makespan + 0.1, len(tasks) - 1, f'Makespan: {makespan:.2f}s', 
            color='red', fontsize=10, va='center')
    
    # Add a title and labels
    ax.set_title(f'Function Execution Timeline (Strategy: {strategy}, CPUs per VM: {num_cpus_per_vm})', fontsize=14)
    ax.set_xlabel('Time (seconds from first start)', fontsize=12)
    ax.set_ylabel('Tasks', fontsize=12)
    
    # Set y-ticks and make sure they fit
    if len(y_ticks) > 20:
        # If too many tasks, only show some of the ticks
        step = max(1, len(y_ticks) // 20)
        ax.set_yticks([y_ticks[i] for i in range(0, len(y_ticks), step)])
        ax.set_yticklabels([y_labels[i] for i in range(0, len(y_labels), step)])
    else:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
    
    # Add a legend for function types
    handles = [plt.Rectangle((0, 0), 1, 1, color=function_colors[func]) for func in function_names]
    ax.legend(handles, function_names, loc='upper right', title='Functions')
    
    # Adjust the plot layout
    plt.tight_layout()
    
    # Save the figure
    timestamp = int(time.time())
    filename = f'gantt_chart_{strategy}_{num_cpus_per_vm}cpus_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGantt chart saved to {filename}")
    
    return makespan

# New function to create a bandwidth-aware scheduling plan using iterative optimization
def create_bandwidth_aware_schedule(tasks_a, tasks_b, prewarm_times, num_cpus):
    """
    Create a schedule that minimizes memory bandwidth contention using iterative optimization
    
    Args:
        tasks_a: List of tuples (benchmark_name, run_index, mem_bandwidth) for group A (deriche)
        tasks_b: List of tuples (benchmark_name, run_index, mem_bandwidth) for group B (doitgen)
        prewarm_times: Dictionary with benchmark execution times from prewarm phase
        num_cpus: Number of available CPU cores
        
    Returns:
        List of tasks in scheduled order: [(benchmark_name, run_index, mem_bandwidth), ...]
    """
    num_cpus = int(num_cpus)
    # Get function names for easy reference
    deriche_name = POLYBENCH_FUNCS_A[0]["name"]  # poly_deriche
    doitgen_name = POLYBENCH_FUNCS_B[0]["name"]  # poly_doitgen
    
    # Get estimated execution times from prewarm phase or use defaults
    deriche_time = prewarm_times.get(deriche_name, 1.0)
    doitgen_time = prewarm_times.get(doitgen_name, 3.0)  # doitgen takes ~3x longer
    
    print(f"Estimated execution times - {deriche_name}: {deriche_time:.3f}s, {doitgen_name}: {doitgen_time:.3f}s")
    print(f"Time ratio (doitgen/deriche): {doitgen_time/deriche_time:.2f}x")
    
    # Get bandwidth requirements
    deriche_bw = POLYBENCH_FUNCS_A[0]["mem_bandwidth"]  # ~2830 byte/us
    doitgen_bw = POLYBENCH_FUNCS_B[0]["mem_bandwidth"]   # ~109 byte/us
    
    # Divide tasks into types
    deriche_tasks = [t for t in tasks_a if t[0] == deriche_name]
    doitgen_tasks = [t for t in tasks_b if t[0] == doitgen_name]
    
    print(f"Task counts - {deriche_name}: {len(deriche_tasks)}, {doitgen_name}: {len(doitgen_tasks)}")
    
    # Create schedule using iterative optimization
    schedule = []
    
    # Calculate optimal ratio of deriche to doitgen tasks
    # Based on execution time and bandwidth requirements
    # We want to balance total bandwidth usage over time
    
    # Time-normalized bandwidth for both task types
    deriche_time_bw = deriche_bw * deriche_time
    doitgen_time_bw = doitgen_bw * doitgen_time
    
    # Calculate optimal ratio (deriche:doitgen)
    # This balances aggregate bandwidth consumption over time
    ratio_deriche_to_doitgen = doitgen_time_bw / deriche_time_bw
    
    # Round to nearest integer ratio for practical scheduling
    if ratio_deriche_to_doitgen >= 1:
        ratio_num = round(ratio_deriche_to_doitgen)
        ratio_denom = 1
    else:
        ratio_denom = round(1 / ratio_deriche_to_doitgen)
        ratio_num = 1
    
    print(f"Optimal ratio (deriche:doitgen): {ratio_deriche_to_doitgen:.2f} ≈ {ratio_num}:{ratio_denom}")
    
    # Calculate how many tasks of each type we can run in parallel
    # considering the number of available cores and our calculated ratio
    total_ratio_units = ratio_num + ratio_denom
    
    # If we have fewer cores than ratio units, we need to scale down
    if num_cpus < total_ratio_units:
        # Scale to fit available cores
        scale_factor = num_cpus / total_ratio_units
        parallel_deriche = max(1, round(ratio_num * scale_factor))
        parallel_doitgen = max(1, round(ratio_denom * scale_factor))
    else:
        # We can run the full ratio in parallel and may have cores to spare
        parallel_deriche = ratio_num
        parallel_doitgen = ratio_denom
        
        # If we have extra cores, distribute them proportionally
        extra_cores = num_cpus - (parallel_deriche + parallel_doitgen)
        if extra_cores > 0:
            # Distribute extra cores based on which task has more remaining
            deriche_portion = len(deriche_tasks) / (len(deriche_tasks) + len(doitgen_tasks))
            extra_deriche = round(extra_cores * deriche_portion)
            extra_doitgen = extra_cores - extra_deriche
            
            parallel_deriche += extra_deriche
            parallel_doitgen += extra_doitgen
    
    print(f"Parallel execution - {deriche_name}: {parallel_deriche}, {doitgen_name}: {parallel_doitgen}")
    
    # Function to generate batches considering the optimal ratio
    def generate_batch():
        batch = []
        if len(deriche_tasks) >= parallel_deriche and len(doitgen_tasks) >= parallel_doitgen:
            # Add both types according to our ratio
            batch.extend(deriche_tasks[:parallel_deriche])
            batch.extend(doitgen_tasks[:parallel_doitgen])
            
            # Remove scheduled tasks
            del deriche_tasks[:parallel_deriche]
            del doitgen_tasks[:parallel_doitgen]
        elif len(deriche_tasks) > 0:
            # Add as many deriche tasks as possible (up to core count)
            to_add = min(len(deriche_tasks), num_cpus)
            batch.extend(deriche_tasks[:to_add])
            del deriche_tasks[:to_add]
        elif len(doitgen_tasks) > 0:
            # Add as many doitgen tasks as possible (up to core count)
            to_add = min(len(doitgen_tasks), num_cpus)
            batch.extend(doitgen_tasks[:to_add])
            del doitgen_tasks[:to_add]
        
        return batch
    
    # Generate batches until all tasks are scheduled
    while deriche_tasks or doitgen_tasks:
        batch = generate_batch()
        if not batch:
            break  # Safety check
        schedule.extend(batch)
    
    # Calculate expected total memory bandwidth for this schedule
    batch_count = 0
    deriche_count = len([t for t in schedule if t[0] == deriche_name])
    doitgen_count = len([t for t in schedule if t[0] == doitgen_name])
    
    # Count how many batches we created
    remaining_tasks = schedule.copy()
    while remaining_tasks:
        batch_size = min(num_cpus, len(remaining_tasks))
        remaining_tasks = remaining_tasks[batch_size:]
        batch_count += 1
    
    print(f"Schedule statistics:")
    print(f"  - Total tasks: {len(schedule)} ({deriche_count} {deriche_name}, {doitgen_count} {doitgen_name})")
    print(f"  - Estimated batches: {batch_count}")
    
    # Calculate theoretical makespan (ignoring overhead)
    total_deriche_time = deriche_count * deriche_time / parallel_deriche
    total_doitgen_time = doitgen_count * doitgen_time / parallel_doitgen
    theoretical_makespan = max(total_deriche_time, total_doitgen_time)
    
    print(f"  - Theoretical makespan: {theoretical_makespan:.2f}s (ignoring overhead)")
    
    return schedule
    
    return schedule

@task(default=True)
def polyfunc4(ctx, num_cpus_per_vm, bench=None, poll_interval=50, strategy="bandwidth_aware", ratio_override=None):
    """
    Run the PolyBench/C microbenchmark tasks concurrently on Faasm.
    
    This function uses a three-phase approach:
    0. Pre-warm phase: invoke each function once synchronously to avoid internal bugs
    1. First, invoke all functions asynchronously according to the scheduling strategy
    2. Then check status of each invocation until all complete
    
    Tasks are scheduled according to the specified strategy:
    
      - "interleave": Alternately dispatch tasks from group A and group B.
      - "a_then_b": Dispatch all tasks from group A first, then group B.
      - "bandwidth_aware": Schedule tasks to minimize memory bandwidth contention using iterative optimization.
    
    Additional parameters:
      - ratio_override: Optional manual override for the deriche:doitgen ratio (format: "d:g" e.g. "3:1")
    """
    # Discover workers and reset the planner
    worker_ips = get_faasm_worker_ips()
    num_vms = len(worker_ips)
    num_cpus_per_vm = int(num_cpus_per_vm)
    
    # Assert that we have exactly 1 VM for simplicity
    assert num_vms == 1, "This implementation assumes exactly 1 VM. Found {num_vms}."
    
    reset_planner(num_vms)
    print(f"Planner reset done, num vms: {num_vms}")
    # Clear the host state
    flush_workers()

    # Initialize the C++ client
    client = create_client()

    # Get all unique benchmarks that will be run
    unique_benchmarks = set()
    if bench:
        unique_benchmarks.add(bench)
    else:
        unique_benchmarks.update([func["name"] for func in POLYBENCH_FUNCS_A])
        unique_benchmarks.update([func["name"] for func in POLYBENCH_FUNCS_B])
    
    # Phase 0: Pre-warm functions to avoid internal bugs
    print("\n=== Phase 0: Pre-warming functions ===")
    
    # Dictionary to store prewarm execution times
    prewarm_times = {}
    
    for benchmark in unique_benchmarks:
        print(f"Pre-warming function: {benchmark}")
        prewarm_result = client.invoke_function(
            user=POLYBENCH_USER,
            function=benchmark,
            input_data="",
            async_execution=False  # Synchronous execution for pre-warming
        )
        
        # Extract metrics from pre-warm run
        if "metrics" in prewarm_result and prewarm_result["metrics"]:
            prewarm_time = get_faasm_exec_time_from_metrics(prewarm_result["metrics"])
            prewarm_times[benchmark] = prewarm_time
            print(f"Pre-warming {benchmark} completed successfully in {prewarm_time:.6f} seconds")
        else:
            # Set a default time if metrics not available
            prewarm_times[benchmark] = 1.0  # Default 1 second
            print(f"Pre-warming {benchmark} completed successfully")
    
    print("All functions pre-warmed successfully")
    
    # Add a small delay after pre-warming to ensure the system is ready
    time.sleep(2)
    
    # Build the task pool: one entry per (benchmark, repeat) for each group A and B
    tasks_a, tasks_b = _get_poly_benchmarks(bench)
    
    # Display the tasks configuration
    print("\nTask configuration:")
    for group_name, tasks in [("Group A", tasks_a), ("Group B", tasks_b)]:
        task_counts = {}
        for task, _, _ in tasks:
            task_counts[task] = task_counts.get(task, 0) + 1
        
        for task, count in task_counts.items():
            print(f"  {group_name} - {task}: {count} iterations")
    
    # Dictionary to store detailed metrics per function
    function_metrics = {}
    
    # Dictionary to track all invoked functions
    # {app_id: {"benchmark": benchmark, "run_index": run_index, "num_messages": 1}}
    invoked_apps = {}
    
    # Get schedule based on strategy
    if strategy == "bandwidth_aware":
        print("\nUsing bandwidth-aware scheduling strategy")
        
        # Parse ratio override if provided
        manual_ratio = None
        if ratio_override:
            try:
                ratio_parts = ratio_override.split(':')
                if len(ratio_parts) == 2:
                    deriche_ratio = int(ratio_parts[0])
                    doitgen_ratio = int(ratio_parts[1])
                    if deriche_ratio > 0 and doitgen_ratio > 0:
                        manual_ratio = (deriche_ratio, doitgen_ratio)
                        print(f"Using manual ratio override: {deriche_ratio}:{doitgen_ratio}")
            except (ValueError, IndexError):
                print(f"Invalid ratio override format: {ratio_override}, should be 'd:g' (e.g. '3:1')")
        
        schedule = create_bandwidth_aware_schedule(
            tasks_a, tasks_b, prewarm_times, num_cpus_per_vm
        )
        print(f"Generated schedule with {len(schedule)} tasks")
    else:
        # For other strategies, we'll handle scheduling during invocation
        schedule = None
        
    # Phase 1: Invoke all functions asynchronously
    print("\n=== Phase 1: Invoking all functions asynchronously ===")
    phase1_start_time = time.perf_counter()
    
    # For bandwidth-aware scheduling, follow the pre-computed schedule
    if strategy == "bandwidth_aware" and schedule:
        remaining_schedule = schedule.copy()
        
        while remaining_schedule:
            # Get current resource metrics
            idle_vms, idle_cpus, utilization, host_usage = client.get_cluster_utilization(num_vms, num_cpus_per_vm)
            # print(f"idle CPUs: {idle_cpus}, tasks remaining: {len(remaining_schedule)}")
            
            # While there are free cores and pending tasks, schedule new tasks
            while idle_cpus > 0 and remaining_schedule:
                benchmark, run_index, mem_bandwidth = remaining_schedule.pop(0)
                
                # Invoke the function asynchronously
                result = client.invoke_function(
                    user=POLYBENCH_USER,
                    function=benchmark,
                    input_data="",
                    async_execution=True
                )
                
                # Track this invocation
                app_id = result["appId"]
                expected_num_messages = result["expectedNumMessages"]
                invoked_apps[app_id] = {
                    "benchmark": benchmark, 
                    "run_index": run_index,
                    "num_messages": expected_num_messages,
                    "start_time": time.perf_counter()  # Track wall-clock start time
                }
                
                # Initialize metrics tracking for this benchmark if not already present
                if benchmark not in function_metrics:
                    function_metrics[benchmark] = {
                        "runtimes": [],             # Pure function execution time
                        "wall_clock_times": [],     # End-to-end wall clock time
                        "hosts": [],                # Hosts that ran this function
                        "return_values": [],        # Return values from each run
                        "app_ids": [],              # App IDs for each invocation
                        "start_timestamps": [],     # Start timestamps for Gantt chart
                        "finish_timestamps": []     # Finish timestamps for Gantt chart
                    }
                
                # Decrement available CPUs
                idle_cpus -= 1
            
            # If more tasks remain but no idle cores, wait a short while before checking again
            if remaining_schedule and idle_cpus == 0:
                # print(f"Waiting for resources to free up...")
                time.sleep(poll_interval / 1000.0)
    else:
        # Use original scheduling strategies (interleave or a_then_b)
        toggle = True  # Toggle used for interleaving strategy
        
        # Process tasks until all are dispatched
        while tasks_a or tasks_b:
            # Get current resource metrics
            idle_vms, idle_cpus, utilization, host_usage = client.get_cluster_utilization(num_vms, num_cpus_per_vm)
            # print(f"idle CPUs: {idle_cpus}, tasks remaining: A:{len(tasks_a)}, B:{len(tasks_b)}")
            
            # While there are free cores and pending tasks, schedule new tasks
            while idle_cpus > 0 and (tasks_a or tasks_b):
                if strategy == "interleave":
                    if toggle:
                        if tasks_a:
                            benchmark, run_index, mem_bandwidth = tasks_a.pop(0)
                        elif tasks_b:
                            benchmark, run_index, mem_bandwidth = tasks_b.pop(0)
                        else:
                            break
                    else:
                        if tasks_b:
                            benchmark, run_index, mem_bandwidth = tasks_b.pop(0)
                        elif tasks_a:
                            benchmark, run_index, mem_bandwidth = tasks_a.pop(0)
                        else:
                            break
                    toggle = not toggle
                elif strategy == "a_then_b":
                    if tasks_a:
                        benchmark, run_index, mem_bandwidth = tasks_a.pop(0)
                    elif tasks_b:
                        benchmark, run_index, mem_bandwidth = tasks_b.pop(0)
                    else:
                        break
                else:
                    raise RuntimeError(f"Unrecognised scheduling strategy: {strategy}. Use 'interleave', 'a_then_b', or 'bandwidth_aware'.")
                
                # print(f"Starting function {benchmark}, run {run_index}")
                
                # Invoke the function asynchronously
                result = client.invoke_function(
                    user=POLYBENCH_USER,
                    function=benchmark,
                    input_data="",
                    async_execution=True
                )
                
                # Track this invocation
                app_id = result["appId"]
                expected_num_messages = result["expectedNumMessages"]
                invoked_apps[app_id] = {
                    "benchmark": benchmark, 
                    "run_index": run_index,
                    "num_messages": expected_num_messages,
                    "start_time": time.perf_counter()  # Track wall-clock start time
                }
                
                # Initialize metrics tracking for this benchmark if not already present
                if benchmark not in function_metrics:
                    function_metrics[benchmark] = {
                        "runtimes": [],             # Pure function execution time
                        "wall_clock_times": [],     # End-to-end wall clock time
                        "hosts": [],                # Hosts that ran this function
                        "return_values": [],        # Return values from each run
                        "app_ids": [],              # App IDs for each invocation
                        "start_timestamps": [],     # Start timestamps for Gantt chart
                        "finish_timestamps": []     # Finish timestamps for Gantt chart
                    }
                
                # Decrement available CPUs
                idle_cpus -= 1

            # If more tasks remain but no idle cores, wait a short while before checking again
            if (tasks_a or tasks_b) and idle_cpus == 0:
                # print(f"Waiting for resources to free up...")
                time.sleep(poll_interval / 1000.0)
    
    phase1_end_time = time.perf_counter()
    invocation_time = phase1_end_time - phase1_start_time
    print(f"All {len(invoked_apps)} functions invoked in {invocation_time:.6f} seconds")
    
    # Phase 2: Check status of all invocations until all complete
    print("\n=== Phase 2: Checking function completion status ===")
    phase2_start_time = time.perf_counter()
    
    # Poll until all apps are complete
    remaining_apps = set(invoked_apps.keys())
    print(f"Starting phase 2 with {len(remaining_apps)} apps to check")
    
    # Use the wait_for_completion method to efficiently wait for all apps to finish
    app_results = client.wait_for_completion(
        list(remaining_apps),
        expected_num_messages=1,
        poll_interval_secs=poll_interval / 1000.0
    )
    
    # Process results
    for app_id, status in app_results.items():
        app_info = invoked_apps[app_id]
        benchmark = app_info["benchmark"]
        run_index = app_info["run_index"]
        wall_clock_end_time = time.perf_counter()
        wall_clock_time = wall_clock_end_time - app_info["start_time"]
        
        # Extract execution metrics
        if status["finished"] and "metrics" in status and status["metrics"]:
            actual_time = get_faasm_exec_time_from_metrics(status["metrics"])
            print(f"Function: {benchmark}, run: {run_index}, Runtime: {actual_time:.6f} seconds, Wall-clock: {wall_clock_time:.6f} seconds")
            
            # Record the metrics
            function_metrics[benchmark]["runtimes"].append(actual_time)
            function_metrics[benchmark]["wall_clock_times"].append(wall_clock_time)
            function_metrics[benchmark]["app_ids"].append(app_id)
            
            # Extract timestamps for Gantt chart
            start_ts = None
            finish_ts = None
            host = None
            
            for metric in status["metrics"]:
                if "startTimestamp" in metric and metric["startTimestamp"] > 0:
                    start_ts = metric["startTimestamp"]
                
                if "finishTimestamp" in metric and metric["finishTimestamp"] > 0:
                    finish_ts = metric["finishTimestamp"]
                
                if "executedHost" in metric and metric["executedHost"]:
                    host = metric["executedHost"]
            
            if start_ts and finish_ts:
                function_metrics[benchmark]["start_timestamps"].append(start_ts)
                function_metrics[benchmark]["finish_timestamps"].append(finish_ts)
                if host:
                    function_metrics[benchmark]["hosts"].append(host)
                else:
                    function_metrics[benchmark]["hosts"].append("unknown")
            
            # Record return value
            if "messageResults" in status and status["messageResults"]:
                for msg in status["messageResults"]:
                    if "returnValue" in msg:
                        function_metrics[benchmark]["return_values"].append(msg["returnValue"])
                        break
        else:
            print(f"Function: {benchmark}, run: {run_index}, Result: No metrics available")
    
    phase2_end_time = time.perf_counter()
    total_time = phase2_end_time - phase1_start_time
    completion_time = phase2_end_time - phase2_start_time
    
    print(f"\nInvocation time: {invocation_time:.6f} seconds")
    print(f"Completion waiting time: {completion_time:.6f} seconds")
    print(f"Total end-to-end time: {total_time:.6f} seconds")

    # Calculate and print detailed statistics for each function
    print("\nDetailed function statistics:")
    for func, metrics in function_metrics.items():
        runtimes = metrics["runtimes"]
        wall_times = metrics["wall_clock_times"]
        hosts = metrics["hosts"]
        
        if not runtimes:
            print(f"Function {func}: No valid runtime data available")
            continue
        
        # Calculate statistics
        avg_runtime = sum(runtimes) / len(runtimes)
        avg_wall_time = sum(wall_times) / len(wall_times)
        
        # Additional statistics if we have enough data points
        if len(runtimes) >= 3:
            median_runtime = statistics.median(runtimes)
            stdev_runtime = statistics.stdev(runtimes) if len(runtimes) > 1 else 0
            min_runtime = min(runtimes)
            max_runtime = max(runtimes)
            
            print(f"Function {func}:")
            print(f"  Runs: {len(runtimes)}")
            print(f"  Average runtime: {avg_runtime:.6f} seconds")
            print(f"  Median runtime: {median_runtime:.6f} seconds")
            print(f"  Std deviation: {stdev_runtime:.6f} seconds")
            print(f"  Min/Max runtime: {min_runtime:.6f}/{max_runtime:.6f} seconds")
            print(f"  Average wall-clock time: {avg_wall_time:.6f} seconds")
            print(f"  Unique hosts: {len(set(hosts))}")
            print(f"  Hosts: {', '.join(set(hosts))}")
        else:
            print(f"Function {func}:")
            print(f"  Runs: {len(runtimes)}")
            print(f"  Average runtime: {avg_runtime:.6f} seconds")
            print(f"  Average wall-clock time: {avg_wall_time:.6f} seconds")
    
    # Generate the Gantt chart
    makespan = generate_gantt_chart(function_metrics, strategy, num_cpus_per_vm)
    print(f"Makespan: {makespan:.6f} seconds")