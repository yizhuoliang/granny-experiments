from faasmctl.util.flush import flush_workers
from faasmctl.util.config import get_faasm_worker_ips
from faasmctl.util.planner import reset as reset_planner
from invoke import task
import time
import json
import statistics
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import logging
from pystream import StreamBenchmark, StreamOperation
from faasmctl.cpp_client import create_client
from tasks.overlap.hrperf_api import (hrperf_start, hrperf_pause)

# Configure logging - reduced verbosity
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mempress_profiler")

# Constants
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

# Memory pressure configurations
PRESSURE_THREADS = [0, 1, 4, 8, 16]  # 0 means no pressure
BASE_ARRAY_SIZE = 67108864  # 64MB base size

# CPU and NUMA configuration
CPU_SET = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 
           36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 
           70, 72, 74, 76, 78]
NUMA_NODES = [0]

# Timing constants
PRESSURE_WARMUP_SECONDS = 20  # Wait for pressure to stabilize
PRESSURE_RUNTIME_SECONDS = 60  # Total runtime for pressure

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

def setup_memory_pressure(num_threads):
    """
    Setup memory bandwidth pressure using pystream.
    
    Args:
        num_threads: Number of threads to use for pressure (0 means no pressure)
        
    Returns:
        Tuple of (StreamBenchmark instance or None, resource usage dict)
    """
    if num_threads == 0:
        return None, {"mem_bw_mb_per_sec": 0}
    
    array_size = BASE_ARRAY_SIZE * num_threads
    
    stream = StreamBenchmark(
        threads=num_threads,
        array_size=array_size,
        operation=StreamOperation.ADD,
        cpus=CPU_SET[:num_threads],
        numa_nodes=NUMA_NODES
    )
    
    # Configure for runtime mode
    stream.set_runtime(PRESSURE_RUNTIME_SECONDS)
    stream.set_silent_mode(True)
    
    # Start non-blocking
    logger.info(f"Starting memory pressure with {num_threads} threads...")
    stream.start(blocking=False)
    
    # Check if running properly
    time.sleep(1)
    if not stream.is_running():
        logger.error("Failed to start memory pressure")
        return None, {"mem_bw_mb_per_sec": 0}
    
    # Wait for pressure to stabilize
    logger.info(f"Waiting {PRESSURE_WARMUP_SECONDS}s for memory pressure to stabilize...")
    time.sleep(PRESSURE_WARMUP_SECONDS)
    
    # Get resource usage - focus on memory bandwidth
    resource_usage = stream.get_resource_usage()
    
    # Estimate memory bandwidth (this is a simplified estimate)
    # You might want to enhance this with actual bandwidth measurement
    mem_bw = 0
    if 'io_read_mb' in resource_usage and 'io_write_mb' in resource_usage:
        # Estimate bandwidth as total I/O over time
        total_io_mb = resource_usage['io_read_mb'] + resource_usage['io_write_mb']
        mem_bw = total_io_mb / PRESSURE_WARMUP_SECONDS  # MB/s
    
    # Add estimated bandwidth to resource usage
    resource_usage["mem_bw_mb_per_sec"] = mem_bw
    
    return stream, resource_usage

@task(default=True)
def prof_polybench_mempress(ctx, num_cpus, iterations=3):
    """
    Profile Polybench functions under different memory bandwidth pressure levels.
    
    Args:
        num_cpus: Number of CPUs available (default: 80)
        iterations: Number of iterations for each test for better accuracy (default: 3)
    """
    worker_ips = get_faasm_worker_ips()
    num_vms = len(worker_ips)
    num_cpus = int(num_cpus)
    iterations = int(iterations)
    
    assert num_vms == 1, "We only use a single worker for profiling."
    
    # Setup cluster
    reset_planner(num_vms)
    flush_workers()
    logger.info("Cluster ready")
    
    # Create client
    client = create_client()
    
    # Results storage
    results_data = []
    
    # CSV file setup
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    csv_file = f"polybench_mempress_results_{timestamp}.csv"
    
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['function', 'mem_bw_mb_per_sec', 'runtime', 'iteration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
        # Prewarm phase
        logger.info("== Prewarming functions ==")
        for func_name in POLYBENCH_FUNC_NAMES:
            logger.info(f"Prewarming: {func_name}")
            res = client.invoke_function(
                user=POLYBENCH_USER,
                function=func_name,
                input_data="",
                async_execution=False
            )
            
            if "metrics" not in res or not res["metrics"]:
                logger.error(f"Failed to prewarm {func_name}")
                return
                
        logger.info("== Prewarming done ==")
        
        # Let the system cool down a bit
        time.sleep(5)
        
        # Main profiling phase
        logger.info("== Starting memory pressure profiling ==")
        
        # Start hardware performance monitoring
        hrperf_start()
        
        # Run each function under different memory pressure levels
        for func_name in POLYBENCH_FUNC_NAMES:
            logger.info(f"Profiling function: {func_name}")
            
            for num_threads in PRESSURE_THREADS:
                # Results for this function at this pressure level across iterations
                iteration_runtimes = []
                mem_bw_value = 0
                
                for iteration in range(iterations):
                    # Setup memory pressure
                    stream, resource_usage = setup_memory_pressure(num_threads)
                    # NOTE: BUG! THIS field DOES NOT EXIST!
                    mem_bw_value = resource_usage.get("mem_bw_mb_per_sec", 0)
                    
                    # Run the function
                    logger.info(f"Running {func_name} (iter {iteration+1}/{iterations}, mem_bw: {mem_bw_value:.2f} MB/s)")
                    res = client.invoke_function(
                        user=POLYBENCH_USER,
                        function=func_name,
                        input_data="",
                        async_execution=False
                    )
                    
                    # Process results
                    runtime = 0
                    if "metrics" in res and res["metrics"]:
                        runtime = get_faasm_exec_time_from_metrics(res["metrics"])
                    else:
                        logger.error(f"Failed to get metrics for {func_name}")
                    
                    # Store results for this iteration
                    iteration_runtimes.append(runtime)
                    
                    # Write to CSV
                    writer.writerow({
                        'function': func_name,
                        'mem_bw_mb_per_sec': mem_bw_value,
                        'runtime': runtime,
                        'iteration': iteration + 1
                    })
                    csvfile.flush()  # Ensure data is written immediately
                    
                    # Stop memory pressure
                    if stream:
                        stream.stop()
                        time.sleep(2)  # Let system recover
                    
                    # Log results for this iteration
                    logger.info(f"Runtime: {runtime:.3f}s")
                
                # Calculate average runtime across iterations
                avg_runtime = statistics.mean(iteration_runtimes) if iteration_runtimes else 0
                
                # Add to results for plotting
                results_data.append({
                    'function': func_name,
                    'mem_bw_mb_per_sec': mem_bw_value,
                    'avg_runtime': avg_runtime
                })
                
                logger.info(f"Average runtime: {avg_runtime:.3f}s")
        
        # Stop hardware performance monitoring
        hrperf_pause()
    
    logger.info(f"Raw results saved to {csv_file}")
    
    # Generate visualizations from averaged data
    generate_visualization(results_data)
    
    logger.info("== Memory pressure profiling completed ==")

def generate_visualization(results_data):
    """
    Generate visualizations using the averaged results data.
    
    Args:
        results_data: List of dictionaries with averaged results data
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Group results by function
    func_results = {}
    for entry in results_data:
        func = entry['function']
        if func not in func_results:
            func_results[func] = []
        func_results[func].append(entry)
    
    # Sort each function's results by memory bandwidth
    for func in func_results:
        func_results[func].sort(key=lambda x: x['mem_bw_mb_per_sec'])
    
    # 1. Line plot: Memory Bandwidth vs Runtime for each function
    plt.figure(figsize=(14, 10))
    
    for func, results in func_results.items():
        bw_values = [r['mem_bw_mb_per_sec'] for r in results]
        runtime_values = [r['avg_runtime'] for r in results]
        
        # Normalize by the lowest-pressure runtime
        base_runtime = runtime_values[0] if runtime_values[0] > 0 else 1.0
        normalized_runtimes = [rt / base_runtime for rt in runtime_values]
        
        plt.plot(bw_values, normalized_runtimes, 'o-', label=func)
    
    plt.xlabel('Memory Bandwidth Pressure (MB/s)')
    plt.ylabel('Normalized Runtime (higher = slower)')
    plt.title('Impact of Memory Bandwidth Pressure on Polybench Functions')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_file = f"polybench_mempress_line_{timestamp}.png"
    plt.savefig(plot_file)
    logger.info(f"Line plot saved to {plot_file}")

    # 2. Heatmap: Memory Bandwidth Sensitivity
    plt.figure(figsize=(12, 10))
    
    # Get unique functions and bandwidth values
    funcs = sorted(list(func_results.keys()))
    
    # Create matrix of bandwidth values vs functions
    # For each function, we'll show performance degradation at each bandwidth level
    
    # First, find all unique bandwidth values
    all_bw_values = sorted(list(set(entry['mem_bw_mb_per_sec'] for entry in results_data)))
    
    # If we have too many unique BW values, bin them for better visualization
    if len(all_bw_values) > 10:
        # Simple binning into 5 categories
        bw_bins = np.linspace(min(all_bw_values), max(all_bw_values), 6)
        bw_labels = [f"{bw_bins[i]:.0f}-{bw_bins[i+1]:.0f}" for i in range(len(bw_bins)-1)]
    else:
        bw_bins = None
        bw_labels = [str(bw) for bw in all_bw_values]
    
    # Create sensitivity matrix
    sensitivity_data = np.zeros((len(funcs), len(bw_labels)))
    
    for i, func in enumerate(funcs):
        results = func_results[func]
        
        # Get baseline runtime (no/minimal pressure)
        base_runtime = results[0]['avg_runtime'] if results[0]['avg_runtime'] > 0 else 1.0
        
        for result in results[1:]:  # Skip the first one (baseline)
            bw = result['mem_bw_mb_per_sec']
            runtime = result['avg_runtime']
            
            # Calculate sensitivity as percentage slowdown
            sensitivity = (runtime / base_runtime - 1.0) * 100
            
            # Find which bin this bw falls into
            if bw_bins is not None:
                bin_idx = np.digitize(bw, bw_bins) - 1
                if bin_idx >= len(bw_labels):
                    bin_idx = len(bw_labels) - 1
            else:
                bin_idx = all_bw_values.index(bw)
            
            sensitivity_data[i, bin_idx] = sensitivity
    
    # Create heatmap
    cmap = plt.cm.get_cmap('YlOrRd')
    
    plt.imshow(sensitivity_data, cmap=cmap, aspect='auto')
    plt.colorbar(label='Slowdown (%)')
    plt.xticks(range(len(bw_labels)), bw_labels, rotation=45)
    plt.yticks(range(len(funcs)), funcs)
    plt.xlabel('Memory Bandwidth Pressure (MB/s)')
    plt.ylabel('Polybench Function')
    plt.title('Sensitivity to Memory Bandwidth Pressure')
    plt.tight_layout()
    
    heatmap_file = f"polybench_mempress_heatmap_{timestamp}.png"
    plt.savefig(heatmap_file)
    logger.info(f"Heatmap saved to {heatmap_file}")
    
    # 3. Scatter plot: Function runtime vs memory bandwidth
    plt.figure(figsize=(14, 10))
    
    # Use different colors for different functions
    color_idx = 0
    colors = plt.cm.tab20(np.linspace(0, 1, len(funcs)))
    
    for func in funcs:
        results = func_results[func]
        bw_values = [r['mem_bw_mb_per_sec'] for r in results]
        runtime_values = [r['avg_runtime'] for r in results]
        
        plt.scatter(bw_values, runtime_values, label=func, color=colors[color_idx], s=100)
        # Add best-fit line
        z = np.polyfit(bw_values, runtime_values, 1)
        p = np.poly1d(z)
        plt.plot(bw_values, p(bw_values), '--', color=colors[color_idx])
        
        color_idx += 1
    
    plt.xlabel('Memory Bandwidth Pressure (MB/s)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Polybench Function Runtime vs Memory Bandwidth Pressure')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    scatter_file = f"polybench_mempress_scatter_{timestamp}.png"
    plt.savefig(scatter_file)
    logger.info(f"Scatter plot saved to {scatter_file}")

if __name__ == "__main__":
    from invoke import Program, Collection
    
    ns = Collection()
    ns.add_task(prof_polybench_mempress)
    
    program = Program(namespace=ns)
    program.run()