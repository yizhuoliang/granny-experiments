import sys
import re
import duckdb
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend so we don't call plt.show()
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_node_bw.py <path_to_profiling_log.txt>")
        sys.exit(1)

    log_path = sys.argv[1]
    if not os.path.isfile(log_path):
        print(f"Error: File '{log_path}' does not exist.")
        sys.exit(1)

    print(f"Analyzing profiling log: {log_path}")

    # --------------------------------------------------------------------------
    # 1. Parse the new profiling log format
    # --------------------------------------------------------------------------
    pattern = re.compile(
        r"^Threads: (\d+), Start: (\d+), Finish: (\d+), Duration: ([0-9.]+)s, Avg Individual: ([0-9.]+)s"
    )

    function_intervals = []
    current_function = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # Check if this is a function header line
            func_match = re.match(r"^Profiling func: ([A-Za-z0-9_\-]+)", line)
            if func_match:
                current_function = func_match.group(1)
                continue

            # Check if this is a thread measurement line
            thread_match = pattern.match(line)
            if thread_match and current_function is not None:
                threads = int(thread_match.group(1))
                start_ms = int(thread_match.group(2))
                end_ms = int(thread_match.group(3))
                duration = float(thread_match.group(4))
                avg_individual = float(thread_match.group(5))

                # Convert ms -> ns
                start_ns = start_ms * 1_000_000
                end_ns = end_ms * 1_000_000

                function_intervals.append({
                    'function_name': current_function,
                    'threads': threads,
                    'start_ns': start_ns,
                    'end_ns': end_ns,
                    'duration': duration,
                    'avg_individual': avg_individual
                })



    if not function_intervals:
        print("No function intervals found in the input file. Exiting.")
        sys.exit(0)

    print(f"Parsed {len(function_intervals)} function intervals across {len(set(interval['function_name'] for interval in function_intervals))} unique functions")

    # --------------------------------------------------------------------------
    # 2. Connect to DuckDB and create TEMP table
    # --------------------------------------------------------------------------
    con = duckdb.connect(database='analysis.duckdb')

    # Create function_invocations table with thread count
    con.execute("""
        CREATE TEMP TABLE function_invocations (
            function_name VARCHAR,
            threads      INTEGER,
            start_ns     BIGINT,
            end_ns       BIGINT,
            duration     FLOAT,
            avg_individual FLOAT
        )
    """)

    insert_sql = "INSERT INTO function_invocations VALUES (?, ?, ?, ?, ?, ?)"
    for interval in function_intervals:
        con.execute(insert_sql, [
            interval['function_name'],
            interval['threads'],
            interval['start_ns'],
            interval['end_ns'],
            interval['duration'],
            interval['avg_individual']
        ])

    # --------------------------------------------------------------------------
    # 3. Merge all function intervals
    # --------------------------------------------------------------------------
    con.execute("""
        CREATE TEMP TABLE sorted_intervals AS
        SELECT start_ns, end_ns
        FROM function_invocations
        ORDER BY start_ns
    """)

    con.execute("""
        CREATE TEMP TABLE interval_flags AS
        SELECT
            start_ns,
            end_ns,
            CASE
                WHEN start_ns <= LAG(end_ns, 1, 0) OVER (ORDER BY start_ns)
                THEN 0
                ELSE 1
            END AS is_new_group
        FROM sorted_intervals
    """)

    con.execute("""
        CREATE TEMP TABLE grouped_intervals AS
        SELECT
            start_ns,
            end_ns,
            SUM(is_new_group) OVER (
                ORDER BY start_ns
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS grp
        FROM interval_flags
        ORDER BY start_ns
    """)

    con.execute("""
        CREATE TEMP TABLE merged_func_intervals AS
        SELECT
            grp AS group_id,
            MIN(start_ns) AS merged_start,
            MAX(end_ns)   AS merged_end
        FROM grouped_intervals
        GROUP BY grp
        ORDER BY merged_start
    """)

    # --------------------------------------------------------------------------
    # 4. Global earliest & latest node_mem_bandwidth timestamps
    # --------------------------------------------------------------------------
    try:
        # First check if the node_memory_bandwidth table exists
        table_exists = con.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='node_memory_bandwidth'
        """).fetchone()

        if not table_exists:
            print("The node_memory_bandwidth table does not exist in the database.")
            print("Please make sure you have created this table with the memory bandwidth data.")
            con.close()
            sys.exit(1)

        earliest_nmb_start, latest_nmb_end = con.execute("""
            SELECT MIN(start_time_ns), MAX(end_time_ns)
            FROM node_memory_bandwidth
        """).fetchone()

        # If the node_memory_bandwidth table has no rows, handle that case:
        if earliest_nmb_start is None or latest_nmb_end is None:
            print("No data in node_memory_bandwidth table. Exiting.")
            con.close()
            sys.exit(0)

        print(f"Found node_memory_bandwidth data spanning from {earliest_nmb_start} to {latest_nmb_end} ns")
    except Exception as e:
        print(f"Error querying node_memory_bandwidth table: {e}")
        print("Make sure the node_memory_bandwidth table exists in the DuckDB database.")
        con.close()
        sys.exit(1)

    # --------------------------------------------------------------------------
    # 5. Create "background intervals" (no function running)
    # --------------------------------------------------------------------------
    con.execute(f"""
        CREATE TEMP TABLE global_bounds AS
        SELECT {earliest_nmb_start}::BIGINT AS global_start,
               {latest_nmb_end}::BIGINT   AS global_end
    """)

    con.execute("""
        CREATE TEMP TABLE background_candidates AS
        WITH cte_first AS (
            SELECT
                global_start AS seg_start,
                (SELECT merged_start FROM merged_func_intervals LIMIT 1) AS seg_end
            FROM global_bounds
        ),
        cte_mid AS (
            SELECT
                LAG(merged_end)  OVER (ORDER BY merged_start) AS seg_start,
                merged_start AS seg_end
            FROM merged_func_intervals
        ),
        cte_last AS (
            SELECT
                (SELECT MAX(merged_end) FROM merged_func_intervals) AS seg_start,
                global_end AS seg_end
            FROM global_bounds
        )
        SELECT seg_start, seg_end FROM cte_first
        UNION ALL
        SELECT seg_start, seg_end FROM cte_mid
        UNION ALL
        SELECT seg_start, seg_end FROM cte_last
    """)

    con.execute("""
        CREATE TEMP TABLE background_intervals AS
        SELECT seg_start, seg_end
        FROM background_candidates
        WHERE seg_start < seg_end
        ORDER BY seg_start
    """)

    # --------------------------------------------------------------------------
    # 6. Compute average background memory bandwidth
    # --------------------------------------------------------------------------
    background_query = r"""
    WITH overlap_cte AS (
        SELECT
            GREATEST(bg.seg_start, nmb.start_time_ns) AS overlap_start,
            LEAST(bg.seg_end,     nmb.end_time_ns)    AS overlap_end,
            nmb.memory_bandwidth_bytes_per_us         AS bandwidth
        FROM background_intervals bg
        JOIN node_memory_bandwidth nmb
             ON nmb.start_time_ns < bg.seg_end
             AND nmb.end_time_ns   > bg.seg_start
    )
    SELECT
       CASE WHEN SUM(overlap_end - overlap_start) = 0 THEN 0.0
            ELSE SUM(bandwidth * (overlap_end - overlap_start))
                 / SUM(overlap_end - overlap_start)
       END AS avg_bg_bw
    FROM overlap_cte
    WHERE overlap_end > overlap_start
    """

    try:
        background_bw = con.execute(background_query).fetchone()[0]
        if background_bw is None:
            background_bw = 0.0
        print(f"Calculated background bandwidth: {background_bw:.4f} bytes/us")
    except Exception as e:
        print(f"Error calculating background bandwidth: {e}")
        print("Setting background bandwidth to 0.0")
        background_bw = 0.0

    # --------------------------------------------------------------------------
    # 7. Compute original time-weighted average mem BW for each function by thread count
    # --------------------------------------------------------------------------
    function_query = r"""
    WITH intervals AS (
        SELECT
            fi.function_name AS function_name,
            fi.threads AS threads,
            GREATEST(fi.start_ns, nmb.start_time_ns) AS overlap_start,
            LEAST(fi.end_ns, nmb.end_time_ns) AS overlap_end,
            nmb.memory_bandwidth_bytes_per_us AS bandwidth
        FROM function_invocations fi
        JOIN node_memory_bandwidth nmb
             ON nmb.start_time_ns < fi.end_ns
             AND nmb.end_time_ns > fi.start_ns
    )
    SELECT
        function_name,
        threads,
        SUM(bandwidth * (overlap_end - overlap_start))
          / NULLIF(SUM(overlap_end - overlap_start), 0) AS avg_node_bw,
        SUM(overlap_end - overlap_start) AS overlap_duration
    FROM intervals
    WHERE overlap_end > overlap_start
    GROUP BY function_name, threads
    ORDER BY function_name, threads
    """

    results_df = con.execute(function_query).fetchdf()

    # --------------------------------------------------------------------------
    # 8. Print background BW & function stats
    # --------------------------------------------------------------------------
    print("=== Timeline / Background Check ===")
    print(f"Background average bandwidth: {background_bw:.4f} bytes/us (when no functions running)")
    print("====================================\n")

    # Calculate per-thread bandwidth
    results_df['orig_per_thread_bw'] = results_df['avg_node_bw'] / results_df['threads']
    results_df['adj_per_thread_bw'] = (results_df['avg_node_bw'] - background_bw) / results_df['threads']

    # Print both aggregate and per-thread values
    print("=== Node Memory Bandwidth (bytes/us) per Function and Thread Count ===")
    print("Format: Original (Per-Thread) | Adjusted (Per-Thread)")
    print("----------------------------------------------------------------")
    for idx, row in results_df.iterrows():
        func_name = row["function_name"]
        threads = row["threads"]
        orig_bw = row["avg_node_bw"] if row["avg_node_bw"] is not None else 0.0
        adj_bw = orig_bw - background_bw
        orig_per_thread = row["orig_per_thread_bw"]
        adj_per_thread = row["adj_per_thread_bw"]

        print(f"Function: {func_name:<15}  Threads: {threads:<3}  "
              f"Original: {orig_bw:.4f} ({orig_per_thread:.4f} per thread),  "
              f"Adjusted: {adj_bw:.4f} ({adj_per_thread:.4f} per thread)")

    print("==============================================================\n")

    # --------------------------------------------------------------------------
    # 9. Plot memory bandwidth by function and thread count (separate original and adjusted)
    # --------------------------------------------------------------------------

    # Get unique function names and thread counts
    unique_funcs = sorted(results_df["function_name"].unique())
    thread_counts = sorted(results_df["threads"].unique())

    # Set up colors for thread counts
    thread_colors = plt.cm.viridis(np.linspace(0, 0.9, len(thread_counts)))

    # Positions for function groups on x-axis
    x = np.arange(len(unique_funcs))

    # Set up bar width and positions
    bar_width = 0.8 / len(thread_counts)

    # -------- Original Total Bandwidth Plot --------
    fig_orig, ax_orig = plt.subplots(figsize=(15, 8))

    # If we have too many functions, adjust figure size
    if len(unique_funcs) > 15:
        fig_orig.set_figwidth(max(15, len(unique_funcs) * 0.8))

    # Create the bar chart for original total bandwidth
    for i, threads in enumerate(thread_counts):
        # Filter data for this thread count
        thread_data = results_df[results_df["threads"] == threads]

        # Prepare data in the order of unique_funcs
        bw_values = []

        for func in unique_funcs:
            func_row = thread_data[thread_data["function_name"] == func]
            if len(func_row) > 0:
                bw = func_row["avg_node_bw"].values[0] if func_row["avg_node_bw"].values[0] is not None else 0.0
                bw_values.append(bw)
            else:
                bw_values.append(0)

        # Position for this thread count
        thread_pos = i * bar_width - bar_width * (len(thread_counts) - 1) / 2

        # Plot original values
        ax_orig.bar(x + thread_pos,
                   bw_values,
                   bar_width * 0.9,
                   alpha=0.8,
                   color=thread_colors[i],
                   label=f'{threads} Threads')

    # Add labels and legend
    ax_orig.set_xlabel("Function Name", fontsize=12)
    ax_orig.set_ylabel("Total Memory Bandwidth (bytes/us)", fontsize=12)
    ax_orig.set_title("Original Total Node Memory Bandwidth by Function and Thread Count", fontsize=14)
    ax_orig.set_xticks(x)
    ax_orig.set_xticklabels(unique_funcs, rotation=45, ha='right', fontsize=10)


    # Create a custom legend
    handles, labels = ax_orig.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_orig.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_orig_png = "function_node_bw_original.png"
    plt.savefig(output_orig_png, dpi=200)
    print(f"Original bandwidth plot saved to '{output_orig_png}'.")

    # -------- Adjusted Total Bandwidth Plot --------
    fig_adj, ax_adj = plt.subplots(figsize=(15, 8))

    # If we have too many functions, adjust figure size
    if len(unique_funcs) > 15:
        fig_adj.set_figwidth(max(15, len(unique_funcs) * 0.8))

    # Create the bar chart for adjusted total bandwidth
    for i, threads in enumerate(thread_counts):
        # Filter data for this thread count
        thread_data = results_df[results_df["threads"] == threads]

        # Prepare data in the order of unique_funcs
        adj_values = []

        for func in unique_funcs:
            func_row = thread_data[thread_data["function_name"] == func]
            if len(func_row) > 0:
                bw = func_row["avg_node_bw"].values[0] if func_row["avg_node_bw"].values[0] is not None else 0.0
                adj_bw = bw - background_bw
                adj_values.append(adj_bw)
            else:
                adj_values.append(0)

        # Position for this thread count
        thread_pos = i * bar_width - bar_width * (len(thread_counts) - 1) / 2

        # Plot adjusted values
        ax_adj.bar(x + thread_pos,
                  adj_values,
                  bar_width * 0.9,
                  alpha=0.8,
                  color=thread_colors[i],
                  label=f'{threads} Threads')

    # Add labels and legend
    ax_adj.set_xlabel("Function Name", fontsize=12)
    ax_adj.set_ylabel("Adjusted Memory Bandwidth (bytes/us)", fontsize=12)
    ax_adj.set_title("Adjusted Total Node Memory Bandwidth by Function and Thread Count", fontsize=14)
    ax_adj.set_xticks(x)
    ax_adj.set_xticklabels(unique_funcs, rotation=45, ha='right', fontsize=10)

    # Create a custom legend
    handles, labels = ax_adj.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_adj.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_adj_png = "function_node_bw_adjusted.png"
    plt.savefig(output_adj_png, dpi=200)
    print(f"Adjusted bandwidth plot saved to '{output_adj_png}'.")

    # --------------------------------------------------------------------------
    # 9b. Plot PER-THREAD memory bandwidth by function and thread count (separate original and adjusted)
    # --------------------------------------------------------------------------

    # -------- Original Per-Thread Bandwidth Plot --------
    fig_pt_orig, ax_pt_orig = plt.subplots(figsize=(15, 8))

    # If we have too many functions, adjust figure size
    if len(unique_funcs) > 15:
        fig_pt_orig.set_figwidth(max(15, len(unique_funcs) * 0.8))

    # Create the bar chart for original per-thread bandwidth
    for i, threads in enumerate(thread_counts):
        # Filter data for this thread count
        thread_data = results_df[results_df["threads"] == threads]

        # Prepare data in the order of unique_funcs
        per_thread_bw_values = []

        for func in unique_funcs:
            func_row = thread_data[thread_data["function_name"] == func]
            if len(func_row) > 0:
                per_thread_bw = func_row["orig_per_thread_bw"].values[0]
                per_thread_bw_values.append(per_thread_bw)
            else:
                per_thread_bw_values.append(0)

        # Position for this thread count
        thread_pos = i * bar_width - bar_width * (len(thread_counts) - 1) / 2

        # Plot original per-thread values
        ax_pt_orig.bar(x + thread_pos,
                      per_thread_bw_values,
                      bar_width * 0.9,
                      alpha=0.8,
                      color=thread_colors[i],
                      label=f'{threads} Threads')

    # Add labels and legend
    ax_pt_orig.set_xlabel("Function Name", fontsize=12)
    ax_pt_orig.set_ylabel("Per-Thread Memory Bandwidth (bytes/us)", fontsize=12)
    ax_pt_orig.set_title("Original Per-Thread Memory Bandwidth by Function and Thread Count", fontsize=14)
    ax_pt_orig.set_xticks(x)
    ax_pt_orig.set_xticklabels(unique_funcs, rotation=45, ha='right', fontsize=10)

    # Create a custom legend
    handles, labels = ax_pt_orig.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_pt_orig.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_pt_orig_png = "function_node_bw_per_thread_original.png"
    plt.savefig(output_pt_orig_png, dpi=200)
    print(f"Original per-thread bandwidth plot saved to '{output_pt_orig_png}'.")

    # -------- Adjusted Per-Thread Bandwidth Plot --------
    fig_pt_adj, ax_pt_adj = plt.subplots(figsize=(15, 8))

    # If we have too many functions, adjust figure size
    if len(unique_funcs) > 15:
        fig_pt_adj.set_figwidth(max(15, len(unique_funcs) * 0.8))

    # Create the bar chart for adjusted per-thread bandwidth
    for i, threads in enumerate(thread_counts):
        # Filter data for this thread count
        thread_data = results_df[results_df["threads"] == threads]

        # Prepare data in the order of unique_funcs
        per_thread_adj_values = []

        for func in unique_funcs:
            func_row = thread_data[thread_data["function_name"] == func]
            if len(func_row) > 0:
                per_thread_adj_bw = func_row["adj_per_thread_bw"].values[0]
                per_thread_adj_values.append(per_thread_adj_bw)
            else:
                per_thread_adj_values.append(0)

        # Position for this thread count
        thread_pos = i * bar_width - bar_width * (len(thread_counts) - 1) / 2

        # Plot adjusted per-thread values
        ax_pt_adj.bar(x + thread_pos,
                     per_thread_adj_values,
                     bar_width * 0.9,
                     alpha=0.8,
                     color=thread_colors[i],
                     label=f'{threads} Threads')

    # Add labels and legend
    ax_pt_adj.set_xlabel("Function Name", fontsize=12)
    ax_pt_adj.set_ylabel("Adjusted Per-Thread Memory Bandwidth (bytes/us)", fontsize=12)
    ax_pt_adj.set_title("Adjusted Per-Thread Memory Bandwidth by Function and Thread Count", fontsize=14)
    ax_pt_adj.set_xticks(x)
    ax_pt_adj.set_xticklabels(unique_funcs, rotation=45, ha='right', fontsize=10)

    # Create a custom legend
    handles, labels = ax_pt_adj.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_pt_adj.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_pt_adj_png = "function_node_bw_per_thread_adjusted.png"
    plt.savefig(output_pt_adj_png, dpi=200)
    print(f"Adjusted per-thread bandwidth plot saved to '{output_pt_adj_png}'.")

    # --------------------------------------------------------------------------
    # 10. Thread scaling comparison plots (separate for original and adjusted per-thread bandwidth)
    # --------------------------------------------------------------------------

    # -------- Original Per-Thread Bandwidth Scaling Plot --------
    fig_scale_orig, ax_scale_orig = plt.subplots(figsize=(15, 8))

    # For each function, plot a line showing original per-thread bandwidth vs thread count
    for i, func in enumerate(unique_funcs):
        func_data = results_df[results_df["function_name"] == func]
        if len(func_data) > 0:
            threads = func_data["threads"].tolist()
            orig_per_thread_bw = func_data["orig_per_thread_bw"].tolist()

            # Skip functions with missing data
            if len(threads) == len(thread_counts):
                ax_scale_orig.plot(threads, orig_per_thread_bw, 'o-', linewidth=2, label=func)

    ax_scale_orig.set_xlabel("Number of Threads", fontsize=12)
    ax_scale_orig.set_ylabel("Original Per-Thread Memory Bandwidth (bytes/us)", fontsize=12)
    ax_scale_orig.set_title("Original Per-Thread Memory Bandwidth Scaling with Thread Count", fontsize=14)

    # Set x-ticks to exactly match the thread counts
    ax_scale_orig.set_xticks(thread_counts)

    # Add legend with multiple columns if needed
    if len(unique_funcs) > 10:
        ax_scale_orig.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5, fontsize=9)
    else:
        ax_scale_orig.legend(loc='best', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_scale_orig_png = "function_node_bw_per_thread_original_scaling.png"
    plt.savefig(output_scale_orig_png, dpi=200)
    print(f"Original per-thread bandwidth scaling plot saved to '{output_scale_orig_png}'.")

    # -------- Adjusted Per-Thread Bandwidth Scaling Plot --------
    fig_scale_adj, ax_scale_adj = plt.subplots(figsize=(15, 8))

    # For each function, plot a line showing adjusted per-thread bandwidth vs thread count
    for i, func in enumerate(unique_funcs):
        func_data = results_df[results_df["function_name"] == func]
        if len(func_data) > 0:
            threads = func_data["threads"].tolist()
            adj_per_thread_bw = func_data["adj_per_thread_bw"].tolist()

            # Skip functions with missing data
            if len(threads) == len(thread_counts):
                ax_scale_adj.plot(threads, adj_per_thread_bw, 'o-', linewidth=2, label=func)

    ax_scale_adj.set_xlabel("Number of Threads", fontsize=12)
    ax_scale_adj.set_ylabel("Adjusted Per-Thread Memory Bandwidth (bytes/us)", fontsize=12)
    ax_scale_adj.set_title("Adjusted Per-Thread Memory Bandwidth Scaling with Thread Count", fontsize=14)

    # Set x-ticks to exactly match the thread counts
    ax_scale_adj.set_xticks(thread_counts)

    # Add legend with multiple columns if needed
    if len(unique_funcs) > 10:
        ax_scale_adj.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5, fontsize=9)
    else:
        ax_scale_adj.legend(loc='best', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_scale_adj_png = "function_node_bw_per_thread_adjusted_scaling.png"
    plt.savefig(output_scale_adj_png, dpi=200)
    print(f"Adjusted per-thread bandwidth scaling plot saved to '{output_scale_adj_png}'.")

    # --------------------------------------------------------------------------
    # 11. Runtime analysis plots
    # --------------------------------------------------------------------------

    # Extract the runtime information from the parsed data
    runtime_data = {}
    for interval in function_intervals:
        func_name = interval['function_name']
        threads = interval['threads']
        avg_individual = interval['avg_individual']  # Use avg_individual instead of duration

        if func_name not in runtime_data:
            runtime_data[func_name] = {}

        if threads not in runtime_data[func_name]:
            runtime_data[func_name][threads] = []

        runtime_data[func_name][threads].append(avg_individual)

    # Convert to DataFrame for easier processing
    runtime_rows = []
    for func_name, thread_data in runtime_data.items():
        for threads, individual_runtimes in thread_data.items():
            avg_runtime = sum(individual_runtimes) / len(individual_runtimes)
            runtime_rows.append({
                'function_name': func_name,
                'threads': threads,
                'runtime': avg_runtime  # This is now the average of avg_individual
            })

    runtime_df = pd.DataFrame(runtime_rows)

    # Create a DataFrame with normalized runtimes
    normalized_df = pd.DataFrame()
    for func_name in runtime_df['function_name'].unique():
        func_data = runtime_df[runtime_df['function_name'] == func_name].copy()
        # Find single-threaded runtime
        single_thread_runtime = func_data[func_data['threads'] == 1]['runtime'].values[0]
        # Normalize all runtimes for this function
        func_data['normalized_runtime'] = func_data['runtime'] / single_thread_runtime
        normalized_df = pd.concat([normalized_df, func_data])

    # --------------------------------------------------------------------------
    # 11a. Raw Runtime Plot
    # --------------------------------------------------------------------------
    fig_runtime, ax_runtime = plt.subplots(figsize=(15, 8))

    # If we have too many functions, adjust figure size
    if len(unique_funcs) > 15:
        fig_runtime.set_figwidth(max(15, len(unique_funcs) * 0.8))

    # Create the bar chart for runtimes
    for i, threads in enumerate(thread_counts):
        thread_data = runtime_df[runtime_df["threads"] == threads]
        runtime_values = []

        for func in unique_funcs:
            func_row = thread_data[thread_data["function_name"] == func]
            if len(func_row) > 0:
                runtime = func_row["runtime"].values[0]
                runtime_values.append(runtime)
            else:
                runtime_values.append(0)

        # Position for this thread count
        thread_pos = i * bar_width - bar_width * (len(thread_counts) - 1) / 2

        # Plot runtime values
        ax_runtime.bar(x + thread_pos,
                       runtime_values,
                       bar_width,
                       alpha=0.8,
                       color=thread_colors[i],
                       label=f'{threads} Threads')

    # Add labels and legend
    ax_runtime.set_xlabel("Function Name", fontsize=12)
    ax_runtime.set_ylabel("Average Individual Execution Time (seconds)", fontsize=12)
    ax_runtime.set_title("Function Individual Execution Time by Thread Count", fontsize=14)
    ax_runtime.set_xticks(x)
    ax_runtime.set_xticklabels(unique_funcs, rotation=45, ha='right', fontsize=10)

    # Create a custom legend
    handles, labels = ax_runtime.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_runtime.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_runtime_png = "function_runtime_by_threads.png"
    plt.savefig(output_runtime_png, dpi=200)
    print(f"Runtime plot saved to '{output_runtime_png}'.")

    # --------------------------------------------------------------------------
    # 11b. Normalized Runtime Plot (showing thread scaling efficiency)
    # --------------------------------------------------------------------------
    fig_norm, ax_norm = plt.subplots(figsize=(15, 8))

    # For each function, plot a line showing normalized runtime vs thread count
    for i, func in enumerate(unique_funcs):
        func_data = normalized_df[normalized_df["function_name"] == func]
        if len(func_data) > 0:
            threads = func_data["threads"].tolist()
            norm_runtime = func_data["normalized_runtime"].tolist()

            # Skip functions with missing data
            if len(threads) == len(thread_counts):
                ax_norm.plot(threads, norm_runtime, 'o-', linewidth=2, label=func)

    # Add a reference line for perfect scaling (y=1)
    ax_norm.axhline(y=1.0, color='r', linestyle='--', label='Perfect Scaling')

    # Add labels and legend
    ax_norm.set_xlabel("Number of Threads", fontsize=12)
    ax_norm.set_ylabel("Normalized Runtime (relative to single thread)", fontsize=12)
    ax_norm.set_title("Thread Scaling Efficiency (lower is better)", fontsize=14)

    # Set x-ticks to exactly match the thread counts
    ax_norm.set_xticks(thread_counts)

    # Add legend with multiple columns if needed
    if len(unique_funcs) > 10:
        ax_norm.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5, fontsize=9)
    else:
        ax_norm.legend(loc='best', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_norm_png = "function_runtime_scaling.png"
    plt.savefig(output_norm_png, dpi=200)
    print(f"Normalized runtime scaling plot saved to '{output_norm_png}'.")

    con.close()

if __name__ == "__main__":
    main()
