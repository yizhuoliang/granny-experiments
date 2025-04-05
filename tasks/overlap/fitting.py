#!/usr/bin/env python3
"""
Memory Bandwidth Pressure Analysis Script

This script analyzes JSON result files from the memory pressure profiling
and fits log-scale linear regression models to characterize how each
Polybench function's performance degrades with increasing memory bandwidth pressure.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import os
from pathlib import Path
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mempress_analyzer")

def load_results(json_file):
    """
    Load results from the JSON file
    
    Args:
        json_file: Path to the JSON results file
        
    Returns:
        Dictionary of results
    """
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
            logger.info(f"Successfully loaded results from {json_file}")
            return results
    except Exception as e:
        logger.error(f"Error loading results from {json_file}: {e}")
        return None

def prepare_dataframe(results):
    """
    Convert the results dictionary to a pandas DataFrame for easier analysis
    
    Args:
        results: Dictionary of results
        
    Returns:
        DataFrame with the results
    """
    data_rows = []
    
    for func_name, func_results in results.items():
        # Get base runtime (no pressure)
        base_runtime = None
        for result in func_results:
            if result["pressure_threads"] == 0:
                base_runtime = result["runtime"]
                break
        
        if base_runtime is None or base_runtime == 0:
            logger.warning(f"No valid base runtime for {func_name}, skipping")
            continue
        
        for result in func_results:
            threads = result["pressure_threads"]
            runtime = result["runtime"]
            
            # Calculate slowdown factor
            slowdown = runtime / base_runtime if base_runtime > 0 else float('nan')
            
            # Extract memory bandwidth metrics if available
            mem_bw = None
            cpu_percent = None
            
            if "mem_bw_usage" in result and result["mem_bw_usage"]:
                mem_usage = result["mem_bw_usage"]
                if "memory_rss_mb" in mem_usage:
                    mem_bw = mem_usage["memory_rss_mb"]
                if "cpu_percent" in mem_usage:
                    cpu_percent = mem_usage["cpu_percent"]
            
            data_rows.append({
                "function": func_name,
                "pressure_threads": threads,
                "runtime": runtime,
                "slowdown": slowdown,
                "mem_bw_mb": mem_bw,
                "cpu_percent": cpu_percent
            })
    
    df = pd.DataFrame(data_rows)
    logger.info(f"Created DataFrame with {len(df)} rows")
    
    return df

def fit_log_regression(df):
    """
    Fit log-scale linear regression models to the data
    
    Args:
        df: DataFrame with the results
        
    Returns:
        DataFrame with regression results
    """
    regression_results = []
    
    # Get unique functions
    functions = df["function"].unique()
    
    for func in functions:
        func_df = df[df["function"] == func].copy()
        
        # Filter out rows where pressure_threads is 0 (no pressure baseline)
        # and rows with invalid slowdown values
        model_df = func_df[(func_df["pressure_threads"] > 0) & 
                          (func_df["slowdown"].notna()) &
                          (func_df["slowdown"] > 0)]
        
        if len(model_df) < 2:
            logger.warning(f"Not enough valid data points for {func}, skipping regression")
            continue
        
        # Use log scale for both pressure and slowdown
        X = np.log(model_df["pressure_threads"].values.reshape(-1, 1))
        y = np.log(model_df["slowdown"].values)
        
        # Fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # The model is: log(slowdown) = intercept + coefficient * log(threads)
        # This can be rewritten as: slowdown = exp(intercept) * threads^coefficient
        coefficient = model.coef_[0]
        intercept = model.intercept_
        
        # Store results
        base_runtime = func_df[func_df["pressure_threads"] == 0]["runtime"].values[0]
        
        regression_results.append({
            "function": func,
            "coefficient": coefficient,
            "intercept": intercept,
            "r2": r2,
            "rmse": rmse,
            "base_runtime": base_runtime,
            "formula": f"slowdown = {np.exp(intercept):.4f} * threads^{coefficient:.4f}"
        })
    
    # Create DataFrame
    regression_df = pd.DataFrame(regression_results)
    
    # Sort by sensitivity (coefficient)
    regression_df = regression_df.sort_values("coefficient", ascending=False)
    
    return regression_df

def plot_regression_curves(df, regression_df, output_dir):
    """
    Plot the original data and the fitted regression curves
    
    Args:
        df: DataFrame with the original data
        regression_df: DataFrame with regression results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Get unique functions
    functions = regression_df["function"].values
    
    # Create a figure for all functions
    plt.figure(figsize=(12, 8))
    
    for func in functions:
        func_df = df[df["function"] == func].copy()
        
        # Get regression parameters
        reg_row = regression_df[regression_df["function"] == func].iloc[0]
        coef = reg_row["coefficient"]
        intercept = reg_row["intercept"]
        
        # Original data points
        pressure = func_df["pressure_threads"].values
        slowdown = func_df["slowdown"].values
        
        # Filter out the base case (pressure=0) for plotting
        valid_idx = pressure > 0
        
        if np.any(valid_idx):
            # Plot the actual data points
            plt.scatter(pressure[valid_idx], slowdown[valid_idx], alpha=0.7, label=None)
            
            # Generate points for the regression curve
            x_reg = np.linspace(min(pressure[valid_idx]), max(pressure[valid_idx]), 100)
            y_reg = np.exp(intercept) * x_reg**coef
            
            # Plot the regression curve
            plt.plot(x_reg, y_reg, label=f"{func} (r²={reg_row['r2']:.2f})")
    
    # Set log scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Add labels and title
    plt.xlabel('Memory Pressure (Threads)')
    plt.ylabel('Slowdown Factor')
    plt.title('Log-Log Regression of Slowdown vs. Memory Pressure')
    
    # Add grid and legend
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "all_regressions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots for each function
    for func in functions:
        func_df = df[df["function"] == func].copy()
        
        # Get regression parameters
        reg_row = regression_df[regression_df["function"] == func].iloc[0]
        coef = reg_row["coefficient"]
        intercept = reg_row["intercept"]
        r2 = reg_row["r2"]
        formula = reg_row["formula"]
        
        # Create a figure
        plt.figure(figsize=(10, 6))
        
        # Original data points
        pressure = func_df["pressure_threads"].values
        slowdown = func_df["slowdown"].values
        
        # Filter out the base case and invalid values
        valid_idx = (pressure > 0) & (~np.isnan(slowdown)) & (slowdown > 0)
        
        if np.any(valid_idx):
            # Plot the actual data points
            plt.scatter(pressure[valid_idx], slowdown[valid_idx], 
                       color='blue', alpha=0.7, label='Measured data')
            
            # Generate points for the regression curve
            x_reg = np.linspace(min(pressure[valid_idx]), max(pressure[valid_idx]), 100)
            y_reg = np.exp(intercept) * x_reg**coef
            
            # Plot the regression curve
            plt.plot(x_reg, y_reg, 'r-', label=f'Regression model: {formula}')
            
            # Set log scales
            plt.xscale('log')
            plt.yscale('log')
            
            # Add labels and title
            plt.xlabel('Memory Pressure (Threads)')
            plt.ylabel('Slowdown Factor')
            plt.title(f'Slowdown vs. Memory Pressure: {func}')
            
            # Add R² and formula to the plot
            text = f"R² = {r2:.4f}\n{formula}"
            plt.annotate(text, xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        ha='left', va='top')
            
            # Add grid and legend
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend(loc='upper left')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(output_dir, f"{func}_regression.png"), dpi=300)
            plt.close()

def plot_sensitivity_heatmap(regression_df, output_dir):
    """
    Create a heatmap showing the sensitivity of each function to memory bandwidth pressure
    
    Args:
        regression_df: DataFrame with regression results
        output_dir: Directory to save the plots
    """
    # Create a figure
    plt.figure(figsize=(10, 12))
    
    # Sort by sensitivity coefficient
    df_sorted = regression_df.sort_values("coefficient", ascending=False)
    
    # Create a bar chart for the sensitivity coefficients
    plt.barh(df_sorted["function"], df_sorted["coefficient"], color='skyblue')
    
    # Add labels for r² values
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        plt.text(row["coefficient"] + 0.02, i, f'r²={row["r2"]:.2f}', 
                va='center', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Sensitivity Coefficient (higher means more sensitive)')
    plt.ylabel('Polybench Function')
    plt.title('Sensitivity to Memory Bandwidth Pressure (Log-Scale Regression)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "sensitivity_ranking.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a heatmap for projection of slowdown at different pressure levels
    # Define pressure levels to project
    pressure_levels = [1, 2, 4, 8, 16, 32]
    
    # Calculate projected slowdowns
    projected_data = []
    
    for _, row in df_sorted.iterrows():
        function = row["function"]
        coef = row["coefficient"]
        intercept = row["intercept"]
        
        # Calculate projected slowdowns for each pressure level
        projected_row = {"function": function}
        for pressure in pressure_levels:
            projected_slowdown = np.exp(intercept) * pressure**coef
            projected_row[f"{pressure} threads"] = projected_slowdown
        
        projected_data.append(projected_row)
    
    # Create DataFrame
    projected_df = pd.DataFrame(projected_data)
    
    # Prepare data for heatmap
    heatmap_data = projected_df.set_index("function")
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", fmt=".2f", 
               linewidths=.5, cbar_kws={'label': 'Projected Slowdown Factor'})
    
    plt.title('Projected Slowdown at Different Memory Pressure Levels')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "projected_slowdown_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(regression_df, output_dir):
    """
    Generate a summary report in markdown format
    
    Args:
        regression_df: DataFrame with regression results
        output_dir: Directory to save the report
    """
    # Calculate summary statistics
    avg_coefficient = regression_df["coefficient"].mean()
    max_coefficient = regression_df["coefficient"].max()
    min_coefficient = regression_df["coefficient"].min()
    
    most_sensitive = regression_df.loc[regression_df["coefficient"].idxmax()]
    least_sensitive = regression_df.loc[regression_df["coefficient"].idxmin()]
    
    # Create report
    report = f"""# Memory Bandwidth Pressure Analysis Report

## Summary Statistics

- **Average Sensitivity Coefficient:** {avg_coefficient:.4f}
- **Maximum Sensitivity Coefficient:** {max_coefficient:.4f} ({most_sensitive['function']})
- **Minimum Sensitivity Coefficient:** {min_coefficient:.4f} ({least_sensitive['function']})

## Regression Models

The relationship between memory bandwidth pressure and performance slowdown is modeled as:

```
slowdown = C * threads^α
```

where:
- `C` is a constant (exp(intercept))
- `α` is the sensitivity coefficient
- Higher α values indicate greater sensitivity to memory bandwidth pressure

## Function Sensitivity Ranking

| Function | Sensitivity (α) | Constant (C) | R² | Formula |
|----------|----------------|-------------|---|---------|
"""
    
    # Add row for each function
    for _, row in regression_df.iterrows():
        report += f"| {row['function']} | {row['coefficient']:.4f} | {np.exp(row['intercept']):.4f} | {row['r2']:.4f} | {row['formula']} |\n"
    
    report += """
## Interpretation

- Functions with higher sensitivity coefficients (α) degrade more rapidly as memory bandwidth pressure increases
- The R² value indicates how well the log-log model fits the data (higher is better)
- The formula can be used to predict slowdown at any pressure level

## Recommendations

Based on the sensitivity analysis:

1. The most memory-sensitive functions should be scheduled with minimal concurrent memory-intensive workloads
2. Functions with low sensitivity can be efficiently co-located with other workloads
3. Consider memory bandwidth allocation and isolation for critical functions

## Visualization

Several visualization files have been generated in the output directory:
- `all_regressions.png`: Combined plot of all regression models
- `sensitivity_ranking.png`: Bar chart ranking functions by sensitivity
- `projected_slowdown_heatmap.png`: Projected slowdowns at different pressure levels
- Individual regression plots for each function
"""
    
    # Write report to file
    report_path = os.path.join(output_dir, "analysis_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Summary report generated at {report_path}")
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Analyze memory bandwidth pressure profiling results')
    parser.add_argument('results_file', help='Path to the JSON results file')
    parser.add_argument('--output-dir', '-o', default='mempress_analysis', 
                        help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    if not results:
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare DataFrame
    df = prepare_dataframe(results)
    
    # Fit regression models
    regression_df = fit_log_regression(df)
    
    # Save regression results to CSV
    csv_path = os.path.join(args.output_dir, "regression_results.csv")
    regression_df.to_csv(csv_path, index=False)
    logger.info(f"Regression results saved to {csv_path}")
    
    # Generate plots
    plot_regression_curves(df, regression_df, args.output_dir)
    plot_sensitivity_heatmap(regression_df, args.output_dir)
    
    # Generate summary report
    report_path = generate_summary_report(regression_df, args.output_dir)
    
    logger.info(f"Analysis completed. Results saved to {args.output_dir}")
    logger.info(f"Summary report: {report_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())