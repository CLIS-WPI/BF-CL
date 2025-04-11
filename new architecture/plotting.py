# -*- coding: utf-8 -*-
"""
Plotting module for the Continual Learning Beamforming project.

This module provides functions for visualizing performance metrics,
including performance matrices, BWT vs. lambda plots, and bar charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List


def plot_performance_matrix(results_dict: Dict, metric: str = 'dot',
                            title_suffix: str = "",
                            filename: str = "perf_matrix.png") -> None:
    """
    Plots the performance matrix using seaborn heatmap.

    Args:
        results_dict: A dictionary containing the results, including
                      the performance matrix.
        metric: The metric to plot ('dot', 'sinr', or 'thrpt').
        title_suffix: A suffix to add to the plot title.
        filename: The filename to save the plot.
    """

    if plt is None or sns is None:
        print("Warning: matplotlib or seaborn not found. Plotting disabled.")
        return
    if metric not in ['dot', 'sinr', 'thrpt']:
        print(f"Error plotting: Metric '{metric}' not supported.")
        return

    num_tasks: int = len(results_dict['final_perf_dot'])  # Assuming all metrics have same number of tasks
    perf_matrix: np.ndarray = np.full((num_tasks, num_tasks), np.nan)
    perf_data_key: str = f"perf_matrix_{metric}"
    fmt: str = ".3f" if metric == 'dot' else ".1f"  # Format based on metric
    cmap: str = "viridis" if metric == 'dot' else "magma"
    cbar_label: str = f'Avg |dot|' if metric == 'dot' else \
        f'Avg {metric.upper()}'

    if perf_data_key not in results_dict:
        print(f"Error plotting: Key '{perf_data_key}' not found in results dictionary.")
        return

    perf_map: Dict = results_dict[perf_data_key]
    task_names: List[str] = [t["name"] for t in results_dict['tasks']]  # Access task names from results

    for i in range(num_tasks):  # Task Learned Up To (Row)
        for j in range(num_tasks):  # Task Evaluated On (Column)
            perf_matrix[i, j] = perf_map.get(i, {}).get(j, np.nan)

    plt.figure(figsize=(8, 6))
    sns.heatmap(perf_matrix, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=task_names,
                yticklabels=task_names,
                linewidths=.5, cbar_kws={'label': cbar_label},
                annot_kws={"size": 10})  # Adjust font size if needed
    plt.xlabel("Evaluation Task")
    plt.ylabel("Task Learned Up To")
    plt.title(f"Performance Matrix ({metric.upper()}) - {results_dict['name']}{title_suffix}")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Performance matrix plot saved to {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    plt.close()


def plot_bwt_vs_lambda(lambda_values: List[float], results_list: List[Dict],
                       metric: str = 'dot',
                       filename: str = 'bwt_vs_lambda.png') -> None:
    """
    Plots Backward Transfer (BWT) vs. lambda (EWC regularization strength).

    Args:
        lambda_values: A list of lambda values.
        results_list: A list of results dictionaries.
        metric: The metric to plot the BWT for ('dot', 'sinr', or 'thrpt').
        filename: The filename to save the plot.
    """

    if plt is None:
        print("Warning: matplotlib not found. Plotting disabled.")
        return
    plt.figure()
    y: List[float] = [r.get(f'bwt_{metric}', np.nan) for r in results_list]
    plt.plot(lambda_values, y, marker='o', label=f'BWT {metric.upper()}')
    plt.xlabel("λ (EWC Regularization Strength)")
    plt.ylabel(f"BWT {metric.upper()}")
    plt.title(f"BWT vs λ ({metric.upper()})")
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"BWT vs λ plot saved to {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    plt.close()


def plot_final_task_bars(results: Dict, filename_prefix: str = 'final_perf') -> None:
    """
    Plots bar charts of the final performance for each task.

    Args:
        results: A dictionary containing the results.
        filename_prefix: A prefix for the filenames to save the plots.
    """

    if plt is None:
        print("Warning: matplotlib not found. Plotting disabled.")
        return
    metrics: List[str] = ['final_perf_dot', 'final_perf_sinr',
                        'final_perf_thrpt', 'final_perf_lat']  # Include latency
    for metric in metrics:
        values: Dict = results[metric]
        plt.figure()
        task_names: List[str] = [t["name"] for t in results['tasks']]  # Access task names from results
        plt.bar(task_names, [values[i] for i in values])
        plt.title(f"{metric.replace('final_perf_', '').upper()} per Task")
        plt.ylabel(metric.replace('final_perf_', '').upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        try:
            plt.savefig(f"{filename_prefix}_{metric}.png")
            print(f"Bar chart plot saved to {filename_prefix}_{metric}.png")
        except Exception as e:
            print(f"Error saving plot {filename_prefix}_{metric}.png: {e}")
        plt.close()


def plot_time_vs_efficiency(results_list: List[Dict], labels: List[str],
                            filename: str = 'time_vs_efficiency.png') -> None:
    """
    Plots a scatter plot of total training time vs. operational efficiency.

    Args:
        results_list: A list of results dictionaries.
        labels: A list of labels for each result.
        filename: The filename to save the plot.
    """

    if plt is None:
        print("Warning: matplotlib not found. Plotting disabled.")
        return
    times: List[float] = [r['time'] for r in results_list]
    effs: List[float] = [
        r['avg_thrpt'] / r['avg_lat'] if r['avg_lat'] > 0 else np.nan for r in
        results_list]
    plt.figure()
    plt.scatter(times, effs)
    for i, label in enumerate(labels):
        plt.annotate(label, (times[i], effs[i]))
    plt.xlabel("Total Training Time (s)")
    plt.ylabel("Throughput / Latency")
    plt.title("Time vs. Operational Efficiency")
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(filename)
        print(f"Time vs. Efficiency plot saved to {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    plt.close()