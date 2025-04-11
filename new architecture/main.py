# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import tensorflow as tf
import numpy as np
import sys
import time
import gc
import logging
from model import MultiHeadCVNNModel
# Import other project files
from config import TASKS, NUM_ANTENNAS, NUM_USERS, NUM_EPOCHS_PER_TASK, LEARNING_RATE, GLOBAL_BATCH_SIZE, EWC_LAMBDA, SUMMARY_LOG_FILE, PLOT_DOT_FILE, PLOT_SINR_FILE, PLOT_THRPT_FILE
from model import MultiHeadCVNNModel
from training import training_loop_baseline, training_loop_cl_ewc_multi_gpu
from plotting import plot_performance_matrix, plot_final_task_bars, plot_time_vs_efficiency, plot_bwt_vs_lambda

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(SUMMARY_LOG_FILE, 'w')]) # Log to file and console

def create_fresh_model(strategy):
    """Factory function for model creation within strategy scope."""
    with strategy.scope():
        logging.info("  Creating new MultiHeadCVNNModel instance...")
        return MultiHeadCVNNModel(num_antennas=NUM_ANTENNAS, num_users=NUM_USERS, num_tasks=len(TASKS))

if __name__ == "__main__":
    # --- GPU Setup ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set NCCL environment variables for stability
            os.environ["NCCL_DEBUG"] = "INFO"
            os.environ["NCCL_IB_DISABLE"] = "1"  # Disable Infiniband if not available
            os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P if issues persist

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"✅ Using GPUs: {[gpu.name for gpu in gpus]}")
            strategy = tf.distribute.MirroredStrategy()  # Use MirroredStrategy

            # Check CUDA compatibility
            for gpu in gpus:
                compute_capability = tf.config.experimental.get_device_details(gpu).get("compute_capability", (0, 0))
                compute_capability_float = float(f"{compute_capability[0]}.{compute_capability[1]}")
                if compute_capability_float < 9.0:
                    logging.warning("⚠️ TensorFlow was not built with CUDA kernel binaries "
                                    "compatible with compute capability 9.0. "
                                    "Kernels will be JIT-compiled from PTX, which may take time.")
        except RuntimeError as e:
            logging.error(f"❌ GPU setup failed: {e}")
            logging.info("ℹ️ Falling back to single GPU or CPU.")
            try:
                # Attempt to use a single GPU
                if len(gpus) > 0:
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    logging.info(f"✅ Using single GPU: {gpus[0].name}")
                    strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:0")
                else:
                    raise RuntimeError("No GPUs available.")
            except Exception as fallback_error:
                logging.error(f"❌ Single GPU fallback failed: {fallback_error}")
                logging.info("ℹ️ Using CPU as the final fallback.")
                strategy = tf.distribute.get_strategy()  # Default strategy (CPU)
    else:
        logging.info("ℹ️ No GPU found, using CPU.")
        strategy = tf.distribute.get_strategy()

    # Ensure optimizer and model are created within strategy.scope()
    with strategy.scope():
        all_results_list = []  # Store results from all runs

        # --- Run Baseline: Full Retraining ---
        logging.info("\n" + "=" * 50 + "\n  RUNNING BASELINE: Full Retraining\n" + "=" * 50)
        results_retraining = training_loop_baseline(
            baseline_name="Retraining",
            create_model_func=lambda: create_fresh_model(strategy),
            tasks=TASKS,
            num_epochs_per_task=NUM_EPOCHS_PER_TASK,
            batch_size=GLOBAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            reset_model_per_task=True,
            strategy=strategy
        )
        all_results_list.append(results_retraining)

        # --- Run Baseline: Fine-tuning ---
        logging.info("\n" + "=" * 50 + "\n  RUNNING BASELINE: Fine-tuning\n" + "=" * 50)
        results_finetuning = training_loop_baseline(
            baseline_name="Finetuning",
            create_model_func=lambda: create_fresh_model(strategy),
            tasks=TASKS,
            num_epochs_per_task=NUM_EPOCHS_PER_TASK,
            batch_size=GLOBAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            reset_model_per_task=False,
            strategy=strategy
        )
        all_results_list.append(results_finetuning)

        # --- Run CL Method: EWC (EMA Fisher) ---
        logging.info("\n" + "=" * 50 + "\n  RUNNING CL METHOD: EWC (EMA Fisher)\n" + "=" * 50)
        results_ewc = training_loop_cl_ewc_multi_gpu(
            create_model_func=lambda: create_fresh_model(strategy),
            tasks=TASKS,
            num_epochs_per_task=NUM_EPOCHS_PER_TASK,
            batch_size=GLOBAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            ewc_lambda=EWC_LAMBDA,
            strategy=strategy
        )
        all_results_list.append(results_ewc)

    # --- Final Comparative Summary ---
    logging.info("\n" + "=" * 70 + "\n--- FINAL COMPARATIVE SUMMARY ---\n" + "=" * 70)

    headers = ["Metric", "Retraining", "Finetuning", "EWC"]
    data = []

    def format_metric(val, precision=4):
        return f"{val:.{precision}f}" if not np.isnan(val) else "N/A"

    data.append(["Avg Final |dot|", format_metric(results_retraining['avg_dot']),
                 format_metric(results_finetuning['avg_dot']), format_metric(results_ewc['avg_dot'])])
    data.append(["Std Dev |dot|", format_metric(results_retraining['std_dot']),
                 format_metric(results_finetuning['std_dot']), format_metric(results_ewc['std_dot'])])
    data.append(["BWT |dot|", format_metric(results_retraining['bwt_dot']),
                 format_metric(results_finetuning['bwt_dot']), format_metric(results_ewc['bwt_dot'])])
    data.append(["-"] * 4)
    data.append(["Avg Final SINR (dB)", format_metric(results_retraining['avg_sinr'], 2),
                 format_metric(results_finetuning['avg_sinr'], 2), format_metric(results_ewc['avg_sinr'], 2)])
    data.append(["Std Dev SINR (dB)", format_metric(results_retraining['std_sinr'], 2),
                 format_metric(results_finetuning['std_sinr'], 2), format_metric(results_ewc['std_sinr'], 2)])
    data.append(["BWT SINR (dB)", format_metric(results_retraining['bwt_sinr'], 2),
                 format_metric(results_finetuning['bwt_sinr'], 2), format_metric(results_ewc['bwt_sinr'], 2)])
    data.append(["-"] * 4)
    data.append(["Avg Final Thrpt (bps/Hz)", format_metric(results_retraining['avg_thrpt'], 3),
                 format_metric(results_finetuning['avg_thrpt'], 3), format_metric(results_ewc['avg_thrpt'], 3)])
    data.append(["Std Dev Thrpt", format_metric(results_retraining['std_thrpt'], 3),
                 format_metric(results_finetuning['std_thrpt'], 3), format_metric(results_ewc['std_thrpt'], 3)])
    data.append(["BWT Thrpt", format_metric(results_retraining['bwt_thrpt'], 3),
                 format_metric(results_finetuning['bwt_thrpt'], 3), format_metric(results_ewc['bwt_thrpt'], 3)])
    data.append(["-"] * 4)
    data.append(["Avg Comp Latency (ms/sample)", format_metric(results_retraining['avg_lat'], 4),
                 format_metric(results_finetuning['avg_lat'], 4), format_metric(results_ewc['avg_lat'], 4)])
    data.append(["Total Train Time (s)", f"{results_retraining['time']:.1f}",
                 f"{results_finetuning['time']:.1f}", f"{results_ewc['time']:.1f}"])
    data.append(["Est. Train Energy (J)", f"{results_retraining['energy_j']:.0f}",
                 f"{results_finetuning['energy_j']:.0f}", f"{results_ewc['energy_j']:.0f}"])

    # Print Summary Table
    col_widths = [max(len(str(item)) for item in col) for col in zip(*([headers] + data))]
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    logging.info(header_line)
    logging.info("-" * len(header_line))
    for row in data:
        if row[0] == "-":
            logging.info("-" * len(header_line))
        else:
            logging.info(" | ".join(f"{item:<{w}}" for item, w in zip(row, col_widths)))
    logging.info("=" * 70)

    # Save summary table to main log file
    try:
        with open(SUMMARY_LOG_FILE, "w") as f:  # Overwrite summary log with final table
            f.write("=" * 70 + "\n");
            f.write("--- FINAL COMPARATIVE SUMMARY ---\n");
            f.write("=" * 70 + "\n")
            f.write(header_line + "\n");
            f.write("-" * len(header_line) + "\n")
            for row in data:
                if row[0] == "-":
                    f.write("-" * len(header_line) + "\n")
                else:
                    f.write(" | ".join(f"{item:<{w}}" for item, w in zip(row, col_widths)) + "\n")
            f.write("=" * 70 + "\n")
            f.write("\n\n--- Detailed Logs ---\n")
            f.write(f"Retraining: RT_LOG_FILE\n")
            f.write(f"Finetuning: FT_LOG_FILE\n")
            f.write(f"EWC: EWC_LOG_FILE\n")
            f.write(f"Performance Plots: {PLOT_DOT_FILE}, {PLOT_SINR_FILE}, {PLOT_THRPT_FILE}\n")
        logging.info(f"Final comparative summary saved to {SUMMARY_LOG_FILE}")
    except Exception as e:
        logging.error(f"Error writing final summary to file: {e}")

    # --- Plotting ---
    if plt and sns:
        plot_performance_matrix(results_ewc, metric='dot', title_suffix=" (EWC)",
                                filename=PLOT_DOT_FILE.replace(".png", "_ewc.png"))
        plot_performance_matrix(results_ewc, metric='sinr', title_suffix=" (EWC)",
                                filename=PLOT_SINR_FILE.replace(".png", "_ewc.png"))
        plot_performance_matrix(results_ewc, metric='thrpt', title_suffix=" (EWC)",
                                filename=PLOT_THRPT_FILE.replace(".png", "_ewc.png"))
        plot_performance_matrix(results_finetuning, metric='dot', title_suffix=" (Finetuning)",
                                filename=PLOT_DOT_FILE.replace(".png", "_ft.png"))
        plot_performance_matrix(results_finetuning, metric='sinr', title_suffix=" (Finetuning)",
                                filename=PLOT_SINR_FILE.replace(".png", "_ft.png"))
        plot_performance_matrix(results_finetuning, metric='thrpt', title_suffix=" (Finetuning)",
                                filename=PLOT_THRPT_FILE.replace(".png", "_ft.png"))
        plot_performance_matrix(results_retraining, metric='dot', title_suffix=" (Retraining)",
                                filename=PLOT_DOT_FILE.replace(".png", "_rt.png"))
        plot_performance_matrix(results_retraining, metric='sinr', title_suffix=" (Retraining)",
                                filename=PLOT_SINR_FILE.replace(".png", "_rt.png"))
        plot_performance_matrix(results_retraining, metric='thrpt', title_suffix=" (Retraining)",
                                filename=PLOT_THRPT_FILE.replace(".png", "_rt.png"))

        # Additional plots
        plot_final_task_bars(results_ewc, filename_prefix="final_ewc")
        plot_final_task_bars(results_finetuning, filename_prefix="final_ft")
        plot_final_task_bars(results_retraining, filename_prefix="final_rt")

        plot_time_vs_efficiency(
            [results_retraining, results_finetuning, results_ewc],
            labels=["Retraining", "Finetuning", "EWC"],
            filename="time_vs_efficiency.png"
        )

        # Optional: Multiple runs with different λ to plot BWT vs λ
        lambda_values = [1, 25, 50, 75, 250, 750, 2500, 10000]
        ewc_runs = []
        for lam in lambda_values:
            logging.info(f"\nRunning EWC with λ = {lam} ...")
            res = training_loop_cl_ewc_multi_gpu(
                create_model_func=lambda: create_fresh_model(strategy),  # Pass strategy to model creation
                tasks=TASKS,
                num_epochs_per_task=NUM_EPOCHS_PER_TASK,
                batch_size=GLOBAL_BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                ewc_lambda=lam,
                strategy=strategy
            )
            ewc_runs.append(res)

        plot_bwt_vs_lambda(lambda_values, ewc_runs, metric='dot', filename='bwt_vs_lambda_dot.png')
        plot_bwt_vs_lambda(lambda_values, ewc_runs, metric='sinr', filename='bwt_vs_lambda_sinr.png')
        plot_bwt_vs_lambda(lambda_values, ewc_runs, metric='thrpt', filename='bwt_vs_lambda_thrpt.png')

        logging.info("\n--- All Training, Evaluation, and Plotting Finished ---")