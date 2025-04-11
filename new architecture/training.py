# -*- coding: utf-8 -*-
"""
Training module for the Continual Learning Beamforming project.

This module defines the training step functions and training loops
for both baseline methods (Retraining, Finetuning) and the EWC
continual learning approach. It also handles multi-GPU distribution.
"""

import tensorflow as tf
import numpy as np
from typing import Callable, List, Dict, Tuple, Optional
import time
import gc
import logging
from tqdm import tqdm
from config import NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ, NOISE_POWER_LIN, GPU_POWER_DRAW, EVAL_BATCHES, NUM_BATCHES_PER_EPOCH
from model import MultiHeadCVNNModel
from losses import compute_ewc_loss
from metrics import calculate_metrics, calculate_metrics_for_logging
from data_generation import generate_synthetic_batch
from config import NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ, NOISE_POWER_LIN, GPU_POWER_DRAW, EVAL_BATCHES


def compute_zf_weights(h: tf.Tensor, reg: float = 1e-5) -> tf.Tensor:
    """
    Computes normalized Zero-Forcing (ZF) weights.

    Args:
        h: The channel tensor of shape [batch_size, num_users, num_antennas].
        reg: The regularization term.

    Returns:
        The normalized ZF weights.
    """

    batch_size: int = tf.shape(h)[0]
    num_users: int = tf.shape(h)[1]
    num_antennas: int = tf.shape(h)[2]
    identity: tf.Tensor = tf.eye(num_users, batch_shape=[batch_size],
                                dtype=tf.complex64)
    try:
        h_herm: tf.Tensor = tf.linalg.adjoint(h)
        hh_herm: tf.Tensor = tf.matmul(h, h_herm)
        inv_term: tf.Tensor = tf.linalg.inv(
            hh_herm + tf.cast(reg, tf.complex64) * identity)
        w_unnormalized_intermediate: tf.Tensor = tf.matmul(h_herm, inv_term)
        w_zf_unnorm: tf.Tensor = tf.linalg.adjoint(w_unnormalized_intermediate)
        w_zf_normalized: tf.Tensor = tf.nn.l2_normalize(w_zf_unnorm, axis=-1,
                                                      epsilon=1e-8)
        return tf.stop_gradient(w_zf_normalized)
    except Exception as e:
        tf.print("Warning: ZF calculation failed...", e, output_stream=sys.stderr)
        real: tf.Tensor = tf.random.normal(
            [batch_size, num_users, num_antennas])
        imag: tf.Tensor = tf.random.normal(
            [batch_size, num_users, num_antennas])
        w_random: tf.Tensor = tf.complex(real, imag)
        return tf.stop_gradient(tf.nn.l2_normalize(w_random, axis=-1))


@tf.function
def train_step_baseline(model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
                        h_batch: tf.Tensor, task_idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Performs one training step for baseline methods (MSE vs ZF loss).

    Args:
        model: The TensorFlow Keras model.
        optimizer: The optimizer.
        h_batch: The channel batch.
        task_idx: The current task index.

    Returns:
        A tuple containing the loss and the raw predicted weights.
    """

    w_zf_target: tf.Tensor = compute_zf_weights(h_batch, reg=1e-5)
    with tf.GradientTape() as tape:
        w_pred_raw: tf.Tensor = model(h_batch, task_idx=task_idx, training=True)
        w_pred_norm: tf.Tensor = tf.nn.l2_normalize(w_pred_raw, axis=-1,
                                                   epsilon=1e-8)
        error: tf.Tensor = w_pred_norm - w_zf_target
        loss: tf.Tensor = tf.reduce_mean(tf.math.real(error * tf.math.conj(error)))
    trainable_vars: List[tf.Variable] = model.trainable_variables
    grads: List[tf.Tensor] = tape.gradient(loss, trainable_vars)
    grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]] = [
        (g, v) for g, v in zip(grads, trainable_vars) if g is not None]
    optimizer.apply_gradients(grads_and_vars)
    return loss, w_pred_raw


@tf.function
def train_step_cl_ewc_single_gpu(model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
                                h_batch: tf.Tensor, task_idx: int,
                                optimal_params_agg_np: Dict,
                                fisher_info_agg_np: Dict,
                                ewc_lambda: float) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor,
                                                          tf.Tensor, List[Tuple[tf.Tensor, tf.Variable]]]:
    """
    Performs one training step using EWC on a single GPU.

    Args:
        model: The TensorFlow Keras model.
        optimizer: The optimizer.
        h_batch: The channel batch.
        task_idx: The current task index.
        optimal_params_agg_np: Dictionary of optimal parameters from previous tasks.
        fisher_info_agg_np: Dictionary of Fisher information matrices from previous tasks.
        ewc_lambda: EWC regularization strength.

    Returns:
        A tuple containing: total loss, current task loss, EWC loss term,
        predicted weights, and gradients with variables.
    """

    w_zf_target: tf.Tensor = compute_zf_weights(h_batch, reg=1e-5)
    with tf.GradientTape() as tape:
        w_pred_raw: tf.Tensor = model(h_batch, task_idx=task_idx, training=True)
        w_pred_norm: tf.Tensor = tf.nn.l2_normalize(w_pred_raw, axis=-1,
                                                   epsilon=1e-8)
        error_current: tf.Tensor = w_pred_norm - w_zf_target
        current_task_loss: tf.Tensor = tf.reduce_mean(
            tf.math.real(error_current * tf.math.conj(error_current)))
        ewc_loss_term: tf.Tensor = (
            compute_ewc_loss(model, optimal_params_agg_np, fisher_info_agg_np,
                             ewc_lambda)
            if task_idx > 0 else tf.constant(0.0, dtype=current_task_loss.dtype))
        total_loss: tf.Tensor = current_task_loss + ewc_loss_term
    trainable_vars: List[tf.Variable] = model.trainable_variables
    grads: List[tf.Tensor] = tape.gradient(total_loss, trainable_vars)
    grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]] = [
        (g, v) for g, v in zip(grads, trainable_vars) if g is not None]
    optimizer.apply_gradients(grads_and_vars)
    return total_loss, current_task_loss, ewc_loss_term, w_pred_raw, grads_and_vars


@tf.function
def distributed_train_step(strategy: tf.distribute.Strategy, model: tf.keras.Model,
                           optimizer: tf.keras.optimizers.Optimizer,
                           h_batch: tf.Tensor, task_idx: int,
                           optimal_params_agg_np: Dict, fisher_info_agg_np: Dict,
                           ewc_lambda: float) -> tf.Tensor:
    """
    Distributes the training step across replicas (GPUs).

    Args:
        strategy: The TensorFlow distribution strategy.
        model: The TensorFlow Keras model.
        optimizer: The optimizer.
        h_batch: The channel batch.
        task_idx: The current task index.
        optimal_params_agg_np: Dictionary of optimal parameters from previous tasks.
        fisher_info_agg_np: Dictionary of Fisher information matrices from previous tasks.
        ewc_lambda: EWC regularization strength.

    Returns:
        The mean loss across replicas.
    """

    def _replicated_train_step(h_batch_per_replica: tf.Tensor,
                               task_idx: int) -> tf.Tensor:
        """
        Performs the training step on a single replica.
        """
        loss, _, _, _, _ = train_step_cl_ewc_single_gpu(
            model, optimizer, h_batch_per_replica, task_idx,
            optimal_params_agg_np, fisher_info_agg_np, ewc_lambda)
        return loss

    per_replica_losses: tf.Tensor = strategy.run(_replicated_train_step, args=(h_batch, task_idx))
    mean_loss: tf.Tensor = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                          per_replica_losses, axis=None)
    return mean_loss


def training_loop_baseline(baseline_name: str,
                           create_model_func: Callable[[], tf.keras.Model],
                           tasks: List[Dict], num_epochs_per_task: int,
                           batch_size: int, learning_rate: float,
                           reset_model_per_task: bool,
                           strategy: tf.distribute.Strategy) -> Dict:
    """
    Runs baseline training (Full Retraining or Finetuning). Includes full head build.

    Args:
        baseline_name: Name of the baseline method ("Retraining" or "Finetuning").
        create_model_func: Function to create a new model instance.
        tasks: List of task dictionaries.
        num_epochs_per_task: Number of epochs to train each task.
        batch_size: Batch size.
        learning_rate: Learning rate.
        reset_model_per_task: Whether to reset the model for each task (Retraining).
        strategy: TensorFlow distribution strategy.

    Returns:
        A dictionary containing the training results.
    """

    logging.info(f"\n🧠 Starting Baseline Training: {baseline_name}...")
    log_file_key: str = f"{baseline_name.upper()}_LOG_FILE"
    log_file: str = globals().get(log_file_key,
                                 f"cl_beamforming_results_{baseline_name}_details.log")
    open(log_file, "w").close()  # Clear log file

    # Performance Tracking Dictionaries
    final_task_performance_dot: Dict[int, float] = {}
    final_task_performance_sinr: Dict[int, float] = {}
    final_task_performance_thrpt: Dict[int, float] = {}
    final_task_comp_latency: Dict[int, float] = {}

    performance_history_dot: Dict[int, Dict[int, float]] = {}
    performance_history_sinr: Dict[int, Dict[int, float]] = {}
    performance_history_thrpt: Dict[int, Dict[int, float]] = {}

    overall_start_time: float = time.time()
    model: Optional[tf.keras.Model] = None
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    num_tasks: int = len(tasks)

    for task_idx, task in enumerate(tasks):
        task_name: str = task['name']
        logging.info(
            f"\n--- {baseline_name} | Task {task_idx + 1}/{num_tasks}: {task_name} ---")

        # Create new model/optimizer for each task if Retraining or first task
        if reset_model_per_task or model is None:
            logging.info("  Initializing new model and optimizer...")
            with strategy.scope():
                model = create_model_func()
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_rate)
            dummy_h: tf.Tensor = tf.zeros((1, NUM_USERS, NUM_ANTENNAS),
                                         dtype=tf.complex64)
            try:
                # ***** FIX: Build ALL heads AFTER creating model *****
                logging.info(
                    "  Building all task heads for the baseline model...")
                for i in range(len(tasks)):
                    _ = model(dummy_h, task_idx=i, training=False)
                logging.info("  Finished calling model for all task heads.")
                # ***** END FIX *****

                # Build optimizer AFTER all heads potentially exist
                optimizer.build(model.trainable_variables)
                logging.info(
                    f"  Model/Optimizer initialized/reset for Task {task_idx}.")
            except Exception as e:
                logging.error(
                    f"❌ Error building model/opt for {baseline_name} task {task_idx}: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        # Adjust LR for Finetuning on subsequent tasks (optional)
        elif not reset_model_per_task and task_idx > 0:
            logging.info(
                "  Continuing with existing model and optimizer (Finetuning).")
            current_lr: float = learning_rate  # Keep LR fixed for now
            optimizer.learning_rate.assign(current_lr)
            logging.info(
                f"  Current LR for Finetuning: {optimizer.learning_rate.numpy():.2e}")
        else:  # Finetuning first task
            optimizer.learning_rate.assign(learning_rate)

        task_start_time: float = time.time()
        # --- Epoch Loop ---
        for epoch in tqdm(range(num_epochs_per_task),
                          desc=f"{baseline_name} {task_name}", leave=False):
            epoch_loss: float = 0.0
            epoch_sinr: float = np.nan
            epoch_dot: float = np.nan
            epoch_thrpt: float = np.nan
            for i_batch in range(NUM_BATCHES_PER_EPOCH):
                h_batch: tf.Tensor = generate_synthetic_batch(task, batch_size,
                                                            NUM_ANTENNAS,
                                                            NUM_USERS, NUM_SLOTS,
                                                            FREQ)
                # Use the basic training step
                loss, w_pred_raw = strategy.run(train_step_baseline,
                                               args=(model, optimizer, h_batch,
                                                     task_idx))
                epoch_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, loss,
                                             axis=None).numpy()

                if i_batch == NUM_BATCHES_PER_EPOCH - 1:  # Metrics on last batch
                    sinr_db, mean_dot, avg_thrpt = calculate_metrics(h_batch,
                                                                    w_pred_raw,
                                                                    NOISE_POWER_LIN)
                    epoch_sinr = sinr_db
                    epoch_dot = mean_dot
                    epoch_thrpt = avg_thrpt
            avg_epoch_loss: float = epoch_loss / NUM_BATCHES_PER_EPOCH
            avg_epoch_sinr: float = epoch_sinr
            avg_epoch_dot: float = epoch_dot
            avg_epoch_thrpt: float = epoch_thrpt
            # --- Logging ---
            if (epoch + 1) % 10 == 0:
                log_str: str = (
                    f"[{baseline_name}][Task {task_idx} - {task_name}][Epoch {epoch + 1}] "
                    f"Avg Loss={avg_epoch_loss:.6f} | Last SINR={avg_epoch_sinr:.2f} dB | "
                    f"Last Thrpt={avg_epoch_thrpt:.4f} | Last |dot|={avg_epoch_dot:.4f}\n")
                logging.info(log_str)
                try:
                    with open(log_file, "a") as f:
                        f.write(log_str)
                except Exception as e:
                    logging.error(f"Error writing to {log_file}: {e}")
        # --- End Epoch Loop ---
        task_end_time: float = time.time()
        logging.info(
            f"  Finished training Task {task_idx + 1} ({baseline_name}) in {task_end_time - task_start_time:.2f}s")

        # --- Evaluation ---
        logging.info(f"  Evaluating performance...")
        final_dot_sum: float = 0.0
        final_sinr_sum: float = 0.0
        final_thrpt_sum: float = 0.0
        final_lat_sum: float = 0.0
        num_eval_steps: int = 0
        for _ in range(EVAL_BATCHES):  # Eval current task
            h_eval: tf.Tensor = generate_synthetic_batch(tasks[task_idx],
                                                        batch_size, NUM_ANTENNAS,
                                                        NUM_USERS, NUM_SLOTS,
                                                        FREQ)
            t_start: float = time.time()
            w_pred_eval_raw: tf.Tensor = model(h_eval, task_idx=task_idx,
                                              training=False)
            t_end: float = time.time()
            sinr_db, mean_dot, avg_thrpt = calculate_metrics(h_eval,
                                                            w_pred_eval_raw,
                                                            NOISE_POWER_LIN)
            comp_lat_ms: float = (t_end - t_start) * 1000 / batch_size
            if not np.isnan(mean_dot):
                final_dot_sum += mean_dot
            if not np.isnan(sinr_db):
                final_sinr_sum += sinr_db
            if not np.isnan(avg_thrpt):
                final_thrpt_sum += avg_thrpt
            if not np.isnan(comp_lat_ms):
                final_lat_sum += comp_lat_ms
            num_eval_steps += 1
        final_task_performance_dot[task_idx] = final_dot_sum / num_eval_steps if num_eval_steps > 0 else np.nan
        final_task_performance_sinr[task_idx] = final_sinr_sum / num_eval_steps if num_eval_steps > 0 else np.nan
        final_task_performance_thrpt[task_idx] = final_thrpt_sum / num_eval_steps if num_eval_steps > 0 else np.nan
        final_task_comp_latency[task_idx] = final_lat_sum / num_eval_steps if num_eval_steps > 0 else np.nan
        logging.info(
            f"  Perf Task {task_idx} ({task_name}): |dot|={final_task_performance_dot[task_idx]:.4f}, "
            f"SINR={final_task_performance_sinr[task_idx]:.2f} dB, "
            f"Thrpt={final_task_performance_thrpt[task_idx]:.4f} bps/Hz, "
            f"Lat={final_task_comp_latency[task_idx]:.4f} ms")
        performance_history_dot[task_idx] = {}
        performance_history_sinr[task_idx] = {}
        performance_history_thrpt[task_idx] = {}
        if not reset_model_per_task and task_idx > 0:  # Eval previous tasks for Finetuning
            logging.info("  Evaluating on previous tasks...")
            for prev_task_idx in range(task_idx):
                prev_task_name: str = tasks[prev_task_idx]['name']
                prev_dot_sum: float = 0.0
                prev_sinr_sum: float = 0.0
                prev_thrpt_sum: float = 0.0
                num_prev_eval_steps: int = 0
                for _ in range(EVAL_BATCHES):
                    h_eval_prev: tf.Tensor = generate_synthetic_batch(
                        tasks[prev_task_idx], batch_size, NUM_ANTENNAS,
                        NUM_USERS, NUM_SLOTS, FREQ)
                    w_pred_prev_raw: tf.Tensor = model(h_eval_prev,
                                                      task_idx=prev_task_idx,
                                                      training=False)
                    sinr_db, mean_dot, avg_thrpt = calculate_metrics(
                        h_eval_prev, w_pred_prev_raw, NOISE_POWER_LIN)
                    if not np.isnan(mean_dot):
                        prev_dot_sum += mean_dot
                    if not np.isnan(sinr_db):
                        prev_sinr_sum += sinr_db
                    if not np.isnan(avg_thrpt):
                        prev_thrpt_sum += avg_thrpt
                    num_prev_eval_steps += 1
                avg_prev_dot: float = prev_dot_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                avg_prev_sinr: float = prev_sinr_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                avg_prev_thrpt: float = prev_thrpt_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                performance_history_dot[task_idx][prev_task_idx] = avg_prev_dot
                performance_history_sinr[task_idx][prev_task_idx] = avg_prev_sinr
                performance_history_thrpt[task_idx][prev_task_idx] = avg_prev_thrpt
                logging.info(
                    f"    Perf on Task {prev_task_idx} ({prev_task_name}): |dot|={avg_prev_dot:.4f}, "
                    f"SINR={avg_prev_sinr:.2f} dB, Thrpt={avg_prev_thrpt:.4f} bps/Hz")

        tf.keras.backend.clear_session()
        gc.collect()  # Optional: Clear memory between tasks

    # --- End Task Loop ---
    overall_end_time: float = time.time()
    total_training_time: float = overall_end_time - overall_start_time
    total_training_energy_J: float = GPU_POWER_DRAW * total_training_time
    total_training_energy_kWh: float = total_training_energy_J / 3600000
    op_energy_efficiency_proxy: float = avg_thrpt / avg_lat if avg_lat > 0 and not np.isnan(
        avg_thrpt) and not np.isnan(avg_lat) else np.nan

    # --- Final Metric Calculation ---
    avg_acc_dot: float = np.nanmean(
        [final_task_performance_dot.get(i, np.nan) for i in range(num_tasks)])
    avg_sinr: float = np.nanmean(
        [final_task_performance_sinr.get(i, np.nan) for i in range(num_tasks)])
    avg_thrpt: float = np.nanmean(
        [final_task_performance_thrpt.get(i, np.nan) for i in range(num_tasks)])
    avg_lat: float = np.nanmean(
        [final_task_comp_latency.get(i, np.nan) for i in range(num_tasks)])

    std_dev_dot: float = np.nanstd(
        [final_task_performance_dot.get(i, np.nan) for i in range(num_tasks)])
    std_dev_sinr: float = np.nanstd(
        [final_task_performance_sinr.get(i, np.nan) for i in range(num_tasks)])
    std_dev_thrpt: float = np.nanstd(
        [final_task_performance_thrpt.get(i, np.nan) for i in range(num_tasks)])

    bwt_dot: float = np.nan
    bwt_sinr: float = np.nan
    bwt_thrpt: float = np.nan
    if num_tasks > 1:
        bwt_terms_dot: List[float] = []
        bwt_terms_sinr: List[float] = []
        bwt_terms_thrpt: List[float] = []
        last_task_idx: int = num_tasks - 1
        for i in range(num_tasks - 1):
            perf_i_i_dot: float = final_task_performance_dot.get(i, np.nan)
            perf_i_N_dot: float = performance_history_dot.get(last_task_idx,
                                                             {}).get(i, np.nan)
            if not np.isnan(perf_i_i_dot) and not np.isnan(perf_i_N_dot):
                bwt_terms_dot.append(perf_i_N_dot - perf_i_i_dot)

            perf_i_i_sinr: float = final_task_performance_sinr.get(i, np.nan)
            perf_i_N_sinr: float = performance_history_sinr.get(last_task_idx,
                                                               {}).get(i,
                                                                       np.nan)
            if not np.isnan(perf_i_i_sinr) and not np.isnan(perf_i_N_sinr):
                bwt_terms_sinr.append(perf_i_N_sinr - perf_i_i_sinr)

            perf_i_i_thrpt: float = final_task_performance_thrpt.get(i, np.nan)
            perf_i_N_thrpt: float = performance_history_thrpt.get(
                last_task_idx, {}).get(i, np.nan)
            if not np.isnan(perf_i_i_thrpt) and not np.isnan(perf_i_N_thrpt):
                bwt_terms_thrpt.append(perf_i_N_thrpt - perf_i_i_thrpt)

        if bwt_terms_dot:
            bwt_dot = np.mean(bwt_terms_dot)
        if bwt_terms_sinr:
            bwt_sinr = np.mean(bwt_terms_sinr)
        if bwt_terms_thrpt:
            bwt_thrpt = np.mean(bwt_terms_thrpt)

    results: Dict = {
        "name": baseline_name,
        "avg_dot": avg_acc_dot,
        "std_dot": std_dev_dot,
        "bwt_dot": bwt_dot,
        "avg_sinr": avg_sinr,
        "std_sinr": std_dev_sinr,
        "bwt_sinr": bwt_sinr,
        "avg_thrpt": avg_thrpt,
        "std_thrpt": std_dev_thrpt,
        "bwt_thrpt": bwt_thrpt,
        "avg_lat": avg_lat,
        "time": total_training_time,
        "energy_j": total_training_energy_J,
        "final_perf_dot": final_task_performance_dot,
        "final_perf_sinr": final_task_performance_sinr,
        "final_perf_thrpt": final_task_performance_thrpt,
        "final_perf_lat": final_task_comp_latency,
        "perf_matrix_dot": {
            i: performance_history_dot.get(i, {}) for i in range(num_tasks)},
        "perf_matrix_sinr": {
            i: performance_history_sinr.get(i, {}) for i in range(num_tasks)},
        "perf_matrix_thrpt": {
            i: performance_history_thrpt.get(i, {}) for i in range(num_tasks)},
        "tasks": tasks  # Store tasks for plotting
    }

    for i, perf in final_task_performance_dot.items():
        results["perf_matrix_dot"].setdefault(i, {})[i] = perf
    for i, perf in final_task_performance_sinr.items():
        results["perf_matrix_sinr"].setdefault(i, {})[i] = perf
    for i, perf in final_task_performance_thrpt.items():
        results["perf_matrix_thrpt"].setdefault(i, {})[i] = perf

    return results


def training_loop_cl_ewc_multi_gpu(create_model_func: Callable[[], tf.keras.Model],
                                   tasks: List[Dict], num_epochs_per_task: int,
                                   batch_size: int, learning_rate: float,
                                   ewc_lambda: float,
                                   strategy: tf.distribute.Strategy) -> Dict:
    """
    Continual Learning Training Loop (EWC - EMA Fisher, Multi GPU).

    Args:
        create_model_func: Function to create a new model instance.
        tasks: List of task dictionaries.
        num_epochs_per_task: Number of epochs to train each task.
        batch_size: Batch size.
        learning_rate: Learning rate.
        ewc_lambda: EWC regularization strength.
        strategy: TensorFlow distribution strategy.

    Returns:
        A dictionary containing the training results.
    """

    logging.info(
        f"\n🧠 Starting Continual Learning Training Loop with EWC (EMA Fisher, Multi GPU)...")
    log_file: str = "ewc_details.log"
    log_file_diag: str = "per_user_diag_cl_ewc_ema_multiGPU.log"
    open(log_file, "w").close()
    open(log_file_diag, "w").close()

    # --- Performance Tracking ---
    final_task_performance_dot: Dict[int, float] = {}
    final_task_performance_sinr: Dict[int, float] = {}
    final_task_performance_thrpt: Dict[int, float] = {}
    final_task_comp_latency: Dict[int, float] = {}

    performance_history_dot: Dict[int, Dict[int, float]] = {}
    performance_history_sinr: Dict[int, Dict[int, float]] = {}
    performance_history_thrpt: Dict[int, Dict[int, float]] = {}

    # --- CL State Storage ---
    optimal_params: Dict[int, Dict[str, np.ndarray]] = {}
    fisher_information: Dict[int, Dict[str, np.ndarray]] = {}

    with strategy.scope():
        model: tf.keras.Model = create_model_func()
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate)

    logging.info("  Initializing EWC model and optimizer...")
    dummy_h: tf.Tensor = tf.zeros((1, NUM_USERS, NUM_ANTENNAS),
                                 dtype=tf.complex64)
    try:
        for i in range(len(tasks)):
            _ = model(dummy_h, task_idx=i, training=False)
        optimizer.build(model.trainable_variables)
        logging.info("  EWC Model and Optimizer built successfully.")
        model.summary()
    except Exception as e:
        logging.error(f"❌ Error building EWC model/opt: {e}")
        sys.exit(1)

    overall_start_time: float = time.time()

    for task_idx, task in enumerate(tasks):
        task_name: str = task['name']
        logging.info(
            f"\n--- EWC | Task {task_idx + 1}/{len(tasks)}: {task_name} ---")
        optimizer.learning_rate.assign(learning_rate)

        task_start_time: float = time.time()

        # Initialize Fisher EMA for current task
        current_task_fisher_ema: Dict[str, tf.Variable] = {
            v.name: tf.Variable(tf.zeros_like(v, dtype=tf.float32),
                               trainable=False)
            for v in model.trainable_variables
        }

        # Aggregate Fisher/Params from previous tasks
        optimal_params_prev_agg: Dict[str, np.ndarray] = {}
        fisher_info_prev_agg: Dict[str, np.ndarray] = {}
        if task_idx > 0:
            for prev_idx in range(task_idx):
                if prev_idx in optimal_params and prev_idx in fisher_information:
                    for var_name, param_val in optimal_params[prev_idx].items():
                        if var_name not in optimal_params_prev_agg:
                            optimal_params_prev_agg[var_name] = param_val
                        fisher_info_prev_agg[var_name] = (
                            fisher_info_prev_agg.get(var_name, 0.0) +
                            fisher_information[prev_idx].get(var_name, 0.0)
                        )

        train_dataset = tf.data.Dataset.from_generator(
            lambda: generate_synthetic_batch(task, batch_size, NUM_ANTENNAS,
                                            NUM_USERS, NUM_SLOTS, FREQ),
            output_types=tf.complex64
        )
        dist_dataset: tf.data.Dataset = strategy.experimental_distribute_dataset(
            train_dataset)

        for epoch in tqdm(range(num_epochs_per_task), desc=f"EWC {task_name}",
                          leave=False):
            epoch_total_loss: float = 0.0
            epoch_task_loss: float = 0.0
            epoch_ewc_loss: float = 0.0
            epoch_sinr: float = np.nan
            epoch_dot: float = np.nan
            epoch_thrpt: float = np.nan
            epoch_log_lines: List[str] = []

            for h_batch in dist_dataset:
                mean_loss = distributed_train_step(strategy, model, optimizer,
                                                    h_batch, task_idx,
                                                    optimal_params_prev_agg,
                                                    fisher_info_prev_agg,
                                                    ewc_lambda)
                epoch_total_loss += mean_loss.numpy()

                # Update Fisher EMA
                for var_name, fisher_variable in current_task_fisher_ema.items():
                    target_var = next(
                        (v_model for v_model in model.trainable_variables if
                         v_model.name == var_name), None)
                    if target_var is not None and target_var.trainable:
                        g = optimizer.get_gradients(mean_loss, target_var)
                        if g is not None:
                            grad_sq_mag = tf.cast(
                                tf.math.real(g * tf.math.conj(g)),
                                dtype=tf.float32)
                            avg_sq_grad_mag_this_step = tf.reduce_mean(
                                grad_sq_mag)
                            new_fisher_val = (
                                fisher_ema_decay * fisher_variable) + (
                                (1.0 - fisher_ema_decay) * avg_sq_grad_mag_this_step)
                            fisher_variable.assign(new_fisher_val)

            avg_epoch_loss = epoch_total_loss / NUM_BATCHES_PER_EPOCH

            if (epoch + 1) % 10 == 0:
                log_str = (
                    f"[EWC][Task {task_idx} - {task_name}][Epoch {epoch + 1}] "
                    f"Avg Loss={avg_epoch_loss:.6f} | Last SINR={epoch_sinr:.2f} dB | "
                    f"Last Thrpt={epoch_thrpt:.4f} | Last |dot|={epoch_dot:.4f}\n"
                )
                logging.info(log_str)
                with open(log_file, "a") as f:
                    f.write(log_str)

            if (epoch + 1) % 20 == 0 and epoch_log_lines:
                with open(log_file_diag, "a") as f:
                    f.write(
                        f"[Task {task_idx} - {task_name}][Epoch {epoch + 1}] ----\n")
                    for line in epoch_log_lines:
                        f.write(line + "\n")

        task_end_time: float = time.time()
        logging.info(
            f"  Finished training Task {task_idx + 1} (EWC) in {task_end_time - task_start_time:.2f}s")

        optimal_params[task_idx] = {
            v.name: v.numpy() for v in model.trainable_variables}
        fisher_information[task_idx] = {
            name: ema_var.numpy() for name, ema_var in
            current_task_fisher_ema.items()
            if name in optimal_params[task_idx]
        }

        logging.info(
            f"  Stored optimal parameters and Fisher info (EMA) for Task {task_idx}.")

        # --- Evaluation ---
        logging.info(f"  Evaluating performance...")
        dot_sum: float = 0.0
        sinr_sum: float = 0.0
        thrpt_sum: float = 0.0
        lat_sum: float = 0.0
        eval_steps: int = 0
        for _ in range(EVAL_BATCHES):
            h_eval: tf.Tensor = generate_synthetic_batch(tasks[task_idx],
                                                        batch_size, NUM_ANTENNAS,
                                                        NUM_USERS, NUM_SLOTS,
                                                        FREQ)
            t0: float = time.time()
            w_pred: tf.Tensor = model(h_eval, task_idx=task_idx, training=False)
            t1: float = time.time()
            sinr_db, dot_val, thrpt_val = calculate_metrics(h_eval, w_pred,
                                                            NOISE_POWER_LIN)
            latency_ms: float = (t1 - t0) * 1000 / batch_size
            if not np.isnan(dot_val):
                dot_sum += dot_val
            if not np.isnan(sinr_db):
                sinr_sum += sinr_db
            if not np.isnan(thrpt_val):
                thrpt_sum += thrpt_val
            if not np.isnan(latency_ms):
                lat_sum += latency_ms
            eval_steps += 1

        final_task_performance_dot[task_idx] = dot_sum / eval_steps
        final_task_performance_sinr[task_idx] = sinr_sum / eval_steps
        final_task_performance_thrpt[task_idx] = thrpt_sum / eval_steps
        final_task_comp_latency[task_idx] = lat_sum / eval_steps
        logging.info(
            f"  Perf Task {task_idx}: |dot|={final_task_performance_dot[task_idx]:.4f}, "
            f"SINR={final_task_performance_sinr[task_idx]:.2f} dB, "
            f"Thrpt={final_task_performance_thrpt[task_idx]:.4f}, "
            f"Latency={final_task_comp_latency[task_idx]:.4f} ms")

        tf.keras.backend.clear_session()
        gc.collect()

    # --- Wrap Up ---
    overall_end_time: float = time.time()
    total_training_time: float = overall_end_time - overall_start_time
    total_energy_joules: float = GPU_POWER_DRAW * total_training_time

    # --- Final Metric Summary ---
    avg_dot: float = np.nanmean(list(final_task_performance_dot.values()))
    avg_sinr: float = np.nanmean(list(final_task_performance_sinr.values()))
    avg_thrpt: float = np.nanmean(list(final_task_performance_thrpt.values()))
    avg_lat: float = np.nanmean(list(final_task_comp_latency.values()))

    std_dev_dot: float = np.nanstd(list(final_task_performance_dot.values()))
    std_dev_sinr: float = np.nanstd(
        list(final_task_performance_sinr.values()))
    std_dev_thrpt: float = np.nanstd(
        list(final_task_performance_thrpt.values()))

    # Backward Transfer (BWT), Forward Transfer (FWT), Forgetting
    bwt_dot: float = np.nanmean([
        final_task_performance_dot[i] - final_task_performance_dot.get(
            i - 1, 0.0)
        for i in range(1, len(tasks))
    ])
    fwt_dot: float = np.nanmean([
        final_task_performance_dot.get(i, 0.0) - final_task_performance_dot.get(
            0, 0.0)
        for i in range(1, len(tasks))
    ])
    forgetting_dot: float = np.nanmean([
        max(final_task_performance_dot[j] for j in range(i + 1)) -
        final_task_performance_dot[i]
        for i in range(len(tasks) - 1)
    ])

    bwt_sinr: float = np.nanmean([
        final_task_performance_sinr[i] - final_task_performance_sinr.get(
            i - 1, 0.0)
        for i in range(1, len(tasks))
    ])
    fwt_sinr: float = np.nanmean([
        final_task_performance_sinr.get(i, 0.0) -
        final_task_performance_sinr.get(0, 0.0)
        for i in range(1, len(tasks))
    ])
    forgetting_sinr: float = np.nanmean([
        max(final_task_performance_sinr[j] for j in range(i + 1)) -
        final_task_performance_sinr[i]
        for i in range(len(tasks) - 1)
    ])

    bwt_thrpt: float = np.nanmean([
        final_task_performance_thrpt[i] - final_task_performance_thrpt.get(
            i - 1, 0.0)
        for i in range(1, len(tasks))
    ])
    fwt_thrpt: float = np.nanmean([
        final_task_performance_thrpt.get(i, 0.0) -
        final_task_performance_thrpt.get(0, 0.0)
        for i in range(1, len(tasks))
    ])
    forgetting_thrpt: float = np.nanmean([
        max(final_task_performance_thrpt[j] for j in range(i + 1)) -
        final_task_performance_thrpt[i]
        for i in range(len(tasks) - 1)
    ])

    # Add perf_matrix for plotting (diagonal-only since full eval isn't done)
    perf_matrix_dot: Dict[int, Dict[int, float]] = {
        i: {i: final_task_performance_dot[i]} for i in range(len(tasks))}
    perf_matrix_sinr: Dict[int, Dict[int, float]] = {
        i: {i: final_task_performance_sinr[i]} for i in range(len(tasks))}
    perf_matrix_thrpt: Dict[int, Dict[int, float]] = {
        i: {i: final_task_performance_thrpt[i]} for i in range(len(tasks))}

    results: Dict = {
        "name": "EWC",
        "avg_dot": avg_dot,
        "std_dot": std_dev_dot,
        "avg_sinr": avg_sinr,
        "std_sinr": std_dev_sinr,
        "avg_thrpt": avg_thrpt,
        "std_thrpt": std_dev_thrpt,
        "avg_lat": avg_lat,
        "energy_j": total_energy_joules,
        "time": total_training_time,
        "final_perf_dot": final_task_performance_dot,
        "final_perf_sinr": final_task_performance_sinr,
        "final_perf_thrpt": final_task_performance_thrpt,
        "final_perf_lat": final_task_comp_latency,
        "bwt_dot": bwt_dot,
        "fwt_dot": fwt_dot,
        "forgetting_dot": forgetting_dot,
        "bwt_sinr": bwt_sinr,
        "fwt_sinr": fwt_sinr,
        "forgetting_sinr": forgetting_sinr,
        "bwt_thrpt": bwt_thrpt,
        "fwt_thrpt": fwt_thrpt,
        "forgetting_thrpt": forgetting_thrpt,
        "perf_matrix_dot": perf_matrix_dot,
        "perf_matrix_sinr": perf_matrix_sinr,
        "perf_matrix_thrpt": perf_matrix_thrpt,
        "tasks": tasks  # Store tasks for plotting
    }

    return results