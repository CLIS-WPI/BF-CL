# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
import numpy as np
# Import Sionna and CVNN components
try:
    import sionna as sn
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
    from sionna.phy.mimo import rzf_precoding_matrix # For baseline
except ImportError as e:
    print("Error importing Sionna. Please ensure it is installed correctly.")
    print(e)
    sys.exit()

try:
    import cvnn.layers as complex_layers # Use alias for clarity
    import cvnn.activations as complex_activations
    
except ImportError as e:
    print("Error importing cvnn. Please install it using: pip install cvnn")
    print(e)
    sys.exit()

# Ensure tf-keras is available if needed by dependencies
try:
    import tf_keras
except ImportError:
    print("Info: tf-keras package not found, assuming TF internal Keras is sufficient.")


from tqdm import tqdm
import time
import os
import sys
import random
import gc # For memory management
import pickle # To save CL state like Fisher info

# --- Basic Settings ---
NUM_ANTENNAS = 64
NUM_USERS = 6
FREQ = 28e9
NUM_SLOTS = 10 # For data generation averaging
NOISE_POWER_LIN = 1e-3

# --- Task Definitions ---
TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": [30e-9, 50e-9], "doppler": [10, 50], "channel": "TDL", "model": "A"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": [50e-9, 100e-9], "doppler": [50, 150], "channel": "Rayleigh"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": [200e-9, 500e-9], "doppler": [500, 2000], "channel": "TDL", "model": "C"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": [100e-9, 300e-9], "doppler": [200, 1000], "channel": "TDL", "model": "A"},
]
NUM_TASKS = len(TASKS)

# --- Training Configuration ---
NUM_EPOCHS_PER_TASK = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 32            # BATCH_SIZE is now the actual batch size
NUM_BATCHES_PER_EPOCH = 50
ZF_REG = 1e-5

# --- Continual Learning Configuration ---
EWC_LAMBDA = 500#1000.0        # EWC regularization strength (needs tuning)
# FISHER_DECAY = 0.99      # Not using decay, using simple accumulation/averaging

# --- Evaluation / Metrics ---
EVAL_BATCHES = 20
GPU_POWER_DRAW = 400       # Watts

# --- GPU Setup (SINGLE GPU) ---
print("--- Setting up for Single GPU Training ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Explicitly use only the first GPU (GPU:0)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ Using only GPU:0 -> {gpus[0].name}")
        NUM_DEVICES = 1
    except RuntimeError as e:
        print(f"❌ Error setting GPU visibility: {e}. Trying default setup.")
        # Fallback if setting visibility fails
        tf.config.experimental.set_memory_growth(gpus[0], True) # Still try memory growth
        NUM_DEVICES = len(tf.config.get_visible_devices('GPU'))
        print(f"✅ Using {NUM_DEVICES} visible GPU(s) with default strategy.")

else:
    print("ℹ️ No GPU found, using CPU.")
    NUM_DEVICES = 0

GLOBAL_BATCH_SIZE = BATCH_SIZE # For single GPU, global = batch size


#------------------------------------------------------
# HELPER FUNCTIONS (generate_synthetic_batch, compute_zf_weights, calculate_metrics)
# (No changes needed from the previous correct versions)
#------------------------------------------------------
def generate_synthetic_batch(task, batch_size, num_antennas, num_users, num_slots, freq):
    """ Generates a batch of channel data. Returns complex channel h [B, U, A]. """
    h_users = []
    for _ in range(num_users):
        delay = np.random.uniform(*task["delay_spread"])
        doppler = np.random.uniform(*task["doppler"])
        sampling_freq = int(max(1 / (delay + 1e-9), 2 * doppler)) * 10
        h_user = None
        if task["channel"] == "TDL":
            tdl = TDL( model=task.get("model", "A"), delay_spread=delay, carrier_frequency=freq,
                       num_tx_ant=num_antennas, num_rx_ant=1, min_speed=task["speed_range"][0],
                       max_speed=task["speed_range"][1] )
            h_time, _ = tdl(batch_size=batch_size, num_time_steps=num_slots, sampling_frequency=sampling_freq)
            h_avg_time = tf.reduce_mean(h_time, axis=-1); h_comb_paths = tf.reduce_sum(h_avg_time, axis=-1)
            h_user = tf.squeeze(h_comb_paths, axis=[1, 2])
        else: # Rayleigh
            rb = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=num_antennas)
            h_block, _ = rb(batch_size=batch_size, num_time_steps=1)
            h_user = tf.squeeze(h_block, axis=[1, 2, 3, 5])
        if h_user is None: raise ValueError("h_user not defined")
        h_user_reshaped = tf.reshape(h_user, [batch_size, 1, num_antennas])
        h_users.append(h_user_reshaped)
    h_stacked = tf.stack(h_users, axis=1)
    if tf.rank(h_stacked) == 4 and tf.shape(h_stacked)[2] == 1:
         h_stacked_squeezed = tf.squeeze(h_stacked, axis=2)
    else: h_stacked_squeezed = h_stacked
    h_norm = h_stacked_squeezed / (tf.cast(tf.norm(h_stacked_squeezed, axis=-1, keepdims=True), tf.complex64) + 1e-8)
    return h_norm

def compute_zf_weights(h, reg=1e-5):
    """ Computes normalized Zero-Forcing (ZF) weights. """
    batch_size = tf.shape(h)[0]; num_users = tf.shape(h)[1]; num_antennas = tf.shape(h)[2]
    identity = tf.eye(num_users, batch_shape=[batch_size], dtype=tf.complex64)
    try:
        h_herm = tf.linalg.adjoint(h); hh_herm = tf.matmul(h, h_herm)
        inv_term = tf.linalg.inv(hh_herm + tf.cast(reg, tf.complex64) * identity)
        w_unnormalized_intermediate = tf.matmul(h_herm, inv_term)
        w_zf_unnorm = tf.linalg.adjoint(w_unnormalized_intermediate)
        w_zf_normalized = tf.nn.l2_normalize(w_zf_unnorm, axis=-1, epsilon=1e-8)
        return tf.stop_gradient(w_zf_normalized)
    except Exception as e:
        tf.print("Warning: ZF calculation failed...", e, output_stream=sys.stderr)
        real = tf.random.normal([batch_size, num_users, num_antennas]); imag = tf.random.normal([batch_size, num_users, num_antennas])
        w_random = tf.complex(real, imag)
        return tf.stop_gradient(tf.nn.l2_normalize(w_random, axis=-1))

def calculate_metrics_for_logging(h, w_pred_norm, noise_power=1e-3):
    """ Calculates SINR (dB), Mean Alignment |dot|, Avg Throughput (bps/Hz) for logging. """
    sinr_dB_batch_mean = tf.constant(np.nan, dtype=tf.float32); mean_dot_abs = tf.constant(np.nan, dtype=tf.float32); avg_throughput = tf.constant(np.nan, dtype=tf.float32); log_lines = []
    try:
        h = tf.stop_gradient(h); w_pred_norm = tf.stop_gradient(w_pred_norm)
        with tf.device('/cpu:0'):
            h_norm = tf.nn.l2_normalize(h, axis=-1, epsilon=1e-8); complex_dots = tf.reduce_sum(h_norm * tf.math.conj(w_pred_norm), axis=-1)
            dot_abs_all = tf.abs(complex_dots); mean_dot_abs = tf.reduce_mean(dot_abs_all)
            num_users = tf.shape(h)[1]; batch_size = tf.shape(h)[0]; signal_matrix = tf.matmul(h, w_pred_norm, adjoint_b=True)
            desired_signal = tf.linalg.diag_part(signal_matrix); desired_power = tf.abs(desired_signal)**2
            mask = 1.0 - tf.eye(num_users, batch_shape=[batch_size], dtype=h.dtype.real_dtype)
            interference_power_masked = tf.abs(signal_matrix)**2 * mask
            interference_power = tf.reduce_sum(interference_power_masked, axis=-1)
            sinr_linear = desired_power / (interference_power + noise_power)
            sinr_dB_batch_mean = 10.0 * tf.math.log(tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0)
            sinr_per_user_bps = tf.math.log(1.0 + sinr_linear) / tf.math.log(2.0); avg_throughput = tf.reduce_mean(sinr_per_user_bps)
            if batch_size > 0:
                sinr_user0 = sinr_per_user_bps[0]; dot_abs_user0 = dot_abs_all[0]; complex_dots_user0 = complex_dots[0]
                for u in range(num_users):
                    complex_dot_u = complex_dots_user0[u]; dot_real = tf.math.real(complex_dot_u); dot_abs = tf.abs(complex_dot_u)
                    angle = tf.math.angle(complex_dot_u); sinr_u_val = sinr_user0[u].numpy()
                    if (dot_abs < 0.8) or (tf.abs(angle) > np.pi / 6):
                        line = f"User {u:02d} | dot={dot_real:.4f} | angle={angle:.4f} rad | |dot|={dot_abs:.4f} | SINR={sinr_u_val:.2f} bps/Hz"
                        log_lines.append(line)
    except Exception as e: tf.print("Error during metric calculation:", e, output_stream=sys.stderr); log_lines = ["Error in metrics calc."]
    return tf.identity(sinr_dB_batch_mean), tf.identity(mean_dot_abs), tf.identity(avg_throughput), log_lines


#------------------------------------------------------
# MULTI-HEAD COMPLEX-VALUED MODEL (CVNN)
# (Using standard initializer again)
#------------------------------------------------------
# --- MULTI-HEAD COMPLEX-VALUED MODEL (CVNN) ---
class MultiHeadCVNNModel(tf.keras.Model):
    # REPLACE THE __init__ METHOD
    def __init__(self, num_antennas, num_users, num_tasks):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.num_tasks = num_tasks

        # Shared Base Layers
        hidden_dim1 = 128
        hidden_dim2 = 256
        # ***** Use standard Keras initializer string AGAIN *****
        initializer = 'glorot_uniform'
        # ***** END CHANGE *****
        activation = complex_activations.cart_relu

        # Define layers using the standard initializer string
        self.dense1 = complex_layers.ComplexDense(hidden_dim1, activation=activation, kernel_initializer=initializer)
        self.dense2 = complex_layers.ComplexDense(hidden_dim2, activation=activation, kernel_initializer=initializer)

        # Task-Specific Output Heads
        self.output_heads = []
        for i in range(num_tasks):
            # No need for separate head_initializer object when using string
            head = complex_layers.ComplexDense(
                self.num_antennas,
                activation='linear',
                kernel_initializer=initializer, # Use standard initializer string
                name=f'head_task_{i}'
            )
            self.output_heads.append(head)

    # call method remains the same
    # @tf.function
    def call(self, inputs, task_idx, training=False):
        # ... (Keep the existing call method content from EWC version) ...
        x1 = self.dense1(inputs)
        x2 = self.dense2(x1)
        if not (0 <= task_idx < self.num_tasks): raise ValueError(f"Invalid task_idx: {task_idx}.")
        selected_head = self.output_heads[task_idx]
        w = selected_head(x2)
        return w # Return raw weights

#------------------------------------------------------
# EWC Related Functions
#------------------------------------------------------
#------------------------------------------------------
# EWC Related Functions
# (Corrected: Handles numpy dicts internally)
#------------------------------------------------------
def compute_ewc_loss(model, optimal_params_agg_np, fisher_info_agg_np, ewc_lambda):
    """
    Computes the EWC regularization loss term.
    Accepts aggregated optimal parameters and fisher info as numpy dictionaries.
    """
    if not optimal_params_agg_np or not fisher_info_agg_np or ewc_lambda == 0:
        return tf.constant(0.0, dtype=tf.float32)

    ewc_loss = 0.0
    for var in model.trainable_variables:
        # Look up var name in the provided numpy dictionaries
        opt_param_np = optimal_params_agg_np.get(var.name)
        fisher_val_np = fisher_info_agg_np.get(var.name)

        if opt_param_np is not None and fisher_val_np is not None:
            try:
                # Convert numpy array to tensor with the *correct dtype* of the model variable
                optimal_param = tf.cast(tf.convert_to_tensor(opt_param_np), var.dtype)
                # Convert fisher numpy array to float32 tensor
                fisher_val = tf.cast(tf.convert_to_tensor(fisher_val_np), tf.float32)

                param_diff = var - optimal_param
                # Calculate squared magnitude |theta - theta_opt|^2 correctly for complex
                sq_diff_mag = tf.square(tf.math.real(param_diff)) + tf.square(tf.math.imag(param_diff))

                # Add term: F_ii * |theta_i - theta_opt_i|^2
                ewc_loss += tf.reduce_sum(fisher_val * tf.cast(sq_diff_mag, tf.float32)) # Ensure float32 for sum

            except Exception as e:
                 # Print warning if conversion or calculation fails for a variable
                 tf.print(f"Warning: Error calculating EWC for var {var.name}. Skipping. Error: {e}", output_stream=sys.stderr)

    # Apply scaling factor (0.5 * lambda)
    return 0.5 * ewc_lambda * tf.cast(ewc_loss, dtype=tf.float32)
#------------------------------------------------------
# TRAINING STEP for CL with EWC (Single GPU version)
#------------------------------------------------------
# @tf.function # Keep commented out for debugging
#------------------------------------------------------
# TRAINING STEP for CL with EWC (Single GPU version)
# (Corrected: Accepts numpy dicts for EWC state)
#------------------------------------------------------
# @tf.function # Keep commented out for debugging
def train_step_cl_ewc_single_gpu(model, optimizer, h_batch, task_idx,
                                 optimal_params_agg_np, fisher_info_agg_np, # Changed inputs to numpy dicts
                                 ewc_lambda):
    """ Performs one training step using EWC on a single GPU. """
    w_zf_target = compute_zf_weights(h_batch, ZF_REG)

    with tf.GradientTape() as tape:
        # 1. Prediction & Current Task Loss (MSE)
        w_pred_raw = model(h_batch, task_idx=task_idx, training=True)
        w_pred_norm = tf.nn.l2_normalize(w_pred_raw, axis=-1, epsilon=1e-8)
        error_current = w_pred_norm - w_zf_target
        current_task_loss = tf.reduce_mean(tf.math.real(error_current * tf.math.conj(error_current)))

        # 2. EWC Loss (Pass numpy dicts directly)
        if task_idx > 0:
            ewc_loss_term = compute_ewc_loss(model, optimal_params_agg_np, fisher_info_agg_np, ewc_lambda)
        else:
            ewc_loss_term = tf.constant(0.0, dtype=current_task_loss.dtype)

        # 3. Total Loss
        total_loss = current_task_loss + ewc_loss_term

    # 4. Gradient Calculation and Application
    trainable_vars = model.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)
    grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
    optimizer.apply_gradients(grads_and_vars)

    # 5. Calculate metrics for logging
    sinr_db, mean_dot, avg_thrpt, log_lines = calculate_metrics_for_logging(h_batch, w_pred_norm, NOISE_POWER_LIN)

    # Return losses, metrics, and grads_and_vars for Fisher update
    return total_loss, current_task_loss, ewc_loss_term, sinr_db, mean_dot, avg_thrpt, log_lines, grads_and_vars
#------------------------------------------------------
# CONTINUAL LEARNING Training Loop (EWC, Single GPU)
#------------------------------------------------------
#------------------------------------------------------
# CONTINUAL LEARNING Training Loop (EWC, Single GPU - CORRECTED)
#------------------------------------------------------
def training_loop_cl_ewc_single_gpu(model, optimizer, tasks, num_epochs_per_task, batch_size, ewc_lambda):
    """ Trains the MultiHead CVNN model sequentially using EWC on a single GPU. """
    print(f"\n🧠 Starting Continual Learning Training Loop with EWC (Single GPU)...")
    num_tasks = len(tasks)
    log_file_summary = "summary_kpi_cl_ewc_singleGPU.log"; log_file_diag = "per_user_diag_cl_ewc_singleGPU.log"
    open(log_file_summary, "w").close(); open(log_file_diag, "w").close()

    # --- CL State Storage (using numpy arrays for saving/aggregation) ---
    optimal_params = {} # Stores optimal parameters (numpy) after each task {task_idx: {var_name: np_array}}
    fisher_information = {} # Stores accumulated Fisher info diagonal (numpy) after each task {task_idx: {var_name: np_array}}

    # --- Performance Tracking ---
    final_task_performance_dot = {}; final_task_performance_sinr = {}
    final_task_performance_thrpt = {}; final_task_comp_latency = {}
    performance_history_dot = {}; performance_history_sinr = {}; performance_history_thrpt = {}

    overall_start_time = time.time()

    for task_idx, task in enumerate(tasks):
        task_name = task['name']
        print(f"\n--- Task {task_idx + 1}/{num_tasks}: {task_name} ---")

        # Adjust LR (optional - kept constant here)
        current_lr = LEARNING_RATE
        optimizer.learning_rate.assign(current_lr)
        print(f"  Current LR: {optimizer.learning_rate.numpy():.2e}")

        task_start_time = time.time()
        # Store Fisher accumulated during THIS task (as TF tensors for efficient update)
        current_task_fisher_accum = {v.name: tf.zeros_like(v, dtype=tf.float32) for v in model.trainable_variables if v is not None}
        num_fisher_steps = 0

        # --- Aggregate EWC info from *ALL* previous tasks ---
        # These remain as NumPy dictionaries until passed to compute_ewc_loss inside train_step
        optimal_params_prev_agg = {}
        fisher_info_prev_agg = {}
        if task_idx > 0:
            print(f"  Aggregating EWC info from previous {task_idx} tasks...")
            for prev_idx in range(task_idx):
                 if prev_idx in optimal_params and prev_idx in fisher_information:
                      for var_name, param_val in optimal_params[prev_idx].items():
                           # Use optimal params from the most recent time this var was stored (no overwriting)
                           if var_name not in optimal_params_prev_agg:
                                optimal_params_prev_agg[var_name] = param_val
                           # Add Fisher information from all past tasks where this var existed
                           if var_name in fisher_information[prev_idx]:
                                fisher_info_prev_agg[var_name] = fisher_info_prev_agg.get(var_name, 0.0) + fisher_information[prev_idx][var_name]
            print(f"  Aggregated info for {len(optimal_params_prev_agg)} params.")
        else:
             print("  No previous tasks, EWC loss will be zero.")

        # --- Epoch Loop ---
        for epoch in tqdm(range(num_epochs_per_task), desc=f"Training {task_name}"):
            epoch_total_loss = 0.0; epoch_task_loss = 0.0; epoch_ewc_loss = 0.0
            epoch_sinr = 0.0; epoch_dot = 0.0; epoch_thrpt = 0.0; epoch_log_lines = []

            # --- Batch Loop ---
            for i_batch in range(NUM_BATCHES_PER_EPOCH):
                 h_batch = generate_synthetic_batch(task, batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)

                 # ***** CHANGE: Removed _tf dict creation *****
                 # optimal_params_tf = ... (DELETED)
                 # fisher_info_tf = ... (DELETED)

                 # --- Run single GPU training step ---
                 # Pass the numpy dictionaries directly to the step function
                 loss, task_loss, ewc_loss, sinr_db, mean_dot, avg_thrpt, log_lines, grads_and_vars = train_step_cl_ewc_single_gpu(
                     model, optimizer, h_batch, task_idx,
                     optimal_params_prev_agg, fisher_info_prev_agg, # Pass numpy dicts
                     ewc_lambda
                 )
                 # ***** END CHANGE *****

                 # --- Accumulate Fisher Information for CURRENT task ---
                 for g, v in grads_and_vars:
                     if g is not None and v.name in current_task_fisher_accum:
                         grad_sq_mag = tf.math.real(g * tf.math.conj(g))
                         # Accumulate sum of squared grad magnitudes
                         current_task_fisher_accum[v.name] += tf.cast(grad_sq_mag, dtype=tf.float32)
                 num_fisher_steps += 1 # Count steps where Fisher was accumulated

                 # Accumulate epoch metrics
                 epoch_total_loss += loss.numpy(); epoch_task_loss += task_loss.numpy(); epoch_ewc_loss += ewc_loss.numpy()
                 epoch_sinr += sinr_db.numpy(); epoch_dot += mean_dot.numpy(); epoch_thrpt += avg_thrpt.numpy()
                 if log_lines and (i_batch == NUM_BATCHES_PER_EPOCH - 1): epoch_log_lines = log_lines
            # --- End Batch Loop ---

            # Average metrics over batches
            avg_epoch_total_loss = epoch_total_loss / NUM_BATCHES_PER_EPOCH
            avg_epoch_task_loss = epoch_task_loss / NUM_BATCHES_PER_EPOCH
            avg_epoch_ewc_loss = epoch_ewc_loss / NUM_BATCHES_PER_EPOCH
            avg_epoch_sinr = epoch_sinr / NUM_BATCHES_PER_EPOCH
            avg_epoch_dot = epoch_dot / NUM_BATCHES_PER_EPOCH
            avg_epoch_thrpt = epoch_thrpt / NUM_BATCHES_PER_EPOCH

            # --- Logging ---
            if (epoch + 1) % 10 == 0:
                 log_str = (f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] "
                            f"Avg Loss={avg_epoch_total_loss:.6f} (T={avg_epoch_task_loss:.6f}, EWC={avg_epoch_ewc_loss:.6f}) | "
                            f"Avg SINR={avg_epoch_sinr:.2f} dB | Avg Thrpt={avg_epoch_thrpt:.4f} | Avg |dot|={avg_epoch_dot:.4f}\n")
                 print(log_str, end='')
                 try:
                     with open(log_file_summary, "a") as f: f.write(log_str)
                 except Exception as e: print(f"Error writing to {log_file_summary}: {e}")
            if (epoch + 1) % 20 == 0 and epoch_log_lines:
                 try:
                    with open(log_file_diag, "a") as f:
                        f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] ----\n")
                        for line in epoch_log_lines: f.write(line + "\n")
                 except Exception as e: print(f"Error writing to {log_file_diag}: {e}")
        # --- End Epoch Loop ---

        task_end_time = time.time()
        print(f"  Finished training Task {task_idx + 1}: {task_name} in {task_end_time - task_start_time:.2f}s")

        # --- Update EWC State AFTER Task Training ---
        print("  Storing EWC parameters...")
        optimal_params[task_idx] = {v.name: v.numpy() for v in model.trainable_variables}
        if num_fisher_steps > 0:
             # Average the accumulated Fisher info over steps
             fisher_information[task_idx] = {
                 name: accum.numpy() / float(num_fisher_steps)
                 for name, accum in current_task_fisher_accum.items() if name in optimal_params[task_idx]
             }
             print(f"  Stored Fisher information for {len(fisher_information[task_idx])} variables.")
        else:
             fisher_information[task_idx] = {}; print("  Warning: No steps processed for Fisher accumulation.")

        # --- Evaluation (Single GPU) ---
        print(f"  Evaluating performance after task {task_idx+1} (single GPU)..."); eval_batches = EVAL_BATCHES
        final_dot_sum = 0.0; final_sinr_sum = 0.0; final_thrpt_sum = 0.0; final_lat_sum = 0.0; num_eval_steps = 0
        # Eval on current task
        for _ in range(eval_batches):
            h_eval = generate_synthetic_batch(tasks[task_idx], batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
            t_start = time.time(); w_pred_eval_raw = model(h_eval, task_idx=task_idx, training=False); t_end = time.time()
            w_pred_eval_norm = tf.nn.l2_normalize(w_pred_eval_raw, axis=-1, epsilon=1e-8)
            sinr_db, mean_dot, avg_thrpt, _ = calculate_metrics_for_logging(h_eval, w_pred_eval_norm, NOISE_POWER_LIN)
            comp_lat_ms = (t_end - t_start) * 1000 / batch_size
            final_dot_sum += mean_dot.numpy(); final_sinr_sum += sinr_db.numpy(); final_thrpt_sum += avg_thrpt.numpy(); final_lat_sum += comp_lat_ms; num_eval_steps += 1
        final_task_performance_dot[task_idx] = final_dot_sum / num_eval_steps; final_task_performance_sinr[task_idx] = final_sinr_sum / num_eval_steps
        final_task_performance_thrpt[task_idx] = final_thrpt_sum / num_eval_steps; final_task_comp_latency[task_idx] = final_lat_sum / num_eval_steps
        print(f"  Performance on Task {task_idx} ({task_name}): |dot|={final_task_performance_dot[task_idx]:.4f}, SINR={final_task_performance_sinr[task_idx]:.2f} dB, Thrpt={final_task_performance_thrpt[task_idx]:.4f} bps/Hz, CompLat={final_task_comp_latency[task_idx]:.4f} ms/sample")
        # Eval on previous tasks
        performance_history_dot[task_idx] = {}; performance_history_sinr[task_idx] = {}; performance_history_thrpt[task_idx] = {}
        if task_idx > 0:
            print("  Evaluating on previous tasks...")
            for prev_task_idx in range(task_idx):
                 prev_task_name = tasks[prev_task_idx]['name']; prev_dot_sum = 0.0; prev_sinr_sum = 0.0; prev_thrpt_sum = 0.0; num_prev_eval_steps = 0
                 for _ in range(eval_batches):
                      h_eval_prev = generate_synthetic_batch(tasks[prev_task_idx], batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
                      w_pred_prev_raw = model(h_eval_prev, task_idx=prev_task_idx, training=False)
                      w_pred_prev_norm = tf.nn.l2_normalize(w_pred_prev_raw, axis=-1, epsilon=1e-8)
                      sinr_db, mean_dot, avg_thrpt, _ = calculate_metrics_for_logging(h_eval_prev, w_pred_prev_norm, NOISE_POWER_LIN)
                      prev_dot_sum += mean_dot.numpy(); prev_sinr_sum += sinr_db.numpy(); prev_thrpt_sum += avg_thrpt.numpy(); num_prev_eval_steps += 1
                 avg_prev_dot = prev_dot_sum / num_prev_eval_steps; avg_prev_sinr = prev_sinr_sum / num_prev_eval_steps; avg_prev_thrpt = prev_thrpt_sum / num_prev_eval_steps
                 performance_history_dot[task_idx][prev_task_idx] = avg_prev_dot; performance_history_sinr[task_idx][prev_task_idx] = avg_prev_sinr; performance_history_thrpt[task_idx][prev_task_idx] = avg_prev_thrpt
                 print(f"    Perf on Task {prev_task_idx} ({prev_task_name}): |dot|={avg_prev_dot:.4f}, SINR={avg_prev_sinr:.2f} dB, Thrpt={avg_prev_thrpt:.4f} bps/Hz")

        tf.keras.backend.clear_session(); gc.collect() # Clear memory

    # --- End of All Tasks ---
    overall_end_time = time.time(); total_training_time = overall_end_time - overall_start_time
    print(f"\n✅ Finished Continual Learning Training Loop (EWC Single GPU) in {total_training_time:.2f}s")

    # --- Final CL Metric Calculation and Summary (same as before) ---
    # ... (The final print and file writing block remains unchanged) ...
    avg_acc_dot = np.mean(list(final_task_performance_dot.values())); avg_sinr = np.mean(list(final_task_performance_sinr.values())); avg_thrpt = np.mean(list(final_task_performance_thrpt.values())); avg_lat = np.mean(list(final_task_comp_latency.values()))
    std_dev_dot = np.std(list(final_task_performance_dot.values())); std_dev_sinr = np.std(list(final_task_performance_sinr.values())); std_dev_thrpt = np.std(list(final_task_performance_thrpt.values()))
    bwt_dot = 0.0; bwt_sinr = 0.0; bwt_thrpt = 0.0
    if num_tasks > 1:
        bwt_terms_dot = []; bwt_terms_sinr = []; bwt_terms_thrpt = []; last_task_idx = num_tasks - 1
        for i in range(num_tasks - 1):
             if i in final_task_performance_dot and i in performance_history_dot.get(last_task_idx, {}): bwt_terms_dot.append(performance_history_dot[last_task_idx][i] - final_task_performance_dot[i])
             if i in final_task_performance_sinr and i in performance_history_sinr.get(last_task_idx, {}): bwt_terms_sinr.append(performance_history_sinr[last_task_idx][i] - final_task_performance_sinr[i])
             if i in final_task_performance_thrpt and i in performance_history_thrpt.get(last_task_idx, {}): bwt_terms_thrpt.append(performance_history_thrpt[last_task_idx][i] - final_task_performance_thrpt[i])
        if bwt_terms_dot: bwt_dot = np.mean(bwt_terms_dot)
        if bwt_terms_sinr: bwt_sinr = np.mean(bwt_terms_sinr)
        if bwt_terms_thrpt: bwt_thrpt = np.mean(bwt_terms_thrpt)
    total_training_energy_J = GPU_POWER_DRAW * total_training_time; total_training_energy_kWh = total_training_energy_J / 3600000
    op_energy_efficiency_proxy = avg_thrpt / avg_lat if avg_lat > 0 else 0
    print("\n--- CL Metrics Summary (EWC Single GPU) ---")
    print(f"  Average Final |dot| (Acc): {avg_acc_dot:.4f} (StdDev: {std_dev_dot:.4f})"); print(f"  Average Final SINR: {avg_sinr:.2f} dB (StdDev: {std_dev_sinr:.2f})"); print(f"  Average Final Throughput: {avg_thrpt:.4f} bps/Hz (StdDev: {std_dev_thrpt:.4f})")
    print(f"  Average Computational Latency: {avg_lat:.4f} ms/sample"); print(f"  Backward Transfer |dot| (BWT): {bwt_dot:.4f}"); print(f"  Backward Transfer SINR (BWT): {bwt_sinr:.2f} dB")
    print(f"  Backward Transfer Thrpt (BWT): {bwt_thrpt:.4f} bps/Hz"); print(f"  Total Training Time: {total_training_time:.2f} s"); print(f"  Estimated Training Energy: {total_training_energy_J:.2f} J ({total_training_energy_kWh:.4f} kWh)")
    print(f"  Operational Efficiency Proxy: {op_energy_efficiency_proxy:.4f} (bps/Hz)/(ms/sample)")
    try:
        with open(log_file_summary, "a") as f: # Append final summary
            f.write("\n--- CL Metrics Summary (EWC Single GPU) ---\n") # ... (write all metrics) ...
            f.write(f"  Average Final |dot| (Acc): {avg_acc_dot:.4f} (StdDev: {std_dev_dot:.4f})\n"); f.write(f"  Average Final SINR: {avg_sinr:.2f} dB (StdDev: {std_dev_sinr:.2f})\n"); f.write(f"  Average Final Throughput: {avg_thrpt:.4f} bps/Hz (StdDev: {std_dev_thrpt:.4f})\n")
            f.write(f"  Average Computational Latency: {avg_lat:.4f} ms/sample\n"); f.write(f"  Backward Transfer |dot| (BWT): {bwt_dot:.4f}\n"); f.write(f"  Backward Transfer SINR (BWT): {bwt_sinr:.2f} dB\n")
            f.write(f"  Backward Transfer Thrpt (BWT): {bwt_thrpt:.4f} bps/Hz\n"); f.write(f"  Total Training Time: {total_training_time:.2f} s\n"); f.write(f"  Estimated Training Energy: {total_training_energy_J:.2f} J ({total_training_energy_kWh:.4f} kWh)\n")
            f.write(f"  Operational Efficiency Proxy: {op_energy_efficiency_proxy:.4f} (bps/Hz)/(ms/sample)\n")
            # ... (Write performance matrices) ...
    except Exception as e: print(f"Error writing final summary: {e}")

# --- Main Execution Block (Single GPU EWC - Corrected Build Step) ---
if __name__ == "__main__":

    # --- Configuration ---
    print(f"--- Starting Continual Learning CVNN Training with EWC (Single GPU) ---")
    # Ensure these variables are defined above in the global scope
    # NUM_TASKS, NUM_EPOCHS_PER_TASK, LEARNING_RATE, BATCH_SIZE, EWC_LAMBDA
    print(f"Num Tasks: {NUM_TASKS}")
    print(f"Epochs per Task: {NUM_EPOCHS_PER_TASK}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"EWC Lambda: {EWC_LAMBDA}")
    print("-" * 36)

    GLOBAL_BATCH_SIZE = BATCH_SIZE # Use BATCH_SIZE directly for single GPU

    # --- Model and Optimizer Initialization ---
    # No strategy scope needed for single GPU
    print("Initializing MultiHead CVNN model...")
    model = MultiHeadCVNNModel(num_antennas=NUM_ANTENNAS, num_users=NUM_USERS, num_tasks=NUM_TASKS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # --- Build All Task Heads FIRST ---
    print("Building all task heads by calling the model once for each task...")
    dummy_h_complex = tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64)
    # Build all heads sequentially by calling the model
    for i in range(NUM_TASKS):
        try:
            _ = model(dummy_h_complex, task_idx=i, training=False)
            print(f"  Called model for task head {i}.")
        except Exception as e:
            print(f"❌ Error explicitly building head for task {i}: {e}")
            # Optionally exit if a head fails to build
            # import sys; sys.exit(1)
    print("Finished calling model for all task heads.")
    # --- END Build All Task Heads ---

    # --- Build Optimizer AFTER building all heads ---
    print("Building optimizer...")
    try:
        # Optimizer build should now include variables from all built heads
        optimizer.build(model.trainable_variables)
        print("✅ MultiHead CVNN Model and Optimizer built successfully.")
        model.summary() # Print summary AFTER all heads are built
    except Exception as e:
        print(f"❌ Error building optimizer: {e}")
        import traceback
        traceback.print_exc()
        import sys; sys.exit(1) # Exit if optimizer build fails

    # --- Start EWC Training Loop (Single GPU) ---
    # Ensure the single GPU training loop function is correctly named and defined above
    training_loop_cl_ewc_single_gpu( # CALL THE SINGLE GPU VERSION
        model=model, optimizer=optimizer, tasks=TASKS,
        num_epochs_per_task=NUM_EPOCHS_PER_TASK, batch_size=GLOBAL_BATCH_SIZE,
        ewc_lambda=EWC_LAMBDA
    )

    print("\n--- Continual Learning CVNN Training (EWC Single GPU) Finished ---")