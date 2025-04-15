# -*- coding: utf-8 -*-
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
    import sys; sys.exit() # Ensure sys is imported

try:
    import cvnn.layers as complex_layers # Use alias for clarity
    import cvnn.activations as complex_activations
    # Using standard initializer 'glorot_uniform'
except ImportError as e:
    print("Error importing cvnn. Please install it using: pip install cvnn")
    print(e)
    import sys; sys.exit() # Ensure sys is imported

# Ensure tf-keras is available if needed by dependencies
try:
    import tf_keras
except ImportError:
    print("Info: tf-keras package not found, assuming TF internal Keras is sufficient.")

# Import other necessary libraries
from tqdm import tqdm
import time
import os
import sys # Ensure sys is imported for exit calls
import random
import gc # For memory management
import pickle # To save CL state like Fisher info
import copy # For deep copying model weights if needed

# Import plotting libraries (install if needed: pip install matplotlib seaborn)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Warning: matplotlib or seaborn not found. Plotting will be disabled.")
    print("Install using: pip install matplotlib seaborn")
    plt = None
    sns = None

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
NUM_TASKS = len(TASKS) # ***** DEFINE NUM_TASKS HERE *****

# --- Training Configuration ---
NUM_EPOCHS_PER_TASK = 1 # Keep relatively low for reasonable total runtime
LEARNING_RATE = 1e-4
BATCH_SIZE = 32            # Global batch size for single GPU
NUM_BATCHES_PER_EPOCH = 50
ZF_REG = 1e-5

# --- Continual Learning Configuration ---
# EWC specific
EWC_LAMBDA = 1000.0        # EWC regularization strength (NEEDS TUNING)
FISHER_EMA_DECAY = 0.99    # Decay factor for EMA Fisher estimate

# --- Evaluation / Metrics ---
EVAL_BATCHES = 20          # Number of batches for evaluation after each task
GPU_POWER_DRAW = 400       # Watts (Approximate for H100, adjust if known)

# --- File Names ---
LOG_FILE_BASE = "cl_beamforming_results"
SUMMARY_LOG_FILE = f"{LOG_FILE_BASE}_summary.log"
EWC_LOG_FILE = f"{LOG_FILE_BASE}_ewc_details.log"
FT_LOG_FILE = f"{LOG_FILE_BASE}_ft_details.log"
RT_LOG_FILE = f"{LOG_FILE_BASE}_rt_details.log"
PLOT_DOT_FILE = f"{LOG_FILE_BASE}_matrix_dot.png"
PLOT_SINR_FILE = f"{LOG_FILE_BASE}_matrix_sinr.png"
PLOT_THRPT_FILE = f"{LOG_FILE_BASE}_matrix_thrpt.png" # Added plot for throughput

# --- Replay Buffer for Continual Learning ---
class ReplayBuffer:
    def __init__(self, capacity_per_task=200):
        self.capacity_per_task = capacity_per_task
        self.buffer = {}  # Dictionary to store samples per task

    def add_samples(self, task_idx, h_batch, w_zf_target):
        if task_idx not in self.buffer:
            self.buffer[task_idx] = []
        batch_size = tf.shape(h_batch)[0]
        for i in range(batch_size):
            sample = (h_batch[i], w_zf_target[i])
            if len(self.buffer[task_idx]) < self.capacity_per_task:
                self.buffer[task_idx].append(sample)
            else:
                # Random replacement
                if random.random() > 0.5:
                    idx = random.randint(0, self.capacity_per_task - 1)
                    self.buffer[task_idx][idx] = sample

    def get_samples(self, task_idx, num_samples):
        if isinstance(num_samples, tf.Tensor):
            num_samples = tf.get_static_value(num_samples)
            if num_samples is None:
                raise ValueError("Cannot get number of samples from symbolic tensor.")
        samples = random.sample(self.buffer[task_idx], int(num_samples))
        h_samples, w_samples = zip(*samples)
        return np.array(h_samples), np.array(w_samples)

    
# --- GPU Setup (SINGLE GPU) ---
print("--- Setting up for Single GPU Training ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True) # Set BEFORE setting visibility
        tf.config.set_visible_devices(gpus[0], 'GPU') # Use only GPU 0
        print(f"✅ Using only GPU:0 -> {gpus[0].name}")
        NUM_DEVICES = 1
    except RuntimeError as e:
        print(f"❌ Error setting GPU visibility: {e}. Trying default setup.")
        try: # Try memory growth on default device anyway
             if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)
        except: pass
        NUM_DEVICES = len(tf.config.get_visible_devices('GPU'))
        print(f"✅ Using {NUM_DEVICES} visible GPU(s) with default strategy.")
else:
    print("ℹ️ No GPU found, using CPU.")
    NUM_DEVICES = 0

GLOBAL_BATCH_SIZE = BATCH_SIZE # For single GPU


#------------------------------------------------------
# HELPER FUNCTIONS (generate_synthetic_batch, compute_zf_weights, calculate_metrics)
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
            # No dtype argument here
            tdl = TDL( model=task.get("model", "A"), delay_spread=delay, carrier_frequency=freq,
                       num_tx_ant=num_antennas, num_rx_ant=1, min_speed=task["speed_range"][0],
                       max_speed=task["speed_range"][1] )
            h_time, _ = tdl(batch_size=batch_size, num_time_steps=num_slots, sampling_frequency=sampling_freq)
            h_avg_time = tf.reduce_mean(h_time, axis=-1); h_comb_paths = tf.reduce_sum(h_avg_time, axis=-1)
            h_user = tf.squeeze(h_comb_paths, axis=[1, 2])
        else: # Rayleigh
            # No dtype argument here
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
#------------------------------------------------------
# HELPER FUNCTION: Calculate Metrics for Logging
# (Ensure this function definition exists BEFORE training loops)
#------------------------------------------------------
def calculate_metrics_for_logging(h, w_pred_norm, noise_power=1e-3):
    """ Calculates SINR (dB), Mean Alignment |dot|, Avg Throughput (bps/Hz) for logging. """
    sinr_dB_batch_mean = tf.constant(np.nan, dtype=tf.float32)
    mean_dot_abs = tf.constant(np.nan, dtype=tf.float32)
    avg_throughput = tf.constant(np.nan, dtype=tf.float32)
    log_lines = []

    try:
        h_stop = tf.stop_gradient(tf.convert_to_tensor(h, dtype=tf.complex64))
        w_pred_norm_stop = tf.stop_gradient(tf.convert_to_tensor(w_pred_norm, dtype=tf.complex64))

        h_norm = tf.nn.l2_normalize(h_stop, axis=-1, epsilon=1e-8)
        complex_dots = tf.reduce_sum(h_norm * tf.math.conj(w_pred_norm_stop), axis=-1)  # [B, U]
        dot_abs_all = tf.abs(complex_dots)  # [B, U]
        mean_dot_abs = tf.reduce_mean(dot_abs_all)  # scalar

        num_users = tf.shape(h_stop)[1]
        batch_size = tf.shape(h_stop)[0]
        if batch_size == 0:
            raise ValueError("Batch size is zero in metric calculation")

        signal_matrix = tf.matmul(h_stop, w_pred_norm_stop, adjoint_b=True)  # [B, U, U]
        desired_signal = tf.linalg.diag_part(signal_matrix)  # [B, U]
        desired_power = tf.abs(desired_signal) ** 2  # [B, U]

        mask = 1.0 - tf.eye(num_users, batch_shape=[batch_size], dtype=h_stop.dtype.real_dtype)  # [B, U, U]
        interference_power = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)  # [B, U]

        sinr_linear = desired_power / (interference_power + noise_power)
        sinr_dB_batch_mean = 10.0 * tf.math.log(tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0)

        sinr_per_user_bps = tf.math.log(1.0 + sinr_linear) / tf.math.log(2.0)
        avg_throughput = tf.reduce_mean(sinr_per_user_bps)

        sinr_user0 = sinr_per_user_bps[0]
        dot_abs_user0 = dot_abs_all[0]
        complex_dots_user0 = complex_dots[0]
        for u in range(num_users):
            complex_dot_u = complex_dots_user0[u]
            dot_real = tf.math.real(complex_dot_u)
            dot_abs = tf.abs(complex_dot_u)
            angle = tf.math.angle(complex_dot_u)
            sinr_u_val = sinr_user0[u].numpy()  # requires CPU transfer
            if (dot_abs < 0.8) or (tf.abs(angle) > np.pi / 6):
                line = f"User {u:02d} | dot={dot_real:.4f} | angle={angle:.4f} rad | |dot|={dot_abs:.4f} | SINR={sinr_u_val:.2f} bps/Hz"
                log_lines.append(line)

    except Exception as e:
        tf.print("Error during metric calculation:", e, output_stream=sys.stderr)
        sinr_dB_batch_mean = tf.constant(np.nan, dtype=tf.float32)
        mean_dot_abs = tf.constant(np.nan, dtype=tf.float32)
        avg_throughput = tf.constant(np.nan, dtype=tf.float32)
        log_lines = ["Error in metrics calc."]

    return tf.identity(sinr_dB_batch_mean), tf.identity(mean_dot_abs), tf.identity(avg_throughput), log_lines

def calculate_metrics(h, w_pred_raw, task_idx, noise_power=1e-3):
    """ Calculates |dot|, SINR (dB), Avg Thrpt (bps/Hz) from raw prediction. Returns numpy values. """

    w_pred_norm = tf.nn.l2_normalize(w_pred_raw, axis=-1, epsilon=1e-8)
    w_pred_norm_real = tf.math.real(w_pred_norm)
    w_pred_norm_imag = tf.math.imag(w_pred_norm)
    if tf.reduce_any(tf.math.is_nan(w_pred_norm_real)) or tf.reduce_any(tf.math.is_nan(w_pred_norm_imag)):
        print(f"NaN in w_pred_norm for Task {task_idx}")
        print(f"Debug - w_pred_raw sample: {w_pred_raw[0, 0, :5]}")  # نمونه از خروجی مدل
    if tf.reduce_any(tf.math.is_nan(tf.math.real(h))) or tf.reduce_any(tf.math.is_nan(tf.math.imag(h))):
        print(f"NaN in h for Task {task_idx}")
        print(f"Debug - h sample: {h[0, 0, :5]}")  # ن

        print(f"NaN in w_pred_norm for Task {task_idx}")
    sinr_dB = np.nan; mean_dot = np.nan; avg_thrpt = np.nan
    try:
        # Use tf.stop_gradient to ensure metric calculation doesn't affect training gradients
        h_stop = tf.stop_gradient(h); w_pred_norm_stop = tf.stop_gradient(w_pred_norm)
        # Run calculations potentially outside tf.function context or on CPU
        h_norm = tf.nn.l2_normalize(h_stop, axis=-1, epsilon=1e-8)
        complex_dots = tf.reduce_sum(h_norm * tf.math.conj(w_pred_norm_stop), axis=-1)
        dot_abs_all = tf.abs(complex_dots)
        mean_dot = tf.reduce_mean(dot_abs_all).numpy() # Convert to numpy

        num_users = tf.shape(h_stop)[1]; batch_size = tf.shape(h_stop)[0]
        signal_matrix = tf.matmul(h_stop, w_pred_norm_stop, adjoint_b=True)
        desired_signal = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.abs(desired_signal)**2
        mask = 1.0 - tf.eye(num_users, batch_shape=[batch_size], dtype=h_stop.dtype.real_dtype)
        interference_power_masked = tf.abs(signal_matrix)**2 * mask
        interference_power = tf.reduce_sum(interference_power_masked, axis=-1)
        sinr_linear = desired_power / (interference_power + noise_power)
        sinr_dB = 10.0 * tf.math.log(tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0)
        sinr_dB = sinr_dB.numpy() # Convert to numpy
        sinr_per_user_bps = tf.math.log(1.0 + sinr_linear) / tf.math.log(2.0)
        avg_thrpt = tf.reduce_mean(sinr_per_user_bps).numpy() # Convert to numpy
    except Exception as e:
        print(f"Warning: Error during metric calculation for Task {task_idx}: {e}", file=sys.stderr)
    return sinr_dB, mean_dot, avg_thrpt
#------------------------------------------------------
# MULTI-HEAD COMPLEX-VALUED MODEL (CVNN)
#------------------------------------------------------
from cvnn.initializers import ComplexGlorotUniform
class MultiHeadCVNNModel(tf.keras.Model):
    def __init__(self, num_antennas, num_users, num_tasks):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.num_tasks = num_tasks

        hidden_dim1 = 128
        hidden_dim2 = 256

        initializer = ComplexGlorotUniform(seed=42)  # ✅ اصلاح initializer
        activation = complex_activations.cart_relu

        self.dense1 = complex_layers.ComplexDense(hidden_dim1, activation=activation, kernel_initializer=initializer)
        self.dense2 = complex_layers.ComplexDense(hidden_dim2, activation=activation, kernel_initializer=initializer)

        self.output_heads = []
        for i in range(num_tasks):
            head = complex_layers.ComplexDense(
                self.num_antennas,
                activation='linear',
                kernel_initializer=initializer,
                name=f'head_task_{i}'
            )
            self.output_heads.append(head)

    def call(self, inputs, task_idx, training=False):
        x1 = self.dense1(inputs)
        x2 = self.dense2(x1)

        if not (0 <= task_idx < self.num_tasks):
            raise ValueError(f"Invalid task_idx: {task_idx}.")

        selected_head = self.output_heads[task_idx]
        w = selected_head(x2)
        return w
#------------------------------------------------------
# EWC Related Function
#------------------------------------------------------
def compute_ewc_loss(model, optimal_params_agg_np, fisher_info_agg_np, ewc_lambda):
    # ... (Keep the function as before) ...
    if not optimal_params_agg_np or not fisher_info_agg_np or ewc_lambda == 0:
        return tf.constant(0.0, dtype=tf.float32)
    ewc_loss = 0.0
    for var in model.trainable_variables:
        opt_param_np = optimal_params_agg_np.get(var.name); fisher_val_np = fisher_info_agg_np.get(var.name)
        if opt_param_np is not None and fisher_val_np is not None:
            try:
                optimal_param = tf.cast(tf.convert_to_tensor(opt_param_np), var.dtype)
                fisher_val = tf.cast(tf.convert_to_tensor(fisher_val_np), tf.float32)
                param_diff = var - optimal_param
                sq_diff_mag = tf.square(tf.math.real(param_diff)) + tf.square(tf.math.imag(param_diff))
                ewc_loss += tf.reduce_sum(fisher_val * tf.cast(sq_diff_mag, tf.float32))
            except Exception as e: tf.print(f"Warning: EWC calc error {var.name}: {e}", output_stream=sys.stderr)
    return 0.5 * ewc_lambda * tf.cast(ewc_loss, dtype=tf.float32)
#------------------------------------------------------
# TRAINING STEP Functions (Baseline and EWC for Single GPU)
#------------------------------------------------------
# --- Baseline Train Step (Retraining / Finetuning) ---
@tf.function # Can add tf.function here if desired
def train_step_baseline(model, optimizer, h_batch, task_idx):
    """ Performs one training step for baseline methods (MSE vs ZF loss). """
    w_zf_target = compute_zf_weights(h_batch, ZF_REG)
    with tf.GradientTape() as tape:
        w_pred_raw = model(h_batch, task_idx=task_idx, training=True)
        w_pred_norm = tf.nn.l2_normalize(w_pred_raw, axis=-1, epsilon=1e-8)
        error = w_pred_norm - w_zf_target
        loss = tf.reduce_mean(tf.math.real(error * tf.math.conj(error)))
    trainable_vars = model.trainable_variables
    grads = tape.gradient(loss, trainable_vars)
    grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
    optimizer.apply_gradients(grads_and_vars)
    return loss, w_pred_raw # Return raw prediction for metrics

# --- EWC Train Step (Single GPU - EMA Fisher) ---
@tf.function # Can add tf.function here if desired
def train_step_cl_ewc_single_gpu(model, optimizer, h_batch, task_idx,
                                 optimal_params_agg_np, fisher_info_agg_np, # Pass numpy dicts
                                 ewc_lambda):
    """ Performs one training step using EWC on a single GPU. Returns grads. """
    w_zf_target = compute_zf_weights(h_batch, ZF_REG)
    with tf.GradientTape() as tape:
        w_pred_raw = model(h_batch, task_idx=task_idx, training=True)
        w_pred_norm = tf.nn.l2_normalize(w_pred_raw, axis=-1, epsilon=1e-8)
        error_current = w_pred_norm - w_zf_target
        current_task_loss = tf.reduce_mean(tf.math.real(error_current * tf.math.conj(error_current)))
        if task_idx > 0:
            ewc_loss_term = compute_ewc_loss(model, optimal_params_agg_np, fisher_info_agg_np, ewc_lambda)
        else: ewc_loss_term = tf.constant(0.0, dtype=current_task_loss.dtype)
        total_loss = current_task_loss + ewc_loss_term
    trainable_vars = model.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)
    grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
    optimizer.apply_gradients(grads_and_vars)
    # Return necessary info including grads for Fisher update
    return total_loss, current_task_loss, ewc_loss_term, w_pred_raw, grads_and_vars

# --- EWC + Replay Train Step (Safe) ---
# --- EWC + Replay Train Step (Safe) ---
@tf.function
def train_step_cl_ewc_replay(model, optimizer, h_batch, task_idx, replay_buffer,
                             optimal_params_agg_np, fisher_info_prev_agg,
                             ewc_lambda, replay_lambda=0.5, num_samples=8):

    w_zf_target = compute_zf_weights(h_batch, ZF_REG)

    with tf.GradientTape() as tape:
        # Forward pass
        w_pred_raw = model(h_batch, task_idx=task_idx, training=True)

        # Check for NaNs/Infs
        w_real = tf.math.real(w_pred_raw)
        w_imag = tf.math.imag(w_pred_raw)
        if (
            tf.reduce_any(tf.math.is_nan(w_real)) or tf.reduce_any(tf.math.is_nan(w_imag)) or
            tf.reduce_any(tf.math.is_inf(w_real)) or tf.reduce_any(tf.math.is_inf(w_imag))
        ):
            tf.print("⚠️ NaN or Inf in w_pred_raw for Task", task_idx)

        # Clip and normalize
        real_clip = tf.clip_by_value(w_real, -1e3, 1e3)
        imag_clip = tf.clip_by_value(w_imag, -1e3, 1e3)
        w_pred_raw = tf.complex(real_clip, imag_clip)

        norm = tf.norm(w_pred_raw, axis=-1, keepdims=True)
        norm_real = tf.math.abs(norm)
        norm_real_safe = tf.where(norm_real < 1e-8, tf.ones_like(norm_real), norm_real)
        w_pred_norm = w_pred_raw / tf.cast(norm_real_safe, tf.complex64)

        # Main loss
        error_current = w_pred_norm - w_zf_target
        current_task_loss = tf.reduce_mean(tf.math.real(error_current * tf.math.conj(error_current)))

        # EWC loss
        ewc_loss_term = tf.constant(0.0, dtype=tf.float32)
        if task_idx > 0:
            ewc_loss_term = compute_ewc_loss(model, optimal_params_agg_np, fisher_info_prev_agg, ewc_lambda)

        # Replay loss
        replay_loss = tf.constant(0.0, dtype=tf.float32)
        if task_idx > 0:
            for prev_task_idx in range(task_idx):
                h_replay, w_replay = replay_buffer.get_samples(prev_task_idx, num_samples)
                if h_replay is not None:
                    h_replay = tf.convert_to_tensor(h_replay, dtype=tf.complex64)
                    w_replay = tf.convert_to_tensor(w_replay, dtype=tf.complex64)

                    # Validate input dtype
                    if h_replay.dtype != tf.complex64 or w_replay.dtype != tf.complex64:
                        tf.print("❌ Invalid dtype in replay tensors for task", prev_task_idx)

                    w_pred_replay_raw = model(h_replay, task_idx=prev_task_idx, training=True)

                    # NaN/Inf check
                    if (
                        tf.reduce_any(tf.math.is_nan(tf.math.real(w_pred_replay_raw))) or
                        tf.reduce_any(tf.math.is_nan(tf.math.imag(w_pred_replay_raw))) or
                        tf.reduce_any(tf.math.is_inf(tf.math.real(w_pred_replay_raw))) or
                        tf.reduce_any(tf.math.is_inf(tf.math.imag(w_pred_replay_raw)))
                    ):
                        tf.print("⚠️ NaN/Inf in replay prediction for Task", prev_task_idx)

                    w_pred_replay_norm = tf.nn.l2_normalize(w_pred_replay_raw, axis=-1, epsilon=1e-8)
                    error_replay = w_pred_replay_norm - w_replay
                    replay_loss += tf.reduce_mean(tf.math.real(error_replay * tf.math.conj(error_replay)))

        total_loss = current_task_loss + ewc_loss_term + replay_lambda * replay_loss

    # Backprop and apply gradients
    trainable_vars = model.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)
    grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
    optimizer.apply_gradients(grads_and_vars)

    return total_loss, current_task_loss, ewc_loss_term, replay_loss, w_pred_raw, grads_and_vars


# --- Baseline Training Loop (Retraining / Finetuning) ---
#------------------------------------------------------
# Training Loop Functions (Baseline and EWC for Single GPU)
#-----------------------------------------------------
def training_loop_baseline(baseline_name, create_model_func, tasks, num_epochs_per_task, batch_size, learning_rate, reset_model_per_task=True):
    """ Runs baseline training (Full Retraining or Finetuning). Includes full head build. """
    print(f"\n🧠 Starting Baseline Training: {baseline_name}...")
    log_file_key = f"{baseline_name.upper()}_LOG_FILE"
    log_file = globals().get(log_file_key, f"{LOG_FILE_BASE}_{baseline_name}_details.log")
    open(log_file, "w").close(); # Clear log file

    # Performance Tracking Dictionaries
    final_task_performance_dot = {}; final_task_performance_sinr = {}
    final_task_performance_thrpt = {}; final_task_comp_latency = {}
    performance_history_dot = {}; performance_history_sinr = {}; performance_history_thrpt = {}

    overall_start_time = time.time()
    model = None; optimizer = None
    num_tasks = NUM_TASKS

    for task_idx, task in enumerate(tasks):
        task_name = task['name']
        print(f"\n--- {baseline_name} | Task {task_idx + 1}/{num_tasks}: {task_name} ---")

        # Create new model/optimizer for each task if Retraining or first task
        if reset_model_per_task or model is None:
            print("  Initializing new model and optimizer...")
            model = create_model_func()
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            dummy_h = tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64)
            try:
                # ***** FIX: Build ALL heads AFTER creating model *****
                print("  Building all task heads for the baseline model...")
                for i in range(NUM_TASKS):
                    _ = model(dummy_h, task_idx=i, training=False)
                    # print(f"    Called model for task head {i}.") # Optional print
                print("  Finished calling model for all task heads.")
                # ***** END FIX *****

                # Build optimizer AFTER all heads potentially exist
                optimizer.build(model.trainable_variables)
                print(f"  Model/Optimizer initialized/reset for Task {task_idx}.")
            except Exception as e:
                print(f"❌ Error building model/opt for {baseline_name} task {task_idx}: {e}")
                import traceback; traceback.print_exc(); sys.exit(1)
        # Adjust LR for Finetuning on subsequent tasks (optional)
        elif not reset_model_per_task and task_idx > 0:
             print("  Continuing with existing model and optimizer (Finetuning).")
             current_lr = learning_rate # Keep LR fixed for now
             optimizer.learning_rate.assign(current_lr)
             print(f"  Current LR for Finetuning: {optimizer.learning_rate.numpy():.2e}")
        else: # Finetuning first task
            optimizer.learning_rate.assign(learning_rate)


        task_start_time = time.time()
        # --- Epoch Loop ---
        for epoch in tqdm(range(num_epochs_per_task), desc=f"{baseline_name} {task_name}", leave=False):
            epoch_loss = 0.0; epoch_sinr = np.nan; epoch_dot = np.nan; epoch_thrpt = np.nan
            for i_batch in range(NUM_BATCHES_PER_EPOCH):
                h_batch = generate_synthetic_batch(task, batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
                # Use the basic training step
                loss, w_pred_raw = train_step_baseline(model, optimizer, h_batch, task_idx)
                epoch_loss += loss.numpy()
                if i_batch == NUM_BATCHES_PER_EPOCH - 1: # Metrics on last batch
                    sinr_db, mean_dot, avg_thrpt = calculate_metrics(h_batch, w_pred_raw, NOISE_POWER_LIN)
                    epoch_sinr = sinr_db; epoch_dot = mean_dot; epoch_thrpt = avg_thrpt
            avg_epoch_loss = epoch_loss / NUM_BATCHES_PER_EPOCH
            avg_epoch_sinr = epoch_sinr; avg_epoch_dot = epoch_dot; avg_epoch_thrpt = epoch_thrpt
            # --- Logging ---
            if (epoch + 1) % 10 == 0:
                 log_str = (f"[{baseline_name}][Task {task_idx} - {task_name}][Epoch {epoch+1}] "
                            f"Avg Loss={avg_epoch_loss:.6f} | Last SINR={avg_epoch_sinr:.2f} dB | "
                            f"Last Thrpt={avg_epoch_thrpt:.4f} | Last |dot|={avg_epoch_dot:.4f}\n")
                 print(log_str, end='')
                 try:
                     with open(log_file, "a") as f: f.write(log_str)
                 except Exception as e: print(f"Error writing to {log_file}: {e}")
        # --- End Epoch Loop ---
        task_end_time = time.time(); print(f"  Finished training Task {task_idx + 1} ({baseline_name}) in {task_end_time - task_start_time:.2f}s")

        # --- Evaluation ---
        print(f"  Evaluating performance...")
        # ... (Evaluation logic remains the same as the previous correct version) ...
        final_dot_sum=0.0; final_sinr_sum=0.0; final_thrpt_sum=0.0; final_lat_sum=0.0; num_eval_steps=0
        for _ in range(EVAL_BATCHES): # Eval current task
            h_eval = generate_synthetic_batch(tasks[task_idx], batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
            t_start = time.time(); w_pred_eval_raw = model(h_eval, task_idx=task_idx, training=False); t_end = time.time()
            sinr_db, mean_dot, avg_thrpt = calculate_metrics(h_eval, w_pred_eval_raw, NOISE_POWER_LIN)
            comp_lat_ms = (t_end - t_start) * 1000 / batch_size
            if not np.isnan(mean_dot): final_dot_sum += mean_dot; # ... (add other metrics similarly) ...
            if not np.isnan(sinr_db): final_sinr_sum += sinr_db
            if not np.isnan(avg_thrpt): final_thrpt_sum += avg_thrpt
            if not np.isnan(comp_lat_ms): final_lat_sum += comp_lat_ms
            num_eval_steps += 1
        final_task_performance_dot[task_idx] = final_dot_sum / num_eval_steps if num_eval_steps > 0 else np.nan
        final_task_performance_sinr[task_idx] = final_sinr_sum / num_eval_steps if num_eval_steps > 0 else np.nan
        final_task_performance_thrpt[task_idx] = final_thrpt_sum / num_eval_steps if num_eval_steps > 0 else np.nan
        final_task_comp_latency[task_idx] = final_lat_sum / num_eval_steps if num_eval_steps > 0 else np.nan
        print(f"  Perf Task {task_idx} ({task_name}): |dot|={final_task_performance_dot[task_idx]:.4f}, SINR={final_task_performance_sinr[task_idx]:.2f} dB, Thrpt={final_task_performance_thrpt[task_idx]:.4f} bps/Hz, Lat={final_task_comp_latency[task_idx]:.4f} ms")
        performance_history_dot[task_idx] = {}; performance_history_sinr[task_idx] = {}; performance_history_thrpt[task_idx] = {}
        if not reset_model_per_task and task_idx > 0: # Eval previous tasks for Finetuning
             print("  Evaluating on previous tasks...")
             for prev_task_idx in range(task_idx):
                 # ... (evaluation logic for previous tasks remains same) ...
                 prev_task_name = tasks[prev_task_idx]['name']; prev_dot_sum = 0.0; prev_sinr_sum = 0.0; prev_thrpt_sum = 0.0; num_prev_eval_steps = 0
                 for _ in range(EVAL_BATCHES):
                      h_eval_prev = generate_synthetic_batch(tasks[prev_task_idx], batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
                      w_pred_prev_raw = model(h_eval_prev, task_idx=prev_task_idx, training=False)
                      sinr_db, mean_dot, avg_thrpt = calculate_metrics(h_eval_prev, w_pred_prev_raw, NOISE_POWER_LIN)
                      if not np.isnan(mean_dot): prev_dot_sum += mean_dot
                      if not np.isnan(sinr_db): prev_sinr_sum += sinr_db
                      if not np.isnan(avg_thrpt): prev_thrpt_sum += avg_thrpt
                      num_prev_eval_steps += 1
                 avg_prev_dot = prev_dot_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                 avg_prev_sinr = prev_sinr_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                 avg_prev_thrpt = prev_thrpt_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                 performance_history_dot[task_idx][prev_task_idx] = avg_prev_dot; performance_history_sinr[task_idx][prev_task_idx] = avg_prev_sinr; performance_history_thrpt[task_idx][prev_task_idx] = avg_prev_thrpt
                 print(f"    Perf on Task {prev_task_idx} ({prev_task_name}): |dot|={avg_prev_dot:.4f}, SINR={avg_prev_sinr:.2f} dB, Thrpt={avg_prev_thrpt:.4f} bps/Hz")

        tf.keras.backend.clear_session(); gc.collect() # Optional: Clear memory between tasks

    # --- End Task Loop ---
    overall_end_time = time.time(); total_training_time = overall_end_time - overall_start_time
    print(f"\n✅ Finished EWC Training Loop in {total_training_time:.2f}s")

    # --- Final Metric Calculation (remains the same) ---
    avg_acc_dot = np.nanmean([final_task_performance_dot.get(i, np.nan) for i in range(NUM_TASKS)])
    avg_sinr = np.nanmean([final_task_performance_sinr.get(i, np.nan) for i in range(NUM_TASKS)])
    avg_thrpt = np.nanmean([final_task_performance_thrpt.get(i, np.nan) for i in range(NUM_TASKS)])
    avg_lat = np.nanmean([final_task_comp_latency.get(i, np.nan) for i in range(NUM_TASKS)])

    std_dev_dot = np.nanstd([final_task_performance_dot.get(i, np.nan) for i in range(NUM_TASKS)])
    std_dev_sinr = np.nanstd([final_task_performance_sinr.get(i, np.nan) for i in range(NUM_TASKS)])
    std_dev_thrpt = np.nanstd([final_task_performance_thrpt.get(i, np.nan) for i in range(NUM_TASKS)])

    bwt_dot = np.nan; bwt_sinr = np.nan; bwt_thrpt = np.nan
    if NUM_TASKS > 1:
        bwt_terms_dot = []; bwt_terms_sinr = []; bwt_terms_thrpt = []
        last_task_idx = NUM_TASKS - 1
        for i in range(NUM_TASKS - 1):
            perf_i_i_dot = final_task_performance_dot.get(i, np.nan)
            perf_i_N_dot = performance_history_dot.get(last_task_idx, {}).get(i, np.nan)
            if not np.isnan(perf_i_i_dot) and not np.isnan(perf_i_N_dot):
                bwt_terms_dot.append(perf_i_N_dot - perf_i_i_dot)

            perf_i_i_sinr = final_task_performance_sinr.get(i, np.nan)
            perf_i_N_sinr = performance_history_sinr.get(last_task_idx, {}).get(i, np.nan)
            if not np.isnan(perf_i_i_sinr) and not np.isnan(perf_i_N_sinr):
                bwt_terms_sinr.append(perf_i_N_sinr - perf_i_i_sinr)

            perf_i_i_thrpt = final_task_performance_thrpt.get(i, np.nan)
            perf_i_N_thrpt = performance_history_thrpt.get(last_task_idx, {}).get(i, np.nan)
            if not np.isnan(perf_i_i_thrpt) and not np.isnan(perf_i_N_thrpt):
                bwt_terms_thrpt.append(perf_i_N_thrpt - perf_i_i_thrpt)

        if bwt_terms_dot: bwt_dot = np.mean(bwt_terms_dot)
        if bwt_terms_sinr: bwt_sinr = np.mean(bwt_terms_sinr)
        if bwt_terms_thrpt: bwt_thrpt = np.mean(bwt_terms_thrpt)

    total_training_energy_J = GPU_POWER_DRAW * total_training_time
    total_training_energy_kWh = total_training_energy_J / 3600000
    op_energy_efficiency_proxy = avg_thrpt / avg_lat if avg_lat > 0 and not np.isnan(avg_thrpt) and not np.isnan(avg_lat) else np.nan

    results = {
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
        "final_perf_thrpt": final_task_performance_thrpt,  # ✅ اضافه شد
        "final_perf_lat": final_task_comp_latency,         # ✅ اضافه شد
        "perf_matrix_dot": {i: performance_history_dot.get(i, {}) for i in range(NUM_TASKS)},
        "perf_matrix_sinr": {i: performance_history_sinr.get(i, {}) for i in range(NUM_TASKS)},
        "perf_matrix_thrpt": {i: performance_history_thrpt.get(i, {}) for i in range(NUM_TASKS)},  # ✅ اضافه شد
    }

    for i, perf in final_task_performance_dot.items():
        results["perf_matrix_dot"].setdefault(i, {})[i] = perf
    for i, perf in final_task_performance_sinr.items():
        results["perf_matrix_sinr"].setdefault(i, {})[i] = perf
    for i, perf in final_task_performance_thrpt.items():  # ✅ اضافه شد
        results["perf_matrix_thrpt"].setdefault(i, {})[i] = perf

    return results

# --- EWC Training Loop (Single GPU - EMA Fisher) ---
def training_loop_cl_ewc_single_gpu(create_model_func, tasks, num_epochs_per_task, batch_size, learning_rate, ewc_lambda):
    print(f"\n🧠 Starting Continual Learning Training Loop with EWC (EMA Fisher, Single GPU)...")
    log_file = EWC_LOG_FILE
    log_file_diag = "per_user_diag_cl_ewc_singleGPU.log"
    open(log_file, "w").close()
    open(log_file_diag, "w").close()

    # Performance Tracking
    final_task_performance_dot = {}
    final_task_performance_sinr = {}
    final_task_performance_thrpt = {}
    final_task_comp_latency = {}
    performance_history_dot = {}
    performance_history_sinr = {}
    performance_history_thrpt = {}

    # CL State Storage
    optimal_params = {}
    fisher_information = {}

    model = create_model_func()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    print("  Initializing EWC model and optimizer...")
    dummy_h = tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64)
    try:
        for i in range(NUM_TASKS):
            _ = model(dummy_h, task_idx=i, training=False)
        optimizer.build(model.trainable_variables)
        print("  EWC Model and Optimizer built successfully.")
        model.summary()
    except Exception as e:
        print(f"❌ Error building EWC model/opt: {e}")
        sys.exit(1)

    overall_start_time = time.time()

    for task_idx, task in enumerate(tasks):
        task_name = task['name']
        print(f"\n--- EWC | Task {task_idx + 1}/{len(tasks)}: {task_name} ---")
        optimizer.learning_rate.assign(learning_rate)

        task_start_time = time.time()

        # Initialize Fisher EMA for current task
        current_task_fisher_ema = {
            v.name: tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
            for v in model.trainable_variables
        }

        # Aggregate Fisher/Params from previous tasks
        optimal_params_prev_agg = {}
        fisher_info_prev_agg = {}
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

        for epoch in tqdm(range(num_epochs_per_task), desc=f"EWC {task_name}", leave=False):
            epoch_total_loss = 0.0
            epoch_task_loss = 0.0
            epoch_ewc_loss = 0.0
            epoch_sinr = np.nan
            epoch_dot = np.nan
            epoch_thrpt = np.nan
            epoch_log_lines = []

            for i_batch in range(NUM_BATCHES_PER_EPOCH):
                h_batch = generate_synthetic_batch(task, batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
                w_zf_target = compute_zf_weights(h_batch, ZF_REG)

                # Training step
                total_loss, task_loss, ewc_loss_term, w_pred_raw, grads_and_vars = train_step_cl_ewc_single_gpu(
                    model, optimizer, h_batch, task_idx,
                    optimal_params_prev_agg, fisher_info_prev_agg, ewc_lambda
                )

                # Update Fisher EMA
                var_to_grad = {v.ref(): g for g, v in grads_and_vars}
                for var_name, fisher_variable in current_task_fisher_ema.items():
                    target_var = next((v_model for v_model in model.trainable_variables if v_model.name == var_name), None)
                    if target_var is not None:
                        g = var_to_grad.get(target_var.ref())
                        if g is not None:
                            grad_sq_mag = tf.cast(tf.math.real(g * tf.math.conj(g)), dtype=tf.float32)
                            avg_sq_grad_mag_this_step = tf.reduce_mean(grad_sq_mag)
                            new_fisher_val = (FISHER_EMA_DECAY * fisher_variable) + \
                                             ((1.0 - FISHER_EMA_DECAY) * avg_sq_grad_mag_this_step)
                            fisher_variable.assign(new_fisher_val)

                epoch_total_loss += total_loss.numpy()
                epoch_task_loss += task_loss.numpy()
                epoch_ewc_loss += ewc_loss_term.numpy()

                if i_batch == NUM_BATCHES_PER_EPOCH - 1:
                    sinr_db, mean_dot, avg_thrpt, log_lines = calculate_metrics_for_logging(h_batch, w_pred_raw)
                    epoch_sinr = sinr_db.numpy()
                    epoch_dot = mean_dot.numpy()
                    epoch_thrpt = avg_thrpt.numpy()
                    epoch_log_lines = log_lines
                if np.isnan(epoch_sinr) or np.isnan(epoch_dot) or np.isnan(epoch_thrpt):
                    print(f"Warning: NaN detected in metrics for Task {task_idx}, Epoch {epoch+1}")

            avg_epoch_loss = epoch_total_loss / NUM_BATCHES_PER_EPOCH

            if (epoch + 1) % 10 == 0:
                log_str = (
                    f"[EWC][Task {task_idx} - {task_name}][Epoch {epoch+1}] "
                    f"Avg Loss={avg_epoch_loss:.6f} | Task Loss={epoch_task_loss/NUM_BATCHES_PER_EPOCH:.6f} | "
                    f"EWC Loss={epoch_ewc_loss/NUM_BATCHES_PER_EPOCH:.6f} | "
                    f"Last SINR={epoch_sinr:.2f} dB | Last Thrpt={epoch_thrpt:.4f} | Last |dot|={epoch_dot:.4f}\n"
                )
                print(log_str, end='')
                with open(log_file, "a") as f:
                    f.write(log_str)

            if (epoch + 1) % 20 == 0 and epoch_log_lines:
                with open(log_file_diag, "a") as f:
                    f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] ----\n")
                    for line in epoch_log_lines:
                        f.write(line + "\n")

        task_end_time = time.time()
        print(f"  Finished training Task {task_idx + 1} (EWC) in {task_end_time - task_start_time:.2f}s")

        # Store optimal parameters and Fisher info
        optimal_params[task_idx] = {v.name: v.numpy() for v in model.trainable_variables}
        fisher_information[task_idx] = {
            name: ema_var.numpy() for name, ema_var in current_task_fisher_ema.items()
            if name in optimal_params[task_idx]
        }
        print(f"  Stored optimal parameters and Fisher info (EMA) for Task {task_idx}.")

        # Evaluation
        print(f"  Evaluating performance...")
        dot_sum = 0.0
        sinr_sum = 0.0
        thrpt_sum = 0.0
        lat_sum = 0.0
        eval_steps = 0
        for _ in range(EVAL_BATCHES):
            h_eval = generate_synthetic_batch(tasks[task_idx], batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
            t0 = time.time()
            w_pred = model(h_eval, task_idx=task_idx, training=False)
            t1 = time.time()
            sinr_db, dot_val, thrpt_val = calculate_metrics(h_eval, w_pred, task_idx)  # اصلاح شده
            latency_ms = (t1 - t0) * 1000 / batch_size
            if not np.isnan(dot_val): dot_sum += dot_val
            if not np.isnan(sinr_db): sinr_sum += sinr_db
            if not np.isnan(thrpt_val): thrpt_sum += thrpt_val
            if not np.isnan(latency_ms): lat_sum += latency_ms
            eval_steps += 1

        final_task_performance_dot[task_idx] = dot_sum / eval_steps
        final_task_performance_sinr[task_idx] = sinr_sum / eval_steps
        final_task_performance_thrpt[task_idx] = thrpt_sum / eval_steps
        final_task_comp_latency[task_idx] = lat_sum / eval_steps
        print(f"  Perf Task {task_idx}: |dot|={final_task_performance_dot[task_idx]:.4f}, "
              f"SINR={final_task_performance_sinr[task_idx]:.2f} dB, "
              f"Thrpt={final_task_performance_thrpt[task_idx]:.4f}, "
              f"Latency={final_task_comp_latency[task_idx]:.4f} ms")

        # Evaluate previous tasks for performance history
        if task_idx > 0:
            print("  Evaluating on previous tasks...")
            performance_history_dot[task_idx] = {}
            performance_history_sinr[task_idx] = {}
            performance_history_thrpt[task_idx] = {}
            for prev_task_idx in range(task_idx):
                prev_task_name = tasks[prev_task_idx]['name']
                prev_dot_sum = 0.0
                prev_sinr_sum = 0.0
                prev_thrpt_sum = 0.0
                num_prev_eval_steps = 0
                for _ in range(EVAL_BATCHES):
                    h_eval_prev = generate_synthetic_batch(tasks[prev_task_idx], batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
                    w_pred_prev = model(h_eval_prev, task_idx=prev_task_idx, training=False)
                    sinr_db, mean_dot, avg_thrpt = calculate_metrics(h_eval_prev, w_pred_prev, prev_task_idx)  # اصلاح شده
                    if not np.isnan(mean_dot): prev_dot_sum += mean_dot
                    if not np.isnan(sinr_db): prev_sinr_sum += sinr_db
                    if not np.isnan(avg_thrpt): prev_thrpt_sum += avg_thrpt
                    num_prev_eval_steps += 1
                avg_prev_dot = prev_dot_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                avg_prev_sinr = prev_sinr_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                avg_prev_thrpt = prev_thrpt_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                performance_history_dot[task_idx][prev_task_idx] = avg_prev_dot
                performance_history_sinr[task_idx][prev_task_idx] = avg_prev_sinr
                performance_history_thrpt[task_idx][prev_task_idx] = avg_prev_thrpt
                print(f"    Perf on Task {prev_task_idx} ({prev_task_name}): |dot|={avg_prev_dot:.4f}, SINR={avg_prev_sinr:.2f} dB, Thrpt={avg_prev_thrpt:.4f} bps/Hz")

        tf.keras.backend.clear_session()
        gc.collect()

    # Wrap Up
    overall_end_time = time.time()
    total_training_time = overall_end_time - overall_start_time
    total_energy_joules = GPU_POWER_DRAW * total_training_time

    # Final Metric Summary
    avg_dot = np.nanmean(list(final_task_performance_dot.values()))
    avg_sinr = np.nanmean(list(final_task_performance_sinr.values()))
    avg_thrpt = np.nanmean(list(final_task_performance_thrpt.values()))
    avg_lat = np.nanmean(list(final_task_comp_latency.values()))

    std_dev_dot = np.nanstd(list(final_task_performance_dot.values()))
    std_dev_sinr = np.nanstd(list(final_task_performance_sinr.values()))
    std_dev_thrpt = np.nanstd(list(final_task_performance_thrpt.values()))

    # Backward Transfer (BWT)
    bwt_dot = np.nan
    bwt_sinr = np.nan
    bwt_thrpt = np.nan
    if len(tasks) > 1:
        bwt_terms_dot = []
        bwt_terms_sinr = []
        bwt_terms_thrpt = []
        last_task_idx = len(tasks) - 1
        for i in range(last_task_idx):
            perf_i_i_dot = final_task_performance_dot.get(i, np.nan)
            perf_i_N_dot = performance_history_dot.get(last_task_idx, {}).get(i, np.nan)
            if not np.isnan(perf_i_i_dot) and not np.isnan(perf_i_N_dot):
                bwt_terms_dot.append(perf_i_N_dot - perf_i_i_dot)

            perf_i_i_sinr = final_task_performance_sinr.get(i, np.nan)
            perf_i_N_sinr = performance_history_sinr.get(last_task_idx, {}).get(i, np.nan)
            if not np.isnan(perf_i_i_sinr) and not np.isnan(perf_i_N_sinr):
                bwt_terms_sinr.append(perf_i_N_sinr - perf_i_i_sinr)

            perf_i_i_thrpt = final_task_performance_thrpt.get(i, np.nan)
            perf_i_N_thrpt = performance_history_thrpt.get(last_task_idx, {}).get(i, np.nan)
            if not np.isnan(perf_i_i_thrpt) and not np.isnan(perf_i_N_thrpt):
                bwt_terms_thrpt.append(perf_i_N_thrpt - perf_i_i_thrpt)

        if bwt_terms_dot: bwt_dot = np.nanmean(bwt_terms_dot)
        if bwt_terms_sinr: bwt_sinr = np.nanmean(bwt_terms_sinr)
        if bwt_terms_thrpt: bwt_thrpt = np.nanmean(bwt_terms_thrpt)

    # Performance matrices
    perf_matrix_dot = {i: {i: final_task_performance_dot[i]} for i in range(len(tasks))}
    perf_matrix_sinr = {i: {i: final_task_performance_sinr[i]} for i in range(len(tasks))}
    perf_matrix_thrpt = {i: {i: final_task_performance_thrpt[i]} for i in range(len(tasks))}
    for i in range(len(tasks)):
        for j in range(i):
            perf_matrix_dot[i][j] = performance_history_dot.get(i, {}).get(j, np.nan)
            perf_matrix_sinr[i][j] = performance_history_sinr.get(i, {}).get(j, np.nan)
            perf_matrix_thrpt[i][j] = performance_history_thrpt.get(i, {}).get(j, np.nan)

    results = {
        "name": "EWC",
        "avg_dot": avg_dot,
        "std_dot": std_dev_dot,
        "bwt_dot": bwt_dot,
        "avg_sinr": avg_sinr,
        "std_sinr": std_dev_sinr,
        "bwt_sinr": bwt_sinr,
        "avg_thrpt": avg_thrpt,
        "std_thrpt": std_dev_thrpt,
        "bwt_thrpt": bwt_thrpt,
        "avg_lat": avg_lat,
        "energy_j": total_energy_joules,
        "time": total_training_time,
        "final_perf_dot": final_task_performance_dot,
        "final_perf_sinr": final_task_performance_sinr,
        "final_perf_thrpt": final_task_performance_thrpt,
        "final_perf_lat": final_task_comp_latency,
        "perf_matrix_dot": perf_matrix_dot,
        "perf_matrix_sinr": perf_matrix_sinr,
        "perf_matrix_thrpt": perf_matrix_thrpt,
    }

    return results
# --- EWC + Replay Training Loop (Single GPU - EMA Fisher) ---
def training_loop_cl_ewc_replay(create_model_func, tasks, num_epochs_per_task, batch_size, learning_rate, ewc_lambda, replay_lambda=0.5):
    print(f"\n🧠 Starting Continual Learning Training Loop with EWC + Replay (EMA Fisher, Single GPU)...")
    log_file = "cl_beamforming_results_ewc_replay.log"
    log_file_diag = "per_user_diag_cl_ewc_replay_singleGPU.log"
    open(log_file, "w").close()
    open(log_file_diag, "w").close()

    # Performance Tracking
    final_task_performance_dot = {}
    final_task_performance_sinr = {}
    final_task_performance_thrpt = {}
    final_task_comp_latency = {}
    performance_history_dot = {}
    performance_history_sinr = {}
    performance_history_thrpt = {}

    # CL State Storage
    optimal_params = {}
    fisher_information = {}
    replay_buffer = ReplayBuffer(capacity_per_task=200)

    model = create_model_func()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    print("  Initializing EWC+Replay model and optimizer...")
    dummy_h = tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64)
    try:
        for i in range(NUM_TASKS):
            _ = model(dummy_h, task_idx=i, training=False)
        optimizer.build(model.trainable_variables)
        print("  EWC+Replay Model and Optimizer built successfully.")
        model.summary()
    except Exception as e:
        print(f"❌ Error building EWC+Replay model/opt: {e}")
        sys.exit(1)

    overall_start_time = time.time()

    for task_idx, task in enumerate(tasks):
        task_name = task['name']
        print(f"\n--- EWC+Replay | Task {task_idx + 1}/{len(tasks)}: {task_name} ---")
        optimizer.learning_rate.assign(learning_rate)

        task_start_time = time.time()

        # Initialize Fisher EMA for current task
        current_task_fisher_ema = {
            v.name: tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
            for v in model.trainable_variables
        }

        # Aggregate Fisher/Params from previous tasks
        optimal_params_prev_agg = {}
        fisher_info_prev_agg = {}
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

        for epoch in tqdm(range(num_epochs_per_task), desc=f"EWC+Replay {task_name}", leave=False):
            epoch_total_loss = 0.0
            epoch_task_loss = 0.0
            epoch_ewc_loss = 0.0
            epoch_replay_loss = 0.0
            epoch_sinr = np.nan
            epoch_dot = np.nan
            epoch_thrpt = np.nan
            epoch_log_lines = []

            for i_batch in range(NUM_BATCHES_PER_EPOCH):
                h_batch = generate_synthetic_batch(task, batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
                h_real = tf.math.real(h_batch)
                h_imag = tf.math.imag(h_batch)

                if (
                    tf.reduce_any(tf.math.is_nan(h_real)) or tf.reduce_any(tf.math.is_nan(h_imag)) or
                    tf.reduce_any(tf.math.is_inf(h_real)) or tf.reduce_any(tf.math.is_inf(h_imag))
                ):

                    print(f"NaN or Inf in h_batch for Task {task_idx}, Epoch {epoch+1}")
                w_zf_target = compute_zf_weights(h_batch, ZF_REG)

                # Add samples to replay buffer
                replay_buffer.add_samples(task_idx, h_batch, w_zf_target)

                # Training step
                total_loss, task_loss, ewc_loss_term, replay_loss, w_pred_raw, grads_and_vars = train_step_cl_ewc_replay(
                    model, optimizer, h_batch, task_idx, replay_buffer,
                    optimal_params_prev_agg, fisher_info_prev_agg, ewc_lambda, replay_lambda  # اصلاح اینجا
                )

                # Update Fisher EMA
                var_to_grad = {v.ref(): g for g, v in grads_and_vars}
                for var_name, fisher_variable in current_task_fisher_ema.items():
                    target_var = next((v_model for v_model in model.trainable_variables if v_model.name == var_name), None)
                    if target_var is not None:
                        g = var_to_grad.get(target_var.ref())
                        if g is not None:
                            grad_sq_mag = tf.cast(tf.math.real(g * tf.math.conj(g)), dtype=tf.float32)
                            avg_sq_grad_mag_this_step = tf.reduce_mean(grad_sq_mag)
                            new_fisher_val = (FISHER_EMA_DECAY * fisher_variable) + \
                                             ((1.0 - FISHER_EMA_DECAY) * avg_sq_grad_mag_this_step)
                            fisher_variable.assign(new_fisher_val)

                epoch_total_loss += total_loss.numpy()
                epoch_task_loss += task_loss.numpy()
                epoch_ewc_loss += ewc_loss_term.numpy()
                epoch_replay_loss += replay_loss.numpy()

                if i_batch == NUM_BATCHES_PER_EPOCH - 1:
                    sinr_db, mean_dot, avg_thrpt, log_lines = calculate_metrics_for_logging(h_batch, w_pred_raw)
                    epoch_sinr = sinr_db.numpy()
                    epoch_dot = mean_dot.numpy()
                    epoch_thrpt = avg_thrpt.numpy()
                    epoch_log_lines = log_lines
                if np.isnan(epoch_sinr) or np.isnan(epoch_dot) or np.isnan(epoch_thrpt):
                    print(f"Warning: NaN detected in metrics for Task {task_idx}, Epoch {epoch+1}")

            avg_epoch_loss = epoch_total_loss / NUM_BATCHES_PER_EPOCH

            if (epoch + 1) % 10 == 0:
                log_str = (
                    f"[EWC+Replay][Task {task_idx} - {task_name}][Epoch {epoch+1}] "
                    f"Avg Loss={avg_epoch_loss:.6f} | Task Loss={epoch_task_loss/NUM_BATCHES_PER_EPOCH:.6f} | "
                    f"EWC Loss={epoch_ewc_loss/NUM_BATCHES_PER_EPOCH:.6f} | Replay Loss={epoch_replay_loss/NUM_BATCHES_PER_EPOCH:.6f} | "
                    f"Last SINR={epoch_sinr:.2f} dB | Last Thrpt={epoch_thrpt:.4f} | Last |dot|={epoch_dot:.4f}\n"
                )
                print(log_str, end='')
                with open(log_file, "a") as f:
                    f.write(log_str)

            if (epoch + 1) % 20 == 0 and epoch_log_lines:
                with open(log_file_diag, "a") as f:
                    f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] ----\n")
                    for line in epoch_log_lines:
                        f.write(line + "\n")

        task_end_time = time.time()
        print(f"  Finished training Task {task_idx + 1} (EWC+Replay) in {task_end_time - task_start_time:.2f}s")

        # Store optimal parameters and Fisher info
        optimal_params[task_idx] = {v.name: v.numpy() for v in model.trainable_variables}
        fisher_information[task_idx] = {
            name: ema_var.numpy() for name, ema_var in current_task_fisher_ema.items()
            if name in optimal_params[task_idx]
        }
        print(f"  Stored optimal parameters and Fisher info (EMA) for Task {task_idx}.")

        # Evaluation (بقیه بدون تغییر باقی می‌مونه)
        print(f"  Evaluating performance...")
        dot_sum = 0.0
        sinr_sum = 0.0
        thrpt_sum = 0.0
        lat_sum = 0.0
        eval_steps = 0
        for _ in range(EVAL_BATCHES):
            h_eval = generate_synthetic_batch(tasks[task_idx], batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
            t0 = time.time()
            w_pred = model(h_eval, task_idx=task_idx, training=False)
            t1 = time.time()
            sinr_db, dot_val, thrpt_val = calculate_metrics(h_eval, w_pred, task_idx)
            latency_ms = (t1 - t0) * 1000 / batch_size
            if not np.isnan(dot_val): dot_sum += dot_val
            if not np.isnan(sinr_db): sinr_sum += sinr_db
            if not np.isnan(thrpt_val): thrpt_sum += thrpt_val
            if not np.isnan(latency_ms): lat_sum += latency_ms
            eval_steps += 1

        final_task_performance_dot[task_idx] = dot_sum / eval_steps
        final_task_performance_sinr[task_idx] = sinr_sum / eval_steps
        final_task_performance_thrpt[task_idx] = thrpt_sum / eval_steps
        final_task_comp_latency[task_idx] = lat_sum / eval_steps
        print(f"  Perf Task {task_idx}: |dot|={final_task_performance_dot[task_idx]:.4f}, "
              f"SINR={final_task_performance_sinr[task_idx]:.2f} dB, "
              f"Thrpt={final_task_performance_thrpt[task_idx]:.4f}, "
              f"Latency={final_task_comp_latency[task_idx]:.4f} ms")

        # Evaluate previous tasks for performance history
        if task_idx > 0:
            print("  Evaluating on previous tasks...")
            performance_history_dot[task_idx] = {}
            performance_history_sinr[task_idx] = {}
            performance_history_thrpt[task_idx] = {}
            for prev_task_idx in range(task_idx):
                prev_task_name = tasks[prev_task_idx]['name']
                prev_dot_sum = 0.0
                prev_sinr_sum = 0.0
                prev_thrpt_sum = 0.0
                num_prev_eval_steps = 0
                for _ in range(EVAL_BATCHES):
                    h_eval_prev = generate_synthetic_batch(tasks[prev_task_idx], batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
                    w_pred_prev = model(h_eval_prev, task_idx=prev_task_idx, training=False)
                    sinr_db, mean_dot, avg_thrpt = calculate_metrics(h_eval_prev, w_pred_prev, task_idx)
                    if not np.isnan(mean_dot): prev_dot_sum += mean_dot
                    if not np.isnan(sinr_db): prev_sinr_sum += sinr_db
                    if not np.isnan(avg_thrpt): prev_thrpt_sum += avg_thrpt
                    num_prev_eval_steps += 1
                avg_prev_dot = prev_dot_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                avg_prev_sinr = prev_sinr_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan  # اصلاح np.nano به np.nan
                avg_prev_thrpt = prev_thrpt_sum / num_prev_eval_steps if num_prev_eval_steps > 0 else np.nan
                performance_history_dot[task_idx][prev_task_idx] = avg_prev_dot
                performance_history_sinr[task_idx][prev_task_idx] = avg_prev_sinr
                performance_history_thrpt[task_idx][prev_task_idx] = avg_prev_thrpt
                print(f"    Perf on Task {prev_task_idx} ({prev_task_name}): |dot|={avg_prev_dot:.4f}, SINR={avg_prev_sinr:.2f} dB, Thrpt={avg_prev_thrpt:.4f} bps/Hz")
                tf.keras.backend.clear_session()
                gc.collect()

    # Wrap Up (بقیه بدون تغییر باقی می‌مونه)
    overall_end_time = time.time()
    total_training_time = overall_end_time - overall_start_time
    total_energy_joules = GPU_POWER_DRAW * total_training_time

    # Final Metric Summary
    avg_dot = np.nanmean(list(final_task_performance_dot.values()))
    avg_sinr = np.nanmean(list(final_task_performance_sinr.values()))
    avg_thrpt = np.nanmean(list(final_task_performance_thrpt.values()))
    avg_lat = np.nanmean(list(final_task_comp_latency.values()))

    std_dev_dot = np.nanstd(list(final_task_performance_dot.values()))
    std_dev_sinr = np.nanstd(list(final_task_performance_sinr.values()))
    std_dev_thrpt = np.nanstd(list(final_task_performance_thrpt.values()))

    # Backward Transfer (BWT)
    bwt_dot = np.nan
    bwt_sinr = np.nan
    bwt_thrpt = np.nan
    if len(tasks) > 1:
        bwt_terms_dot = []
        bwt_terms_sinr = []
        bwt_terms_thrpt = []
        last_task_idx = len(tasks) - 1
        for i in range(last_task_idx):
            perf_i_i_dot = final_task_performance_dot.get(i, np.nan)
            perf_i_N_dot = performance_history_dot.get(last_task_idx, {}).get(i, np.nan)
            if not np.isnan(perf_i_i_dot) and not np.isnan(perf_i_N_dot):
                bwt_terms_dot.append(perf_i_N_dot - perf_i_i_dot)

            perf_i_i_sinr = final_task_performance_sinr.get(i, np.nan)
            perf_i_N_sinr = performance_history_sinr.get(last_task_idx, {}).get(i, np.nan)
            if not np.isnan(perf_i_i_sinr) and not np.isnan(perf_i_N_sinr):
                bwt_terms_sinr.append(perf_i_N_sinr - perf_i_i_sinr)

            perf_i_i_thrpt = final_task_performance_thrpt.get(i, np.nan)
            perf_i_N_thrpt = performance_history_thrpt.get(last_task_idx, {}).get(i, np.nan)
            if not np.isnan(perf_i_i_thrpt) and not np.isnan(perf_i_N_thrpt):
                bwt_terms_thrpt.append(perf_i_N_thrpt - perf_i_i_thrpt)

        if bwt_terms_dot: bwt_dot = np.nanmean(bwt_terms_dot)
        if bwt_terms_sinr: bwt_sinr = np.nanmean(bwt_terms_sinr)
        if bwt_terms_thrpt: bwt_thrpt = np.nanmean(bwt_terms_thrpt)

    # Performance matrices
    perf_matrix_dot = {i: {i: final_task_performance_dot[i]} for i in range(len(tasks))}
    perf_matrix_sinr = {i: {i: final_task_performance_sinr[i]} for i in range(len(tasks))}
    perf_matrix_thrpt = {i: {i: final_task_performance_thrpt[i]} for i in range(len(tasks))}
    for i in range(len(tasks)):
        for j in range(i):
            perf_matrix_dot[i][j] = performance_history_dot.get(i, {}).get(j, np.nan)
            perf_matrix_sinr[i][j] = performance_history_sinr.get(i, {}).get(j, np.nan)
            perf_matrix_thrpt[i][j] = performance_history_thrpt.get(i, {}).get(j, np.nan)

    results = {
        "name": "EWC+Replay",
        "avg_dot": avg_dot,
        "std_dot": std_dev_dot,
        "bwt_dot": bwt_dot,
        "avg_sinr": avg_sinr,
        "std_sinr": std_dev_sinr,
        "bwt_sinr": bwt_sinr,
        "avg_thrpt": avg_thrpt,
        "std_thrpt": std_dev_thrpt,
        "bwt_thrpt": bwt_thrpt,
        "avg_lat": avg_lat,
        "energy_j": total_energy_joules,
        "time": total_training_time,
        "final_perf_dot": final_task_performance_dot,
        "final_perf_sinr": final_task_performance_sinr,
        "final_perf_thrpt": final_task_performance_thrpt,
        "final_perf_lat": final_task_comp_latency,
        "perf_matrix_dot": perf_matrix_dot,
        "perf_matrix_sinr": perf_matrix_sinr,
        "perf_matrix_thrpt": perf_matrix_thrpt,
    }

    return results
#------------------------------------------------------
# Plotting Function
#------------------------------------------------------
def plot_performance_matrix(results_dict, metric='dot', title_suffix="", filename="perf_matrix.png"):
    """ Plots the performance matrix using seaborn heatmap. """
    if plt is None or sns is None: return # Skip if libs not installed
    if metric not in ['dot', 'sinr', 'thrpt']: return # Support thrpt

    num_tasks = len(TASKS)
    perf_matrix = np.full((num_tasks, num_tasks), np.nan)
    perf_data_key = f"perf_matrix_{metric}"
    fmt = ".3f" if metric in ['dot', 'thrpt'] else ".1f"
    cmap = "viridis" if metric == 'dot' else "magma" if metric == 'sinr' else "plasma"
    cbar_label = f'Avg |dot|' if metric == 'dot' else 'Avg SINR (dB)' if metric == 'sinr' else 'Avg Throughput (bps/Hz)'

    if perf_data_key not in results_dict:
        print(f"Error plotting: Key '{perf_data_key}' not found in results dictionary.")
        return

    perf_map = results_dict[perf_data_key]
    for i in range(num_tasks):
        for j in range(num_tasks):
             perf_matrix[i, j] = perf_map.get(i, {}).get(j, np.nan)

    plt.figure(figsize=(8, 6))
    sns.heatmap(perf_matrix, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=[t["name"] for t in TASKS],
                yticklabels=[t["name"] for t in TASKS],
                linewidths=.5, cbar_kws={'label': cbar_label},
                annot_kws={"size": 10})
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

def plot_bwt_vs_lambda(lambda_values, results_list, metric='dot', filename='bwt_vs_lambda.png'):
    if plt is None: return
    plt.figure()
    y = [r.get(f'bwt_{metric}', np.nan) for r in results_list]
    plt.plot(lambda_values, y, marker='o', label=f'BWT {metric.upper()}')
    plt.xlabel("λ (EWC Regularization Strength)")
    plt.ylabel(f"BWT {metric.upper()}")
    plt.title(f"BWT vs λ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_final_task_bars(results, filename_prefix='final_perf'):
    if plt is None: return
    metrics = ['final_perf_dot', 'final_perf_sinr', 'final_perf_thrpt']
    for metric in metrics:
        values = results[metric]
        plt.figure()
        plt.bar([TASKS[i]['name'] for i in values], [values[i] for i in values])
        plt.title(f"{metric.replace('final_perf_', '').upper()} per Task")
        plt.ylabel(metric.replace('final_perf_', '').upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_{metric}.png")
        plt.close()

def plot_time_vs_efficiency(results_list, labels, filename='time_vs_efficiency.png'):
    if plt is None: return
    times = [r['time'] for r in results_list]
    effs = [r['avg_thrpt'] / r['avg_lat'] if r['avg_lat'] > 0 else np.nan for r in results_list]
    plt.figure()
    plt.scatter(times, effs)
    for i, label in enumerate(labels):
        plt.annotate(label, (times[i], effs[i]))
    plt.xlabel("Total Training Time (s)")
    plt.ylabel("Throughput / Latency")
    plt.title("Time vs. Operational Efficiency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
#------------------------------------------------------
# Main Execution Block
#------------------------------------------------------
if __name__ == "__main__":

    # --- Define factory function for model creation ---
    def create_fresh_model():
         print("  Creating new MultiHeadCVNNModel instance...")
         return MultiHeadCVNNModel(num_antennas=NUM_ANTENNAS, num_users=NUM_USERS, num_tasks=NUM_TASKS)

    all_results_list = []  # Store results from all runs

    # --- Run Baseline: Full Retraining ---
    print("\n" + "="*50); print("  RUNNING BASELINE: Full Retraining"); print("="*50)
    results_retraining = training_loop_baseline(
        baseline_name="Retraining", create_model_func=create_fresh_model, tasks=TASKS,
        num_epochs_per_task=NUM_EPOCHS_PER_TASK, batch_size=GLOBAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE, reset_model_per_task=True )
    all_results_list.append(results_retraining)

    # --- Run Baseline: Fine-tuning ---
    print("\n" + "="*50); print("  RUNNING BASELINE: Fine-tuning"); print("="*50)
    results_finetuning = training_loop_baseline(
        baseline_name="Finetuning", create_model_func=create_fresh_model, tasks=TASKS,
        num_epochs_per_task=NUM_EPOCHS_PER_TASK, batch_size=GLOBAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE, reset_model_per_task=False )
    all_results_list.append(results_finetuning)

    # --- Run CL Methods with all lambda values ---
    lambda_values = [50]#[1, 25, 50, 75, 250, 750, 2500, 10000]
    for lam in lambda_values:
        # EWC
        print(f"\n" + "="*50); print(f"  RUNNING CL METHOD: EWC (EMA Fisher) with λ = {lam}"); print("="*50)
        res_ewc = training_loop_cl_ewc_single_gpu(
            create_model_func=create_fresh_model,
            tasks=TASKS,
            num_epochs_per_task=NUM_EPOCHS_PER_TASK,
            batch_size=GLOBAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            ewc_lambda=lam
        )
        all_results_list.append(res_ewc)
        with open(f'ewc_lambda_{lam}_results.txt', 'w') as f:
            f.write(f"EWC Results for λ = {lam}\n")
            f.write(str(res_ewc) + "\n")

        # EWC + Replay
        print(f"\n" + "="*50); print(f"  RUNNING CL METHOD: EWC + Replay with λ = {lam}"); print("="*50)
        res_ewc_replay = training_loop_cl_ewc_replay(
            create_model_func=create_fresh_model,
            tasks=TASKS,
            num_epochs_per_task=NUM_EPOCHS_PER_TASK,
            batch_size=GLOBAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            ewc_lambda=lam,
            replay_lambda=0.5
        )
        all_results_list.append(res_ewc_replay)
        with open(f'ewc_replay_lambda_{lam}_results.txt', 'w') as f:
            f.write(f"EWC+Replay Results for λ = {lam}\n")
            f.write(str(res_ewc_replay) + "\n")

    # --- Final Comparative Summary ---
    print("\n" + "="*70); print("--- FINAL COMPARATIVE SUMMARY ---"); print("="*70)

    # Update headers to include all lambda runs
    headers = ["Metric", "Retraining", "Finetuning"] + \
              [f"EWC λ={lam}" for lam in lambda_values] + \
              [f"EWC+Replay λ={lam}" for lam in lambda_values]
    data = []
    def format_metric(val, precision=4):
        return f"{val:.{precision}f}" if not np.isnan(val) else "N/A"

    # Populate data for all metrics
    data.append(["Avg Final |dot|"] + 
                [format_metric(r['avg_dot']) for r in all_results_list])
    data.append(["Std Dev |dot|"] + 
                [format_metric(r['std_dot']) for r in all_results_list])
    data.append(["BWT |dot|"] + 
                [format_metric(r['bwt_dot']) for r in all_results_list])
    data.append(["-"] * len(headers))
    data.append(["Avg Final SINR (dB)"] + 
                [format_metric(r['avg_sinr'], 2) for r in all_results_list])
    data.append(["Std Dev SINR (dB)"] + 
                [format_metric(r['std_sinr'], 2) for r in all_results_list])
    data.append(["BWT SINR (dB)"] + 
                [format_metric(r['bwt_sinr'], 2) for r in all_results_list])
    data.append(["-"] * len(headers))
    data.append(["Avg Final Thrpt (bps/Hz)"] + 
                [format_metric(r['avg_thrpt'], 3) for r in all_results_list])
    data.append(["Std Dev Thrpt"] + 
                [format_metric(r['std_thrpt'], 3) for r in all_results_list])
    data.append(["BWT Thrpt"] + 
                [format_metric(r['bwt_thrpt'], 3) for r in all_results_list])
    data.append(["-"] * len(headers))
    data.append(["Avg Comp Latency (ms/sample)"] + 
                [format_metric(r['avg_lat'], 4) for r in all_results_list])
    data.append(["Total Train Time (s)"] + 
                [f"{r['time']:.1f}" for r in all_results_list])
    data.append(["Est. Train Energy (J)"] + 
                [f"{r['energy_j']:.0f}" for r in all_results_list])

    # Print Summary Table
    col_widths = [max(len(str(item)) for item in col) for col in zip(*([headers] + data))]
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line); print("-" * len(header_line))
    for row in data:
        if row[0] == "-": print("-" * len(header_line))
        else: print(" | ".join(f"{item:<{w}}" for item, w in zip(row, col_widths)))
    print("="*70)

    # Save summary table to main log file
    try:
        with open(SUMMARY_LOG_FILE, "w") as f:
            f.write("="*70 + "\n"); f.write("--- FINAL COMPARATIVE SUMMARY ---\n"); f.write("="*70 + "\n")
            f.write(header_line + "\n"); f.write("-" * len(header_line) + "\n")
            for row in data:
                 if row[0] == "-": f.write("-" * len(header_line) + "\n")
                 else: f.write(" | ".join(f"{item:<{w}}" for item, w in zip(row, col_widths)) + "\n")
            f.write("="*70 + "\n")
            f.write("\n\n--- Detailed Logs ---\n")
            f.write(f"Retraining: {RT_LOG_FILE}\n")
            f.write(f"Finetuning: {FT_LOG_FILE}\n")
            for lam in lambda_values:
                f.write(f"EWC λ={lam}: ewc_lambda_{lam}_results.txt\n")
                f.write(f"EWC+Replay λ={lam}: ewc_replay_lambda_{lam}_results.txt\n")
            f.write(f"Performance Plots: {PLOT_DOT_FILE}, {PLOT_SINR_FILE}, {PLOT_THRPT_FILE}\n")
        print(f"Final comparative summary saved to {SUMMARY_LOG_FILE}")
    except Exception as e: print(f"Error writing final summary to file: {e}")

    # --- Plotting ---
    if plt and sns:
        # Plot performance matrices for selected runs
        plot_performance_matrix(results_retraining, metric='dot', title_suffix=" (Retraining)", filename=PLOT_DOT_FILE.replace(".png", "_rt.png"))
        plot_performance_matrix(results_retraining, metric='sinr', title_suffix=" (Retraining)", filename=PLOT_SINR_FILE.replace(".png", "_rt.png"))
        plot_performance_matrix(results_finetuning, metric='dot', title_suffix=" (Finetuning)", filename=PLOT_DOT_FILE.replace(".png", "_ft.png"))
        plot_performance_matrix(results_finetuning, metric='sinr', title_suffix=" (Finetuning)", filename=PLOT_SINR_FILE.replace(".png", "_ft.png"))
        
        # Plot for each lambda (example for one lambda, you can loop if needed)
        ewc_runs = [r for r in all_results_list if r['name'] == 'EWC']
        ewc_replay_runs = [r for r in all_results_list if r['name'] == 'EWC+Replay']
        for i, lam in enumerate(lambda_values):
            plot_performance_matrix(ewc_runs[i], metric='dot', title_suffix=f" (EWC λ={lam})", filename=f"perf_matrix_dot_ewc_lambda_{lam}.png")
            plot_performance_matrix(ewc_runs[i], metric='sinr', title_suffix=f" (EWC λ={lam})", filename=f"perf_matrix_sinr_ewc_lambda_{lam}.png")
            plot_performance_matrix(ewc_replay_runs[i], metric='dot', title_suffix=f" (EWC+Replay λ={lam})", filename=f"perf_matrix_dot_ewc_replay_lambda_{lam}.png")
            plot_performance_matrix(ewc_replay_runs[i], metric='sinr', title_suffix=f" (EWC+Replay λ={lam})", filename=f"perf_matrix_sinr_ewc_replay_lambda_{lam}.png")
            plot_performance_matrix(ewc_replay_runs[i], metric='thrpt', title_suffix=f" (EWC+Replay λ={lam})", filename=f"perf_matrix_thrpt_ewc_replay_lambda_{lam}.png")

        # Plot final task bars
        plot_final_task_bars(results_retraining, filename_prefix="final_rt")
        plot_final_task_bars(results_finetuning, filename_prefix="final_ft")
        for i, lam in enumerate(lambda_values):
            plot_final_task_bars(ewc_runs[i], filename_prefix=f"final_ewc_lambda_{lam}")
            plot_final_task_bars(ewc_replay_runs[i], filename_prefix=f"final_ewc_replay_lambda_{lam}")

        # Plot BWT vs Lambda
        plot_bwt_vs_lambda(lambda_values, ewc_runs, metric='dot', filename='bwt_vs_lambda_dot_ewc.png')
        plot_bwt_vs_lambda(lambda_values, ewc_runs, metric='sinr', filename='bwt_vs_lambda_sinr_ewc.png')
        plot_bwt_vs_lambda(lambda_values, ewc_replay_runs, metric='dot', filename='bwt_vs_lambda_dot_ewc_replay.png')
        plot_bwt_vs_lambda(lambda_values, ewc_replay_runs, metric='sinr', filename='bwt_vs_lambda_sinr_ewc_replay.png')

        # Plot time vs efficiency
        plot_time_vs_efficiency(
            all_results_list,
            labels=["Retraining", "Finetuning"] + [f"EWC λ={lam}" for lam in lambda_values] + [f"EWC+Replay λ={lam}" for lam in lambda_values],
            filename="time_vs_efficiency.png"
        )

    print("\n--- All Training, Evaluation, and Plotting Finished ---")