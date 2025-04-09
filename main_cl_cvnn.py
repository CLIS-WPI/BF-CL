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
    sys.exit()

try:
    import cvnn.layers as complex_layers # Use alias for clarity
    import cvnn.activations as complex_activations
    # No need for CvnnModel base class
except ImportError as e:
    print("Error importing cvnn. Please install it using: pip install cvnn")
    print(e)
    sys.exit()

# Ensure tf-keras is available if needed by dependencies (already installed hopefully)
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

# --- Basic Settings ---
NUM_ANTENNAS = 64
NUM_USERS = 6
FREQ = 28e9
NUM_SLOTS = 10 # For data generation averaging
NOISE_POWER_LIN = 1e-3

# --- Task Definitions (Using all defined tasks now) ---
TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": [30e-9, 50e-9], "doppler": [10, 50], "channel": "TDL", "model": "A"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": [50e-9, 100e-9], "doppler": [50, 150], "channel": "Rayleigh"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": [200e-9, 500e-9], "doppler": [500, 2000], "channel": "TDL", "model": "C"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": [100e-9, 300e-9], "doppler": [200, 1000], "channel": "TDL", "model": "A"},
    # Add more tasks if desired
]
NUM_TASKS = len(TASKS)

# --- Training Configuration ---
NUM_EPOCHS_PER_TASK = 50  # Number of epochs per task
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_BATCHES_PER_EPOCH = 50 # How many batches to generate/train per epoch
ZF_REG = 1e-5              # Regularization for ZF calculation

# --- Continual Learning Configuration ---
USE_REPLAY = True
REPLAY_BUFFER_SIZE_PER_TASK = 200 # Number of samples to store per task
REPLAY_BATCH_SIZE = 16         # Number of replay samples per training step (if using replay)
REPLAY_LAMBDA = 0.5          # Weight for the replay loss term (0.0 to disable replay loss)

# --- GPU Setup ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ Using only GPU:0 -> {gpus[0].name}")
    except RuntimeError as e: print(f"❌ Error setting GPU visibility: {e}")
else: print("ℹ️ No GPU found, using CPU.")

#------------------------------------------------------
# Replay Buffer Implementation (Simple List-Based)
#------------------------------------------------------
# Store (h, w_zf) pairs for past tasks
replay_buffer_h = [[] for _ in range(NUM_TASKS)]
replay_buffer_w_zf = [[] for _ in range(NUM_TASKS)]
replay_samples_seen = [0 for _ in range(NUM_TASKS)] # Track samples added per task

def add_to_replay_buffer(task_idx, h_batch, w_zf_batch):
    """ Adds samples to the replay buffer for the given task_idx. """
    if not USE_REPLAY: return

    samples_in_batch = tf.shape(h_batch)[0]
    buffer_limit = REPLAY_BUFFER_SIZE_PER_TASK

    # Convert tensors to list of numpy arrays for easier handling in list buffer
    h_list = [h for h in tf.identity(h_batch).numpy()] # Use tf.identity to ensure tensor is copied
    w_zf_list = [w for w in tf.identity(w_zf_batch).numpy()]

    current_buffer_h = replay_buffer_h[task_idx]
    current_buffer_w_zf = replay_buffer_w_zf[task_idx]

    for i in range(samples_in_batch):
        idx_to_replace = replay_samples_seen[task_idx] % buffer_limit
        if len(current_buffer_h) < buffer_limit:
            current_buffer_h.append(h_list[i])
            current_buffer_w_zf.append(w_zf_list[i])
        else:
            # Simple reservoir sampling / overwrite strategy
            current_buffer_h[idx_to_replace] = h_list[i]
            current_buffer_w_zf[idx_to_replace] = w_zf_list[i]
            # Alternative: random index replacement: idx_to_replace = random.randrange(buffer_limit)
        replay_samples_seen[task_idx] += 1

def sample_from_replay_buffer(current_task_idx, num_samples):
    """ Samples data randomly from buffers of PAST tasks. """
    if not USE_REPLAY or current_task_idx == 0 or num_samples == 0:
        return None, None

    h_replay_samples = []
    w_zf_replay_samples = []
    samples_per_past_task = num_samples // current_task_idx # Distribute samples among past tasks

    for past_task_idx in range(current_task_idx):
        past_buffer_h = replay_buffer_h[past_task_idx]
        past_buffer_w_zf = replay_buffer_w_zf[past_task_idx]
        buffer_len = len(past_buffer_h)

        if buffer_len == 0: continue

        num_to_sample = min(samples_per_past_task, buffer_len)
        if num_to_sample > 0:
            indices = random.sample(range(buffer_len), num_to_sample)
            h_replay_samples.extend([past_buffer_h[i] for i in indices])
            w_zf_replay_samples.extend([past_buffer_w_zf[i] for i in indices])

    if not h_replay_samples: # If no samples could be drawn
        return None, None

    # Convert back to tensors
    h_replay = tf.convert_to_tensor(np.array(h_replay_samples), dtype=tf.complex64)
    w_zf_replay = tf.convert_to_tensor(np.array(w_zf_replay_samples), dtype=tf.complex64)

    return h_replay, w_zf_replay


#------------------------------------------------------
# HELPER FUNCTION: Generate Channel Data
# (Corrected: Fixed squeeze axes for Rayleigh)
#------------------------------------------------------
def generate_synthetic_batch(task, batch_size, num_antennas, num_users, num_slots, freq):
    """ Generates a batch of channel data. Returns complex channel h [B, U, A]. """
    h_users = []
    for _ in range(num_users):
        # Sample parameters
        delay = np.random.uniform(*task["delay_spread"])
        doppler = np.random.uniform(*task["doppler"])
        sampling_freq = int(max(1 / (delay + 1e-9), 2 * doppler)) * 10

        h_user = None
        if task["channel"] == "TDL":
            # --- Remove dtype here ---
            tdl = TDL( model=task.get("model", "A"), delay_spread=delay, carrier_frequency=freq,
                       num_tx_ant=num_antennas, num_rx_ant=1, min_speed=task["speed_range"][0],
                       max_speed=task["speed_range"][1] )
            # -------------------------
            # Output shape: [B, 1(Rx), 1(RxAnt), A(TxAnt), P(Paths), T(Time)]
            h_time, _ = tdl(batch_size=batch_size, num_time_steps=num_slots, sampling_frequency=sampling_freq)
            h_avg_time = tf.reduce_mean(h_time, axis=-1)      # Avg over T -> [B, 1, 1, A, P]
            h_comb_paths = tf.reduce_sum(h_avg_time, axis=-1) # Sum over P -> [B, 1, 1, A]
            h_user = tf.squeeze(h_comb_paths, axis=[1, 2])   # Squeeze -> [B, A]

        else: # Rayleigh
            # --- Remove dtype here ---
            rb = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=num_antennas)
             # -------------------------
            # Output shape: [B, 1(Rx), 1(RxAnt), 1(Tx), A(TxAnt), 1(Time)]
            h_block, _ = rb(batch_size=batch_size, num_time_steps=1)
            # Squeeze axes 1, 2, 3, 5 (Rx, RxAnt, Tx, Time)
            h_user = tf.squeeze(h_block, axis=[1, 2, 3, 5]) # Shape -> [B, A]

        if h_user is None: raise ValueError("h_user not defined")
        h_user_reshaped = tf.reshape(h_user, [batch_size, 1, num_antennas])
        h_users.append(h_user_reshaped)

    h_stacked = tf.stack(h_users, axis=1)  # Shape [B, U, 1, A]
    if tf.rank(h_stacked) == 4 and tf.shape(h_stacked)[2] == 1:
         h_stacked_squeezed = tf.squeeze(h_stacked, axis=2) # Shape -> [B, U, A]
    else:
         tf.print("Warning/Info: Shape before squeeze in generate_synthetic_batch was not [B, U, 1, A]:", tf.shape(h_stacked))
         h_stacked_squeezed = h_stacked

    # Normalize channel per user
    h_norm = h_stacked_squeezed / (tf.cast(tf.norm(h_stacked_squeezed, axis=-1, keepdims=True), tf.complex64) + 1e-8)
    return h_norm # Return only complex channel

def compute_zf_weights(h, reg=1e-5):
    # ... (Paste the LATEST CORRECT version of compute_zf_weights here) ...
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
        real = tf.random.normal([batch_size, num_users, num_antennas])
        imag = tf.random.normal([batch_size, num_users, num_antennas])
        w_random = tf.complex(real, imag)
        return tf.stop_gradient(tf.nn.l2_normalize(w_random, axis=-1))

#------------------------------------------------------
# HELPER FUNCTION: Calculate Metrics for Logging
# (Corrected: Removed tf.cast from interference calculation)
#------------------------------------------------------
#------------------------------------------------------
# HELPER FUNCTION: Calculate Metrics for Logging
# (Corrected: Returns Throughput as well)
#------------------------------------------------------
def calculate_metrics_for_logging(h, w_pred_norm, noise_power=1e-3):
    """ Calculates SINR (dB), Mean Alignment |dot|, Avg Throughput (bps/Hz) for logging. """
    sinr_dB_batch_mean = tf.constant(np.nan, dtype=tf.float32)
    mean_dot_abs = tf.constant(np.nan, dtype=tf.float32)
    avg_throughput = tf.constant(np.nan, dtype=tf.float32) # Added throughput
    log_lines = []

    try:
        with tf.device('/cpu:0'):
            # Alignment Calculation
            h_norm = tf.nn.l2_normalize(h, axis=-1, epsilon=1e-8)
            complex_dots = tf.reduce_sum(h_norm * tf.math.conj(w_pred_norm), axis=-1)
            dot_abs_all = tf.abs(complex_dots)
            mean_dot_abs = tf.reduce_mean(dot_abs_all)

            # SINR & Throughput Calculation
            num_users = tf.shape(h)[1]; batch_size = tf.shape(h)[0]
            signal_matrix = tf.matmul(h, w_pred_norm, adjoint_b=True)
            desired_signal = tf.linalg.diag_part(signal_matrix)
            desired_power = tf.abs(desired_signal)**2
            mask = 1.0 - tf.eye(num_users, batch_shape=[batch_size], dtype=h.dtype.real_dtype)
            interference_power_masked = tf.abs(signal_matrix)**2 * mask
            interference_power = tf.reduce_sum(interference_power_masked, axis=-1)
            sinr_linear = desired_power / (interference_power + noise_power)
            sinr_dB_batch_mean = 10.0 * tf.math.log(tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0)

            # Calculate Throughput (bps/Hz)
            sinr_per_user_bps = tf.math.log(1.0 + sinr_linear) / tf.math.log(2.0) # [B, U]
            avg_throughput = tf.reduce_mean(sinr_per_user_bps) # Scalar

            # Per-user diagnostics info
            if batch_size > 0:
                sinr_user0 = sinr_per_user_bps[0]; dot_abs_user0 = dot_abs_all[0]; complex_dots_user0 = complex_dots[0]
                for u in range(num_users):
                    complex_dot_u = complex_dots_user0[u]
                    dot_real = tf.math.real(complex_dot_u); dot_abs = tf.abs(complex_dot_u)
                    angle = tf.math.angle(complex_dot_u); sinr_u_val = sinr_user0[u].numpy()
                    if (dot_abs < 0.8) or (tf.abs(angle) > np.pi / 6):
                        line = f"User {u:02d} | dot={dot_real:.4f} | angle={angle:.4f} rad | |dot|={dot_abs:.4f} | SINR={sinr_u_val:.2f} bps/Hz"
                        log_lines.append(line)

    except Exception as e:
        tf.print("Error during metric calculation:", e, output_stream=sys.stderr)
        sinr_dB_batch_mean = tf.constant(np.nan, dtype=tf.float32)
        mean_dot_abs = tf.constant(np.nan, dtype=tf.float32)
        avg_throughput = tf.constant(np.nan, dtype=tf.float32)
        log_lines = ["Error in metrics calc."]

    # Return metrics
    return tf.identity(sinr_dB_batch_mean), tf.identity(mean_dot_abs), tf.identity(avg_throughput), log_lines

#------------------------------------------------------
# MULTI-HEAD COMPLEX-VALUED MODEL (CVNN)
#------------------------------------------------------
# --- MULTI-HEAD COMPLEX-VALUED MODEL (CVNN) ---
class MultiHeadCVNNModel(tf.keras.Model): # Inherit from tf.keras.Model
    # REPLACE THE __init__ METHOD
    def __init__(self, num_antennas, num_users, num_tasks):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.num_tasks = num_tasks

        # Shared Base Layers
        hidden_dim1 = 128
        hidden_dim2 = 256
        # ***** CHANGE: Use standard Keras initializer string *****
        initializer = 'glorot_uniform' # Standard Keras initializer
        # ***** END CHANGE *****
        activation = complex_activations.cart_relu # Use the function object

        self.dense1 = complex_layers.ComplexDense(hidden_dim1, activation=activation, kernel_initializer=initializer)
        self.dense2 = complex_layers.ComplexDense(hidden_dim2, activation=activation, kernel_initializer=initializer)

        # Task-Specific Output Heads
        self.output_heads = []
        for i in range(num_tasks):
            head = complex_layers.ComplexDense(
                self.num_antennas,
                activation='linear', # Linear activation for output weights
                kernel_initializer=initializer, # Use standard initializer
                name=f'head_task_{i}'
            )
            self.output_heads.append(head)
            # Optionally use setattr if list access causes issues later
            # setattr(self, f"head_task_{i}", head)

    # call method remains the same as previous CL CVNN version
    # @tf.function # Keep commented out for debugging initially
    def call(self, inputs, task_idx, training=False):
        # inputs: Complex channel h [B, U, A]
        # task_idx: Integer indicating which head to use
        x1 = self.dense1(inputs)    # Output: [B, U, H1]
        x2 = self.dense2(x1)        # Output: [B, U, H2]
        if not (0 <= task_idx < self.num_tasks):
            raise ValueError(f"Invalid task_idx: {task_idx}.")
        selected_head = self.output_heads[task_idx]
        w = selected_head(x2)      # Output: [B, U, A]
        return w

#------------------------------------------------------
# TRAINING STEP for CL with CVNN (Replay + MSE vs ZF)
#------------------------------------------------------
# @tf.function
#------------------------------------------------------
# TRAINING STEP for CL with CVNN (Replay + MSE vs ZF)
# (Modified to return throughput)
#------------------------------------------------------
# @tf.function
def train_step_cl_cvnn(model, optimizer, h_batch, task_idx, replay_lambda):
    """
    Performs one training step for Continual Learning using CVNN:
    - Calculates MSE loss vs ZF target for the current task.
    - Calculates MSE loss vs ZF target for replay samples from past tasks.
    - Combines losses and applies gradients.
    - Returns metrics including throughput.
    """
    w_zf_target_current = compute_zf_weights(h_batch, ZF_REG) # Use ZF_REG
    h_replay, w_zf_replay_target = sample_from_replay_buffer(task_idx, REPLAY_BATCH_SIZE)

    with tf.GradientTape() as tape:
        # Current Task Loss
        w_pred_current_raw = model(h_batch, task_idx=task_idx, training=True)
        w_pred_current_norm = tf.nn.l2_normalize(w_pred_current_raw, axis=-1, epsilon=1e-8)
        error_current = w_pred_current_norm - w_zf_target_current
        current_task_loss = tf.reduce_mean(tf.math.real(error_current * tf.math.conj(error_current)))

        total_loss = current_task_loss
        replay_task_loss = tf.constant(0.0, dtype=total_loss.dtype)

        # Replay Task Loss
        if h_replay is not None and w_zf_replay_target is not None and replay_lambda > 0:
            w_pred_replay_raw = model(h_replay, task_idx=task_idx, training=True) # Simple replay
            w_pred_replay_norm = tf.nn.l2_normalize(w_pred_replay_raw, axis=-1, epsilon=1e-8)
            error_replay = w_pred_replay_norm - w_zf_replay_target
            replay_task_loss = tf.reduce_mean(tf.math.real(error_replay * tf.math.conj(error_replay)))
            total_loss += replay_lambda * replay_task_loss

    # Gradient Calculation and Application
    trainable_vars = model.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)
    grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
    optimizer.apply_gradients(grads_and_vars)

    # Calculate metrics for logging (using current batch prediction)
    # ***** CHANGE: Get throughput from helper function *****
    sinr_db, mean_dot, avg_thrpt, log_lines = calculate_metrics_for_logging(
        h_batch, w_pred_current_norm, NOISE_POWER_LIN
    )
    # ***** END CHANGE *****

    return total_loss, current_task_loss, replay_task_loss, sinr_db, mean_dot, avg_thrpt, log_lines # Return throughput
#------------------------------------------------------
# CONTINUAL LEARNING Training Loop
#------------------------------------------------------
#------------------------------------------------------
# CONTINUAL LEARNING Training Loop (Modified for more KPIs)
#------------------------------------------------------
def training_loop_cl_cvnn(model, optimizer, tasks, num_epochs_per_task, batch_size, replay_lambda):
    """ Trains the MultiHead CVNN model sequentially on tasks with replay and logs extended KPIs. """
    print(f"\n🧠 Starting Continual Learning Training Loop...")
    num_tasks = len(tasks)
    log_file_summary = "summary_kpi_cl_cvnn.log"
    log_file_diag = "per_user_diag_cl_cvnn.log"
    open(log_file_summary, "w").close()
    open(log_file_diag, "w").close()

    # --- Dictionaries to store final performance metrics ---
    final_task_performance_dot = {}
    final_task_performance_sinr = {}
    final_task_performance_thrpt = {} # Added Throughput
    final_task_comp_latency = {}      # Added Latency

    # --- Dictionaries to store performance history for BWT ---
    performance_history_dot = {}
    performance_history_sinr = {}
    performance_history_thrpt = {}    # Added Throughput

    overall_start_time = time.time() # For total training time

    for task_idx, task in enumerate(tasks):
        task_name = task['name']
        print(f"\n--- Task {task_idx + 1}/{num_tasks}: {task_name} ---")

        # Adjust LR
        current_lr = LEARNING_RATE / (task_idx + 1) # Simple decay example
        optimizer.learning_rate.assign(current_lr)
        print(f"  Adjusted LR to: {optimizer.learning_rate.numpy():.2e}")

        task_start_time = time.time() # Time per task

        for epoch in tqdm(range(num_epochs_per_task), desc=f"Training {task_name}"):
            epoch_total_loss = 0.0; epoch_task_loss = 0.0; epoch_replay_loss = 0.0
            epoch_sinr = 0.0; epoch_dot = 0.0; epoch_thrpt = 0.0 # Added Throughput
            epoch_log_lines = []

            for i_batch in range(NUM_BATCHES_PER_EPOCH):
                 h_batch = generate_synthetic_batch(task, batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
                 w_zf_batch = compute_zf_weights(h_batch, ZF_REG) # Use ZF_REG
                 add_to_replay_buffer(task_idx, h_batch, w_zf_batch)
                 loss, task_loss, replay_loss, sinr_db, mean_dot, thrpt, log_lines = train_step_cl_cvnn( # Added thrpt
                     model, optimizer, h_batch, task_idx, replay_lambda
                 )

                 epoch_total_loss += loss.numpy()
                 epoch_task_loss += task_loss.numpy()
                 epoch_replay_loss += replay_loss.numpy()
                 epoch_sinr += sinr_db.numpy() # Ensure numpy conversion
                 epoch_dot += mean_dot.numpy()
                 epoch_thrpt += thrpt.numpy() # Added throughput
                 if log_lines and (i_batch == NUM_BATCHES_PER_EPOCH -1):
                      epoch_log_lines = log_lines

            # Average metrics over batches
            avg_epoch_total_loss = epoch_total_loss / NUM_BATCHES_PER_EPOCH
            avg_epoch_task_loss = epoch_task_loss / NUM_BATCHES_PER_EPOCH
            avg_epoch_replay_loss = epoch_replay_loss / NUM_BATCHES_PER_EPOCH
            avg_epoch_sinr = epoch_sinr / NUM_BATCHES_PER_EPOCH
            avg_epoch_dot = epoch_dot / NUM_BATCHES_PER_EPOCH
            avg_epoch_thrpt = epoch_thrpt / NUM_BATCHES_PER_EPOCH # Added throughput

            # Log summary periodically
            if (epoch + 1) % 10 == 0:
                 log_str = (f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] "
                            f"Avg Loss={avg_epoch_total_loss:.6f} "
                            f"(T={avg_epoch_task_loss:.6f}, R={avg_epoch_replay_loss:.6f}) | " # Shorter labels
                            f"Avg SINR={avg_epoch_sinr:.2f} dB | "
                            f"Avg Thrpt={avg_epoch_thrpt:.4f} | " # Added throughput
                            f"Avg |dot|={avg_epoch_dot:.4f}\n")
                 print(log_str, end='') # Print without extra newline
                 try:
                     with open(log_file_summary, "a") as f: f.write(log_str)
                 except Exception as e: print(f"Error writing to {log_file_summary}: {e}")

            # Log diagnostics periodically
            if (epoch + 1) % 20 == 0 and epoch_log_lines:
                try:
                    with open(log_file_diag, "a") as f:
                        f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] ----\n")
                        for line in epoch_log_lines: f.write(line + "\n")
                except Exception as e: print(f"Error writing to {log_file_diag}: {e}")

        # --- End of Epochs for Current Task ---
        task_end_time = time.time()
        print(f"  Finished training Task {task_idx + 1}: {task_name} in {task_end_time - task_start_time:.2f}s")

        # --- Evaluate performance after training task ---
        print(f"  Evaluating performance...")
        eval_batches = 20
        final_dot_sum = 0.0; final_sinr_sum = 0.0; final_thrpt_sum = 0.0; final_lat_sum = 0.0

        # Evaluate on current task data
        for _ in range(eval_batches):
             h_eval = generate_synthetic_batch(tasks[task_idx], batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
             t_start = time.time()
             w_pred_eval_raw = model(h_eval, task_idx=task_idx, training=False)
             t_end = time.time()
             w_pred_eval_norm = tf.nn.l2_normalize(w_pred_eval_raw, axis=-1, epsilon=1e-8)
             # Get all metrics
             sinr_db, mean_dot, avg_thrpt, _ = calculate_metrics_for_logging(h_eval, w_pred_eval_norm, NOISE_POWER_LIN)
             comp_lat_ms = (t_end - t_start) * 1000 / batch_size # Latency per sample in ms

             final_dot_sum += mean_dot.numpy()
             final_sinr_sum += sinr_db.numpy()
             final_thrpt_sum += avg_thrpt.numpy()
             final_lat_sum += comp_lat_ms

        final_task_performance_dot[task_idx] = final_dot_sum / eval_batches
        final_task_performance_sinr[task_idx] = final_sinr_sum / eval_batches
        final_task_performance_thrpt[task_idx] = final_thrpt_sum / eval_batches
        final_task_comp_latency[task_idx] = final_lat_sum / eval_batches
        print(f"  Performance on Task {task_idx} ({task_name}): "
              f"|dot|={final_task_performance_dot[task_idx]:.4f}, "
              f"SINR={final_task_performance_sinr[task_idx]:.2f} dB, "
              f"Thrpt={final_task_performance_thrpt[task_idx]:.4f} bps/Hz, "
              f"CompLat={final_task_comp_latency[task_idx]:.4f} ms/sample")

        # Evaluate on previous tasks (for BWT)
        performance_history_dot[task_idx] = {}
        performance_history_sinr[task_idx] = {}
        performance_history_thrpt[task_idx] = {} # Added Thrpt history
        if task_idx > 0:
             print("  Evaluating on previous tasks...")
             for prev_task_idx in range(task_idx):
                  prev_task_name = tasks[prev_task_idx]['name']
                  prev_dot_sum = 0.0; prev_sinr_sum = 0.0; prev_thrpt_sum = 0.0; prev_lat_sum = 0.0
                  for _ in range(eval_batches):
                       h_eval_prev = generate_synthetic_batch(tasks[prev_task_idx], batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
                       t_start = time.time()
                       w_pred_prev_raw = model(h_eval_prev, task_idx=prev_task_idx, training=False)
                       t_end = time.time()
                       w_pred_prev_norm = tf.nn.l2_normalize(w_pred_prev_raw, axis=-1, epsilon=1e-8)
                       sinr_db, mean_dot, avg_thrpt, _ = calculate_metrics_for_logging(h_eval_prev, w_pred_prev_norm, NOISE_POWER_LIN)
                       comp_lat_ms = (t_end - t_start) * 1000 / batch_size

                       prev_dot_sum += mean_dot.numpy()
                       prev_sinr_sum += sinr_db.numpy()
                       prev_thrpt_sum += avg_thrpt.numpy()
                       # Latency isn't typically stored for BWT calc, but we could track it

                  avg_prev_dot = prev_dot_sum / eval_batches
                  avg_prev_sinr = prev_sinr_sum / eval_batches
                  avg_prev_thrpt = prev_thrpt_sum / eval_batches # Added
                  performance_history_dot[task_idx][prev_task_idx] = avg_prev_dot
                  performance_history_sinr[task_idx][prev_task_idx] = avg_prev_sinr
                  performance_history_thrpt[task_idx][prev_task_idx] = avg_prev_thrpt # Added
                  print(f"    Perf on Task {prev_task_idx} ({prev_task_name}): "
                        f"|dot|={avg_prev_dot:.4f}, SINR={avg_prev_sinr:.2f} dB, Thrpt={avg_prev_thrpt:.4f} bps/Hz")

        # --- Optional: Clear session/memory ---
        tf.keras.backend.clear_session()
        gc.collect()

    # --- End of All Tasks ---
    overall_end_time = time.time()
    total_training_time = overall_end_time - overall_start_time
    print(f"\n✅ Finished Continual Learning Training Loop in {total_training_time:.2f}s")

    # --- Calculate Final CL Metrics ---
    avg_acc_dot = np.mean(list(final_task_performance_dot.values()))
    avg_sinr = np.mean(list(final_task_performance_sinr.values()))
    avg_thrpt = np.mean(list(final_task_performance_thrpt.values()))
    avg_lat = np.mean(list(final_task_comp_latency.values()))

    bwt_dot = 0.0; bwt_sinr = 0.0; bwt_thrpt = 0.0
    if num_tasks > 1:
        bwt_terms_dot = []; bwt_terms_sinr = []; bwt_terms_thrpt = []
        last_task_idx = num_tasks - 1
        for i in range(num_tasks - 1): # Task learned at step i
            perf_after_i_dot = final_task_performance_dot[i]
            perf_after_last_dot = performance_history_dot[last_task_idx][i]
            bwt_terms_dot.append(perf_after_last_dot - perf_after_i_dot)

            perf_after_i_sinr = final_task_performance_sinr[i]
            perf_after_last_sinr = performance_history_sinr[last_task_idx][i]
            bwt_terms_sinr.append(perf_after_last_sinr - perf_after_i_sinr)

            perf_after_i_thrpt = final_task_performance_thrpt[i]
            perf_after_last_thrpt = performance_history_thrpt[last_task_idx][i]
            bwt_terms_thrpt.append(perf_after_last_thrpt - perf_after_i_thrpt)

        bwt_dot = np.mean(bwt_terms_dot)
        bwt_sinr = np.mean(bwt_terms_sinr)
        bwt_thrpt = np.mean(bwt_terms_thrpt)

    # Estimate Training Energy
    GPU_POWER_DRAW = 400 # Watts (Adjust if needed)
    total_training_energy_J = GPU_POWER_DRAW * total_training_time # Joules
    total_training_energy_kWh = total_training_energy_J / (3600 * 1000) # kWh

    print("\n--- CL Metrics Summary ---")
    print(f"  Average Final |dot| (Acc): {avg_acc_dot:.4f}")
    print(f"  Average Final SINR: {avg_sinr:.2f} dB")
    print(f"  Average Final Throughput: {avg_thrpt:.4f} bps/Hz")
    print(f"  Average Computational Latency: {avg_lat:.4f} ms/sample")
    print(f"  Backward Transfer |dot| (BWT): {bwt_dot:.4f}")
    print(f"  Backward Transfer SINR (BWT): {bwt_sinr:.2f} dB")
    print(f"  Backward Transfer Thrpt (BWT): {bwt_thrpt:.4f} bps/Hz")
    print(f"  Total Training Time: {total_training_time:.2f} s")
    print(f"  Estimated Training Energy: {total_training_energy_J:.2f} J ({total_training_energy_kWh:.4f} kWh)")

    # Write summary metrics to file
    try:
        with open(log_file_summary, "a") as f:
            f.write("\n--- CL Metrics Summary ---\n")
            f.write(f"  Average Final |dot| (Acc): {avg_acc_dot:.4f}\n")
            f.write(f"  Average Final SINR: {avg_sinr:.2f} dB\n")
            f.write(f"  Average Final Throughput: {avg_thrpt:.4f} bps/Hz\n")
            f.write(f"  Average Computational Latency: {avg_lat:.4f} ms/sample\n")
            f.write(f"  Backward Transfer |dot| (BWT): {bwt_dot:.4f}\n")
            f.write(f"  Backward Transfer SINR (BWT): {bwt_sinr:.2f} dB\n")
            f.write(f"  Backward Transfer Thrpt (BWT): {bwt_thrpt:.4f} bps/Hz\n")
            f.write(f"  Total Training Time: {total_training_time:.2f} s\n")
            f.write(f"  Estimated Training Energy: {total_training_energy_J:.2f} J ({total_training_energy_kWh:.4f} kWh)\n")

            # Performance Matrix (|dot|)
            f.write("--- Task Performance Matrix (|dot|) ---\n")
            header = "Task Learned | " + " | ".join([f"Eval T{i}" for i in range(num_tasks)]) + "\n"
            f.write(header + "-" * len(header) + "\n")
            for i in range(num_tasks):
                 row = f"     Task {i}    | "
                 perf_row = [f"{final_task_performance_dot.get(i, np.nan):.4f}" if j == i else f"{performance_history_dot.get(max(i,j), {}).get(min(i,j), np.nan):.4f}" if j<i else "  nan " for j in range(num_tasks)]
                 row += " | ".join(perf_row) + "\n"; f.write(row)

            # Performance Matrix (SINR)
            f.write("--- Task Performance Matrix (SINR dB) ---\n")
            header = "Task Learned | " + " | ".join([f"Eval T{i}" for i in range(num_tasks)]) + "\n"
            f.write(header + "-" * len(header) + "\n")
            for i in range(num_tasks):
                 row = f"     Task {i}    | "
                 perf_row = [f"{final_task_performance_sinr.get(i, np.nan):.2f}" if j == i else f"{performance_history_sinr.get(max(i,j), {}).get(min(i,j), np.nan):.2f}" if j<i else " nan " for j in range(num_tasks)]
                 row += " | ".join(perf_row) + "\n"; f.write(row)

    except Exception as e: print(f"Error writing final summary: {e}")


# --- Main Execution Block (Using CL CVNN) ---
if __name__ == "__main__":
    # --- Configuration ---
    print(f"--- Starting Continual Learning CVNN Training ---")
    print(f"Num Tasks: {NUM_TASKS}")
    print(f"Epochs per Task: {NUM_EPOCHS_PER_TASK}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Replay Enabled: {USE_REPLAY}")
    if USE_REPLAY:
        print(f"Replay Buffer Size per Task: {REPLAY_BUFFER_SIZE_PER_TASK}")
        print(f"Replay Batch Size: {REPLAY_BATCH_SIZE}")
        print(f"Replay Lambda: {REPLAY_LAMBDA}")
    print("-" * 36)

    # --- Model and Optimizer Initialization ---
    print("Initializing MultiHead CVNN model...")
    model = MultiHeadCVNNModel(num_antennas=NUM_ANTENNAS, num_users=NUM_USERS, num_tasks=NUM_TASKS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # --- Build Model and Optimizer ---
    print("Building model...")
    # ***** ADD THIS BLOCK *****
    # --- Build all task heads before building optimizer ---
    print("Building all task heads by calling the model once for each task...")
    dummy_h_complex = tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64)
    for i in range(NUM_TASKS):
        try:
            # Call model once with each task_idx to build the corresponding head
            _ = model(dummy_h_complex, task_idx=i, training=False)
            print(f"  Successfully called model for task head {i}.")
        except Exception as e:
            print(f"❌ Error explicitly building head for task {i}: {e}")
            # Decide if you want to exit or continue if a head fails to build
            # sys.exit(1)
    print("Finished calling model for all task heads.")
    # ***** END OF ADDED BLOCK *****

    # --- Build Model and Optimizer (Now optimizer sees all variables) ---
    print("Re-calling build for optimizer with all variables...") # Message changed slightly
    try:
        # Optimizer build should now include variables from all heads
        optimizer.build(model.trainable_variables)
        print("✅ MultiHead CVNN Model and Optimizer built successfully.")
        model.summary() # Optional: summary might look different now
    except Exception as e:
        print(f"❌ Error building optimizer after building all heads: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    dummy_h_complex = tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64)
    try:
        # Build by calling with first task index
        _ = model(dummy_h_complex, task_idx=0, training=False)
        optimizer.build(model.trainable_variables)
        print("✅ MultiHead CVNN Model and Optimizer built successfully.")
        model.summary()
    except Exception as e:
        print(f"❌ Error building MultiHead CVNN model/optimizer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Start Continual Learning Training Loop ---
    training_loop_cl_cvnn(
        model=model,
        optimizer=optimizer,
        tasks=TASKS,
        num_epochs_per_task=NUM_EPOCHS_PER_TASK,
        batch_size=BATCH_SIZE,
        replay_lambda=REPLAY_LAMBDA
    )

    print("\n--- Continual Learning CVNN Training Finished ---")