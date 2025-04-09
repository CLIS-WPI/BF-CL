# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
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

from tqdm import tqdm
import time
import os
import sys

# --- Basic Settings (Keep them simple for now) ---
NUM_ANTENNAS = 64
NUM_USERS = 6
FREQ = 28e9
NUM_SLOTS = 10 # Used for data generation averaging
NOISE_POWER_LIN = 1e-3 # Linear noise power for SINR calculation (logging only)

# --- Task Definition (Only using the first one) ---
TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": [30e-9, 50e-9], "doppler": [10, 50], "channel": "TDL", "model": "A"},
    # Add other tasks back later if needed for CL
]
CHOSEN_TASK_INDEX = 0

# --- Training Configuration ---
NUM_EPOCHS = 100      # Train longer to see if learning occurs
LEARNING_RATE = 1e-4 # Start with this LR
BATCH_SIZE = 32       # Can potentially increase batch size now

# --- GPU Setup ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ Using only GPU:0 -> {gpus[0].name}")
    except RuntimeError as e:
        print(f"❌ Error setting GPU visibility: {e}")
else:
    print("ℹ️ No GPU found, using CPU.")

#------------------------------------------------------
# HELPER FUNCTION: Generate Channel Data
# (Based on the last working version)
#------------------------------------------------------
def generate_synthetic_batch(task, batch_size, num_antennas, num_users, num_slots, freq):
    h_users = []
    for _ in range(num_users):
        delay = np.random.uniform(*task["delay_spread"])
        doppler = np.random.uniform(*task["doppler"])
        sampling_freq = int(max(1 / (delay + 1e-9), 2 * doppler)) * 10

        h_user = None
        if task["channel"] == "TDL":
            tdl = TDL(model=task.get("model", "A"), delay_spread=delay, carrier_frequency=freq,
                      num_tx_ant=num_antennas, num_rx_ant=1, min_speed=task["speed_range"][0],
                      max_speed=task["speed_range"][1])  # dtype حذف شده
            h_time, _ = tdl(batch_size=batch_size, num_time_steps=num_slots, sampling_frequency=sampling_freq)
            h_avg_time = tf.reduce_mean(h_time, axis=-1)
            h_comb_paths = tf.reduce_sum(h_avg_time, axis=-1)
            h_user = tf.squeeze(h_comb_paths, axis=[1, 2])
        else:
            rb = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=num_antennas, dtype=tf.complex64)
            h_block, _ = rb(batch_size=batch_size, num_time_steps=1)
            h_user = tf.squeeze(h_block, axis=[1, 2, 4])

        if h_user is None: raise ValueError("h_user not defined")
        h_user_reshaped = tf.reshape(h_user, [batch_size, 1, num_antennas])
        h_users.append(h_user_reshaped)

    h_stacked = tf.stack(h_users, axis=1)  # Shape [B, U, 1, A]
    if tf.rank(h_stacked) == 4 and tf.shape(h_stacked)[2] == 1:
         h_stacked_squeezed = tf.squeeze(h_stacked, axis=2)
    else:
         tf.print("Warning: Unexpected shape in generate_synthetic_batch:", tf.shape(h_stacked))
         h_stacked_squeezed = h_stacked

    h_norm = h_stacked_squeezed / (tf.cast(tf.norm(h_stacked_squeezed, axis=-1, keepdims=True), tf.complex64) + 1e-8)
    return h_norm

#------------------------------------------------------
# HELPER FUNCTION: Calculate ZF Weights (Target)
#------------------------------------------------------
def compute_zf_weights(h, reg=1e-5):
    """ Computes normalized Zero-Forcing (ZF) weights. """
    batch_size = tf.shape(h)[0]
    num_users = tf.shape(h)[1]
    num_antennas = tf.shape(h)[2]
    identity = tf.eye(num_users, batch_shape=[batch_size], dtype=tf.complex64)
    try:
        h_herm = tf.linalg.adjoint(h) # [B, A, U]
        hh_herm = tf.matmul(h, h_herm) # [B, U, U]
        inv_term = tf.linalg.inv(hh_herm + tf.cast(reg, tf.complex64) * identity) # [B, U, U]
        w_unnormalized_intermediate = tf.matmul(h_herm, inv_term) # [B, A, U]
        w_zf_unnorm = tf.linalg.adjoint(w_unnormalized_intermediate) # [B, U, A]
        w_zf_normalized = tf.nn.l2_normalize(w_zf_unnorm, axis=-1, epsilon=1e-8)
        return tf.stop_gradient(w_zf_normalized)
    except Exception as e:
        tf.print("Warning: ZF calculation failed, returning random weights.", e, output_stream=sys.stderr)
        real = tf.random.normal([batch_size, num_users, num_antennas])
        imag = tf.random.normal([batch_size, num_users, num_antennas])
        w_random = tf.complex(real, imag)
        return tf.stop_gradient(tf.nn.l2_normalize(w_random, axis=-1))

#------------------------------------------------------
# HELPER FUNCTION: Calculate Metrics for Logging
#------------------------------------------------------
def calculate_metrics_for_logging(h, w_pred_norm, noise_power=1e-3):
    with tf.device('/cpu:0'):
        h_norm = tf.nn.l2_normalize(h, axis=-1, epsilon=1e-8)
        complex_dots = tf.reduce_sum(h_norm * tf.math.conj(w_pred_norm), axis=-1)
        dot_abs_all = tf.abs(complex_dots)
        mean_dot_abs = tf.reduce_mean(dot_abs_all)

        num_users = tf.shape(h)[1]
        batch_size = tf.shape(h)[0]
        signal_matrix = tf.matmul(h, w_pred_norm, adjoint_b=True)
        desired_signal = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.abs(desired_signal)**2
        mask = 1.0 - tf.eye(num_users, batch_shape=[batch_size], dtype=h.dtype.real_dtype)  # float32 یا real_dtype
        interference_power_masked = tf.abs(signal_matrix)**2 * mask  # ضرب در ماسک float
        interference_power = tf.reduce_sum(tf.math.real(interference_power_masked), axis=-1)
        sinr_linear = desired_power / (interference_power + noise_power)
        sinr_dB_batch_mean = 10.0 * tf.math.log(tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0)

        log_lines = []
        sinr_per_user_bps = tf.math.log(1.0 + sinr_linear) / tf.math.log(2.0)
        sinr_user0 = sinr_per_user_bps[0]; dot_abs_user0 = dot_abs_all[0]; complex_dots_user0 = complex_dots[0]
        for u in range(num_users):
            complex_dot_u = complex_dots_user0[u]
            dot_real = tf.math.real(complex_dot_u); dot_abs = tf.abs(complex_dot_u)
            angle = tf.math.angle(complex_dot_u); sinr_u_val = sinr_user0[u].numpy()
            if (dot_abs < 0.8) or (tf.abs(angle) > np.pi / 6):
                line = f"User {u:02d} | dot={dot_real:.4f} | angle={angle:.4f} rad | |dot|={dot_abs:.4f} | SINR={sinr_u_val:.2f} bps/Hz"
                log_lines.append(line)

    return sinr_dB_batch_mean, mean_dot_abs, log_lines


#------------------------------------------------------
# COMPLEX-VALUED MODEL (CVNN)
#------------------------------------------------------
class SimpleCVNNModel(tf.keras.Model):
    def __init__(self, num_antennas, num_users):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        hidden_dim1 = 128
        hidden_dim2 = 256
        self.input_dim = num_antennas

        self.dense1 = complex_layers.ComplexDense(hidden_dim1, activation=complex_activations.cart_relu)
        self.dense2 = complex_layers.ComplexDense(hidden_dim2, activation=complex_activations.cart_relu)
        self.dense_out = complex_layers.ComplexDense(self.num_antennas, activation='linear')

    @tf.function
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(inputs, [-1, self.input_dim])  # [B*U, A]
        x1 = self.dense1(x)    # [B*U, H1]
        x2 = self.dense2(x1)   # [B*U, H2]
        w = self.dense_out(x2) # [B*U, A]
        w = tf.reshape(w, [batch_size, self.num_users, self.num_antennas])  # [B, U, A]
        return w

#------------------------------------------------------
# TRAINING STEP for CVNN + MSE vs ZF Loss
#------------------------------------------------------
# @tf.function # Keep commented out for debugging
def train_step_cvnn_zf(model, optimizer, h_batch):
    """
    Performs one training step using CVNN model and MSE loss vs ZF target.
    """
    # Calculate target ZF weights
    w_zf_target = compute_zf_weights(h_batch) # Normalized target

    with tf.GradientTape() as tape:
        # 1. Get model's prediction (complex output)
        w_pred_raw = model(h_batch, training=True)

        # 2. Normalize prediction for MSE calculation
        w_pred_norm = tf.nn.l2_normalize(w_pred_raw, axis=-1, epsilon=1e-8)

        # 3. Calculate MSE Loss
        error = w_pred_norm - w_zf_target
        total_loss = tf.reduce_mean(tf.math.real(error * tf.math.conj(error)))

    # 4. Calculate and Apply Gradients
    # Use model.trainable_variables for CVNN model too
    trainable_vars = model.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)

    # Clip gradients (optional, but can help stability)
    # grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in grads]

    grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
    optimizer.apply_gradients(grads_and_vars)

    # 5. Calculate metrics for logging (using normalized prediction)
    sinr_db, mean_dot, log_lines = calculate_metrics_for_logging(h_batch, w_pred_norm, NOISE_POWER_LIN)

    return total_loss, sinr_db, mean_dot, log_lines

#------------------------------------------------------
# TRAINING LOOP (Adapted for CVNN)
#------------------------------------------------------
def training_loop_single_task_cvnn(model, optimizer, task, task_idx, num_epochs, batch_size):
    """ Trains the CVNN model on a single specified task using ZF target. """
    print(f"\n🧠 Training CVNN Model on Task: {task['name']} (Index: {task_idx}) for {num_epochs} epochs.")
    task_name = task['name']
    log_file_summary = "summary_kpi_cvnn.log"
    log_file_diag = "per_user_diag_cvnn.log"
    open(log_file_summary, "w").close()
    open(log_file_diag, "w").close()

    # Generate entire dataset once (if memory allows, otherwise generate per epoch)
    # Consider reducing NUM_SLOTS_TOTAL if memory is an issue
    NUM_SLOTS_TOTAL = 200 # Example total slots for dataset
    print(f"Generating {NUM_SLOTS_TOTAL} channel slots for training...")
    h_data = generate_synthetic_batch(task, NUM_SLOTS_TOTAL, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)
    # Transpose if needed? generate_synthetic_batch returns [B, U, A] - need to adapt dataset creation
    # Let's assume generate_synthetic_batch was adapted to return [NumSamples, U, A]
    # Need to re-think data generation slightly if using tf.data.Dataset

    # Simpler approach: Generate data per epoch inside the loop
    print(f"Data will be generated per epoch.")

    for epoch in tqdm(range(num_epochs), desc=f"Training {task_name}"):
        # Generate data for this epoch
        # Calculate number of batches needed
        num_batches = 50 # Generate fixed number of batches per epoch, e.g. 50
        epoch_loss = 0.0
        epoch_sinr = 0.0
        epoch_dot = 0.0
        epoch_log_lines = []

        for i_batch in range(num_batches):
             # Generate a batch of channel data
             h_batch = generate_synthetic_batch(task, batch_size, NUM_ANTENNAS, NUM_USERS, NUM_SLOTS, FREQ)

             # Perform one training step
             loss, sinr_db, mean_dot, log_lines = train_step_cvnn_zf(model, optimizer, h_batch)

             epoch_loss += loss.numpy()
             epoch_sinr += sinr_db # Already numpy scalar
             epoch_dot += mean_dot.numpy()
             if log_lines and (i_batch == num_batches -1): # Keep logs from last batch only
                  epoch_log_lines = log_lines

        # Average metrics over batches
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_sinr = epoch_sinr / num_batches
        avg_epoch_dot = epoch_dot / num_batches

        # Logging
        if (epoch + 1) % 10 == 0:
             try:
                 with open(log_file_summary, "a") as f:
                     f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] "
                             f"Avg MSE Loss={avg_epoch_loss:.6f} | "
                             f"Avg SINR={avg_epoch_sinr:.2f} dB | "
                             f"Avg |dot|={avg_epoch_dot:.4f}\n")
             except Exception as e: print(f"Error writing to {log_file_summary}: {e}")

        if (epoch + 1) % 20 == 0 and epoch_log_lines:
            try:
                with open(log_file_diag, "a") as f:
                    f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] ----\n")
                    for line in epoch_log_lines: f.write(line + "\n")
            except Exception as e: print(f"Error writing to {log_file_diag}: {e}")

    print(f"\n✅ Finished training loop for task: {task_name}")

# --- Main Execution Block (Using CVNN) ---
if __name__ == "__main__":

    print(f"--- Starting Single Task CVNN Training ---")
    single_task = TASKS[CHOSEN_TASK_INDEX]
    print(f"Task: {single_task['name']} (Index: {CHOSEN_TASK_INDEX})")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("-" * 36)

    # --- No Baselines computed for CVNN run initially ---
    print("\n📊 Skipping Baseline Calculation for CVNN run.")
    print("-" * 30)

    # --- Model and Optimizer Initialization ---
    print("Initializing CVNN model...")
    model = SimpleCVNNModel(num_antennas=NUM_ANTENNAS, num_users=NUM_USERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # --- Build Model and Optimizer ---
    print("Building model...")
    # Use complex dummy input
    dummy_h_complex = tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64)
    try:
        _ = model(dummy_h_complex, training=False) # Build model
        # Build optimizer (optional but good practice)
        # Need trainable variables AFTER first call
        optimizer.build(model.trainable_variables)
        print("✅ CVNN Model and Optimizer built successfully.")
        model.summary() # Print model summary
    except Exception as e:
        print(f"❌ Error building CVNN model/optimizer: {e}")
        # print traceback
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Skip Sanity Check for now ---
    print("\n Skipping sanity check pass...")

    # --- Start Single Task Training Loop for CVNN ---
    training_loop_single_task_cvnn(
        model=model,
        optimizer=optimizer,
        task=single_task,
        task_idx=CHOSEN_TASK_INDEX,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE
    )

    print("\n--- Single Task CVNN Training Finished ---")