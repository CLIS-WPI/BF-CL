import tensorflow as tf
import numpy as np
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
from tqdm import tqdm
import time
from tensorflow.keras import layers
import os
import sys
from sionna.phy.mimo import cbf_precoding_matrix, rzf_precoding_matrix, normalize_precoding_power

# Enable Eager execution for easier debugging if needed (can comment out for performance)
# tf.config.run_functions_eagerly(True)

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

# --- Settings ---
NUM_ANTENNAS = 64
FREQ = 28e9
NUM_SLOTS = 10
BATCH_SIZE = 4 # Keep small for debugging
NUM_USERS = 6

# --- Task Definitions ---
TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": [30e-9, 50e-9], "doppler": [10, 50], "coherence_time": 0.423 / 50, "channel": "TDL", "model": "A"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": [50e-9, 100e-9], "doppler": [50, 150], "coherence_time": 0.423 / 150, "channel": "Rayleigh"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": [200e-9, 500e-9], "doppler": [500, 2000], "coherence_time": 0.423 / 2000, "channel": "TDL", "model": "C"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": [100e-9, 300e-9], "doppler": [200, 1000], "coherence_time": 0.423 / 1000, "channel": "TDL", "model": "A"},
]

# --- Custom Layers ---
# ... (NormalizedDense and LearnableBeamspace definitions remain the same) ...
class NormalizedDense(tf.keras.layers.Dense):
    def call(self, inputs):
        norm_weights = self.kernel / tf.norm(self.kernel, axis=0, keepdims=True) + 1e-8
        output = tf.matmul(inputs, norm_weights)
        if self.use_bias:
             output = output + self.bias
        return output

class LearnableBeamspace(tf.keras.layers.Layer):
    def __init__(self, num_antennas, trainable=True, use_pruning=True, pruning_threshold=0.05):
        super().__init__()
        self.num_antennas = num_antennas
        self.use_pruning = use_pruning
        self.pruning_threshold = pruning_threshold
        init_dft = tf.signal.fft(tf.eye(num_antennas, dtype=tf.complex64))
        self.transform_matrix = tf.Variable(init_dft, trainable=trainable, name="beamspace_matrix")

    def call(self, h):
        res = tf.einsum('bua,ac->buc', h, self.transform_matrix)
        if self.use_pruning:
            mask = tf.abs(res) > self.pruning_threshold
            res = tf.where(mask, res, tf.zeros_like(res))
        return res


# --- Beamforming Model ---
# --- Beamforming Model ---
class BeamformingMetaAttentionModel(tf.keras.Model):
    # REPLACE THE __init__ METHOD
    def __init__(self, num_antennas, num_users, num_tasks, use_replay=True, use_fisher=True):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.num_tasks = num_tasks
        self.hidden_dim = 128 # Dimension after Conv1D
        self.lambda_reg = 10.0

        self.use_replay = use_replay
        self.use_fisher = use_fisher

        # --- Layers for Conv1D + Per-User MLP Path ---
        initializer = 'glorot_uniform'

        # 1. Conv1D Layer
        self.conv1 = tf.keras.layers.Conv1D(
            filters=self.hidden_dim,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_initializer=initializer,
            name="conv1_k3"
        )
        # 2. LayerNorm after Conv1D
        self.norm = tf.keras.layers.LayerNormalization(name="conv_layer_norm")

        # 3. Per-User MLP layers
        # These operate on the last dimension of [B, U, H] input
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer, name="fc1_per_user")
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer, name="fc2_per_user")
        # Output layer produces A*2 outputs PER USER
        self.out = tf.keras.layers.Dense(num_antennas * 2, kernel_initializer=initializer, name="output_dense_per_user")
        # --- End Layers for This Path ---

        # --- Original/Unused Layers ---
        # ...

        # --- Task-specific Layers (Keep for potential future use, but not called) ---
        # ...

        # --- CL Buffers (unused in single-task) ---
        # ...
        if self.use_replay:
             self.replay_x = [[] for _ in range(num_tasks)]
             self.replay_h = [[] for _ in range(num_tasks)]
             self.replay_limit = 1000
        if self.use_fisher:
             self.old_params = {}
             self.fisher = {}

    # REPLACE THE call METHOD
    def call(self, x, channel_features, task_idx, training=False):
        # x: Input complex channel H [B, U, A]
        # channel_features: SIDE FEATURES ARE IGNORED
        # task_idx: Integer (fixed to 0 in single-task mode)
        batch_size = tf.shape(x)[0]

        # --- Input Processing with Conv1D(k=3) ---
        real_part = tf.math.real(x)
        imag_part = tf.math.imag(x)
        x_split = tf.concat([real_part, imag_part], axis=-1)  # Shape: [B, U, 2*A]
        x_conv_feat = self.conv1(x_split) # Shape: [B, U, hidden_dim]
        x_norm = self.norm(x_conv_feat)   # Shape: [B, U, hidden_dim]

        # --- NO Aggregation ---

        # --- Apply MLP Per User ---
        # Input is x_norm with shape [B, U, hidden_dim]
        x_fc1_out = self.fc1(x_norm)      # Shape: [B, U, 256]
        x_fc2_out = self.fc2(x_fc1_out)   # Shape: [B, U, 128]
        out_real_imag = self.out(x_fc2_out) # Shape: [B, U, A*2]

        # --- Reshape to Complex Beamforming Weights ---
        real_w = out_real_imag[..., :self.num_antennas] # Shape: [B, U, A]
        imag_w = out_real_imag[..., self.num_antennas:] # Shape: [B, U, A]
        w = tf.complex(real_w, imag_w) # Shape: [B, U, A]

        # --- Return NON-NORMALIZED weights ---
        # (Normalization happens inside train_step_single_task for loss/eval)
        return w

    # --- Other methods (generate_synthetic_batch, eval_kpi, CL methods) ---
    # Make sure these are the latest correct versions from previous steps
    # generate_synthetic_batch should be the one that worked without shape errors
    # eval_kpi should be the one corrected for the Mul error
    # CL methods remain unused

    # ...(Paste the LATEST CORRECT versions of generate_synthetic_batch and eval_kpi here)...
    # ...(Paste the CL helper methods here)...
    # --- Data Generation (Should be correct now) ---
    def generate_synthetic_batch(self, task, batch_size=8):
        """ Generates a batch of channel data and features for a given task. """
        h_users = []
        features = []
        for i_user in range(self.num_users):
            delay = np.random.uniform(*task["delay_spread"])
            doppler = np.random.uniform(*task["doppler"])
            snr_db = np.random.uniform(5, 25)
            sampling_freq = int(max(1 / (delay + 1e-9), 2 * doppler)) * 10

            h_user = None
            if task["channel"] == "TDL":
                tdl = TDL( model=task.get("model", "A"), delay_spread=delay, carrier_frequency=FREQ,
                           num_tx_ant=self.num_antennas, num_rx_ant=1, min_speed=task["speed_range"][0],
                           max_speed=task["speed_range"][1] )
                h_time, _ = tdl(batch_size=batch_size, num_time_steps=NUM_SLOTS, sampling_frequency=sampling_freq)
                h_avg_time = tf.reduce_mean(h_time, axis=-1)
                h_comb_paths = tf.reduce_sum(h_avg_time, axis=-1)
                h_user = tf.squeeze(h_comb_paths, axis=[1, 2])
            else: # Rayleigh
                rb = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=self.num_antennas)
                h_block, _ = rb(batch_size=batch_size, num_time_steps=1)
                h_user = tf.squeeze(h_block, axis=[1, 2, 4])

            if h_user is None: raise ValueError("h_user not defined")
            h_user_reshaped = tf.reshape(h_user, [batch_size, 1, self.num_antennas])
            h_users.append(h_user_reshaped)
            f = tf.tile([[doppler, delay, snr_db]], [batch_size, 1])
            features.append(f)

        h_stacked = tf.stack(h_users, axis=1)
        feats_stacked = tf.stack(features, axis=1)

        if tf.rank(h_stacked) == 4 and tf.shape(h_stacked)[2] == 1:
             h_stacked_squeezed = tf.squeeze(h_stacked, axis=2)
        else:
             tf.print("Warning/Info: Shape before squeeze was not [B, U, 1, A]:", tf.shape(h_stacked))
             h_stacked_squeezed = h_stacked

        h_norm = h_stacked_squeezed / (tf.cast(tf.norm(h_stacked_squeezed, axis=-1, keepdims=True), tf.complex64) + 1e-8)

        doppler_mean, doppler_std = 1000, 800; delay_mean, delay_std = 200e-9, 150e-9; snr_mean, snr_std = 15, 7
        means = tf.constant([[[doppler_mean, delay_mean, snr_mean]]], dtype=tf.float32)
        stds = tf.constant([[[doppler_std, delay_std, snr_std]]], dtype=tf.float32)
        feats_norm = (feats_stacked - means) / (stds + 1e-8)

        return tf.identity(h_norm), tf.identity(h_norm), tf.cast(feats_norm, tf.float32)

    # --- Evaluation (CORRECTED FOR BASELINE ERROR) ---
    # --- Evaluation (CORRECTED FOR BASELINE ERROR) ---
    # REPLACE THIS ENTIRE METHOD in your BeamformingMetaAttentionModel class
    def eval_kpi(self, w, h, noise_power_lin=1e-3):
        """ Evaluates SINR and Throughput. Corrected mask casting. """
        batch_size = tf.shape(w)[0]
        num_users = self.num_users

        # Ensure w is normalized (redundant if model output is already normalized)
        w_norm = tf.nn.l2_normalize(w, axis=-1, epsilon=1e-8)

        # Calculate received signal matrix: y_ik = h_i * w_k^H
        signal_matrix = tf.matmul(h, w_norm, adjoint_b=True) # [B, U, U]

        # Desired signal power for user i: |y_ii|^2
        desired_signal = tf.linalg.diag_part(signal_matrix) # [B, U]
        desired_power = tf.abs(desired_signal)**2 # [B, U], Float

        # --- CORRECTED Interference Calculation ---
        # Multiply float powers (|signal_matrix|^2) by float mask directly
        mask = 1.0 - tf.eye(num_users, batch_shape=[batch_size], dtype=tf.float32) # [B, U, U], Float
        interference_power_masked = tf.abs(signal_matrix)**2 * mask # Float * Float -> Float
        interference_power = tf.reduce_sum(tf.math.real(interference_power_masked), axis=-1) # Sum interference, shape [B, U], Float
        # --- End Correction ---

        # SINR calculation (linear)
        sinr_linear = desired_power / (interference_power + noise_power_lin) # [B, U]

        # Average SINR over batch and users (in dB)
        sinr_dB = 10.0 * tf.math.log(tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0) # Scalar

        # Throughput (Shannon capacity, average over batch and users)
        throughput = tf.math.log(1.0 + sinr_linear) / tf.math.log(2.0) # [B, U]
        avg_throughput = tf.reduce_mean(throughput) # Scalar

        # Latency (Placeholder)
        latency_ms = tf.constant(0.0, dtype=tf.float32)

        # Return scalar numpy values
        # Use tf.stop_gradient to be safe if called within a gradient tape context elsewhere
        return tf.stop_gradient(sinr_dB).numpy(), tf.stop_gradient(avg_throughput).numpy(), tf.stop_gradient(latency_ms).numpy()

    # --- CL Methods (update_replay, sample_replay, update_fisher_info, regularization_loss) remain the same ---
    # ... (Keep the CL methods as they were, they are not used when use_replay/use_fisher are False) ...
    def update_replay(self, x, h, task_idx):
        if not self.use_replay: return
        if len(self.replay_x[task_idx]) >= self.replay_limit:
            self.replay_x[task_idx] = self.replay_x[task_idx][self.replay_limit // 2:]
            self.replay_h[task_idx] = self.replay_h[task_idx][self.replay_limit // 2:]
        real = tf.math.real(x); imag = tf.math.imag(x)
        x_split = tf.concat([real, imag], axis=-1)
        self.replay_x[task_idx].extend(tf.unstack(x_split))
        self.replay_h[task_idx].extend(tf.unstack(h))

    def sample_replay(self, task_idx, num_samples):
        if not self.use_replay or len(self.replay_x[task_idx]) < num_samples: return None, None, None
        indices = np.random.choice(len(self.replay_x[task_idx]), size=num_samples, replace=False)
        x_float = tf.stack([self.replay_x[task_idx][i] for i in indices])
        h_complex = tf.stack([self.replay_h[task_idx][i] for i in indices])
        a = self.num_antennas; real_part = x_float[..., :a]; imag_part = x_float[..., a:]
        x_complex = tf.complex(real_part, imag_part)
        dummy_features = tf.zeros_like(tf.math.real(h_complex[..., :3]))
        return x_complex, h_complex, dummy_features

    def update_fisher_info(self, x, h, channel_features, task_idx):
        if not self.use_fisher: return
        with tf.GradientTape() as tape:
            w_pred = self(x, channel_features, task_idx, training=True)
            loss = fisher_sinr_loss(w_pred, h)
        grads = tape.gradient(loss, self.trainable_variables)
        new_fisher = {}
        for v, g in zip(self.trainable_variables, grads):
            if g is not None:
                g_float = tf.cast(tf.square(g), dtype=tf.float32)
                new_fisher[v.name] = self.fisher.get(v.name, 0.0) + g_float
        self.fisher.update(new_fisher)
        self.old_params = {v.name: tf.identity(v) for v in self.trainable_variables}
        print(f"ℹ️ Updated Fisher Information for task {task_idx}. Param count: {len(self.old_params)}")

    def regularization_loss(self):
        if not self.use_fisher or not self.fisher or not self.old_params: return tf.constant(0.0, dtype=tf.float32)
        reg_loss = 0.0
        for v in self.trainable_variables:
            if v.name in self.fisher and v.name in self.old_params:
                fisher_val = tf.cast(self.fisher[v.name], dtype=v.dtype)
                param_diff = v - self.old_params[v.name]
                reg_loss += tf.reduce_sum(fisher_val * tf.square(param_diff))
        return self.lambda_reg * tf.cast(reg_loss, dtype=tf.float32)


# --- Helper Functions (Loss, Benchmarks) ---
def fisher_sinr_loss(w, h, noise_power_lin=1e-3):
    # ... (fisher_sinr_loss remains the same, but needs the corrected interference calc if used) ...
    # Correction: Apply the same fix as in eval_kpi
    num_users = tf.shape(w)[1]
    batch_size = tf.shape(w)[0]
    signal_matrix = tf.matmul(h, w, adjoint_b=True)
    desired_signal = tf.linalg.diag_part(signal_matrix)
    desired_power = tf.abs(desired_signal)**2
    mask = 1.0 - tf.eye(num_users, batch_shape=[batch_size], dtype=tf.float32)
    interference_power_masked = tf.abs(signal_matrix)**2 * mask # Float * Float
    interference_power = tf.reduce_sum(tf.math.real(interference_power_masked), axis=-1)
    sinr = desired_power / (interference_power + noise_power_lin)
    loss = -tf.reduce_mean(tf.math.log(1.0 + sinr + 1e-9)) # Added epsilon
    return loss
# --- Helper Function for ZF Calculation ---
# Place this function OUTSIDE the model class definition
def compute_zf_weights(h, reg=1e-5):
    """
    Computes Zero-Forcing (ZF) precoding weights.

    Args:
        h (tf.Tensor): Channel matrix, shape [B, U, A], complex64.
        reg (float): Regularization factor for matrix inversion stability.

    Returns:
        tf.Tensor: Normalized ZF weights, shape [B, U, A], complex64.
    """
    batch_size = tf.shape(h)[0]
    num_users = tf.shape(h)[1]
    num_antennas = tf.shape(h)[2]

    try:
        # Calculate H^H (Hermitian transpose)
        h_herm = tf.linalg.adjoint(h) # Shape: [B, A, U]

        # Calculate H * H^H
        hh_herm = tf.matmul(h, h_herm) # Shape: [B, U, U]

        # Regularized inverse: (H * H^H + reg * I)^-1
        identity = tf.eye(num_users, batch_shape=[batch_size], dtype=tf.complex64)
        inv_term = tf.linalg.inv(hh_herm + tf.cast(reg, tf.complex64) * identity) # Shape: [B, U, U]

        # Calculate W_unnormalized = H^H * (H * H^H + reg * I)^-1
        w_unnormalized_intermediate = tf.matmul(h_herm, inv_term) # Shape: [B, A, U]

        # Transpose to get per-user weights: [B, U, A]
        w_zf_unnorm = tf.linalg.adjoint(w_unnormalized_intermediate) # Shape: [B, U, A]

        # Normalize weights per user
        w_zf_normalized = tf.nn.l2_normalize(w_zf_unnorm, axis=-1, epsilon=1e-8)

        return tf.stop_gradient(w_zf_normalized) # Return as non-trainable target

    except tf.errors.InvalidArgumentError as e:
         # Handle potential errors during inversion (e.g., matrix not invertible)
         tf.print("Warning: ZF calculation failed, returning random weights.", e, output_stream=sys.stderr)
         # Fallback to random weights if ZF fails
         real = tf.random.normal([batch_size, num_users, num_antennas])
         imag = tf.random.normal([batch_size, num_users, num_antennas])
         w_random = tf.complex(real, imag)
         return tf.stop_gradient(tf.nn.l2_normalize(w_random, axis=-1))
    except Exception as e: # Catch any other unexpected error
         tf.print("Warning: Unexpected error in ZF calculation, returning random weights.", e, output_stream=sys.stderr)
         real = tf.random.normal([batch_size, num_users, num_antennas])
         imag = tf.random.normal([batch_size, num_users, num_antennas])
         w_random = tf.complex(real, imag)
         return tf.stop_gradient(tf.nn.l2_normalize(w_random, axis=-1))
    
# --- Benchmarking Functions (Corrected call to eval_kpi which is now fixed) ---
def compute_mmse(h, noise_power=1e-3, power=1.0):
    start = time.time()
    B, U, A = h.shape
    try:
        alpha_rzf = U * noise_power
        w_precoder = rzf_precoding_matrix(h, alpha=alpha_rzf) # [B, A, U]
        w_mmse_approx = tf.linalg.adjoint(w_precoder) # [B, U, A]
        w_mmse_approx = normalize_precoding_power(w_mmse_approx)
    except Exception as e:
        print(f"Error calculating RZF, using random: {e}")
        real = tf.random.normal([B, U, A]); imag = tf.random.normal([B, U, A])
        w_mmse_approx = tf.complex(real, imag)
        w_mmse_approx = tf.nn.l2_normalize(w_mmse_approx, axis=-1)

    # Use a dummy model to access the CORRECTED eval_kpi
    # Need to define num_tasks used by dummy model constructor
    num_tasks_dummy = 1 # Or len(TASKS)
    dummy_model = BeamformingMetaAttentionModel(NUM_ANTENNAS, NUM_USERS, num_tasks_dummy, False, False)
    sinr_m, thrpt_m, _ = dummy_model.eval_kpi(w_mmse_approx, h, noise_power)
    latency_m = (time.time() - start) * 1000
    return float(sinr_m), float(thrpt_m), latency_m

def compute_random(h, noise_power=1e-3):
    start = time.time()
    B, U, A = h.shape
    real = tf.random.normal([B, U, A]); imag = tf.random.normal([B, U, A])
    w_random = tf.complex(real, imag); w_random = tf.nn.l2_normalize(w_random, axis=-1)
    # Use a dummy model to access the CORRECTED eval_kpi
    num_tasks_dummy = 1 # Or len(TASKS)
    dummy_model = BeamformingMetaAttentionModel(NUM_ANTENNAS, NUM_USERS, num_tasks_dummy, False, False)
    sinr_r, thrpt_r, _ = dummy_model.eval_kpi(w_random, h, noise_power)
    latency_r = (time.time() - start) * 1000
    return float(sinr_r), float(thrpt_r), latency_r


# --- Simplified Training Step (No changes needed here from previous correct version) ---
# --- Simplified Training Step (Single Task, Basic Loss) ---
# REPLACE THIS ENTIRE FUNCTION
# @tf.function # Keep commented out during debugging with tf.print
# --- Simplified Training Step (Single Task, Alignment Loss) ---
# REPLACE THIS ENTIRE FUNCTION
# @tf.function # Keep commented out during debugging
# --- Simplified Training Step (Single Task, Alignment Loss v2) ---
# REPLACE THIS ENTIRE FUNCTION
# @tf.function # Keep commented out during debugging
# --- Simplified Training Step (Single Task, Squared Alignment Loss) ---
# REPLACE THIS ENTIRE FUNCTION
# @tf.function # Keep commented out during debugging
def train_step_single_task(model, optimizer, x, h, channel_features, task_idx, noise_power=1e-3):
    """
    Performs one training step focused on a single task.
    Uses Mean Squared Error (MSE) loss to match Zero-Forcing (ZF) weights.
    Calculates SINR and Alignment (|dot|) for logging purposes only.
    Assumes model.call returns NON-NORMALIZED weights.
    """
    with tf.GradientTape() as tape:
        # 1. Get model's NON-NORMALIZED prediction
        w_pred_raw = model(x, channel_features, task_idx, training=True) # Shape: [B, U, A]

        # 2. Calculate the target ZF weights (normalized)
        w_zf_target = compute_zf_weights(h) # Shape: [B, U, A] (Normalized)

        # 3. Normalize model's prediction for fair comparison in MSE loss
        w_pred_norm = tf.nn.l2_normalize(w_pred_raw, axis=-1, epsilon=1e-8)

        # --- NEW Loss Function: MSE vs ZF ---
        # Calculate MSE between the normalized prediction and normalized target
        # MSE = mean(|w_pred_norm - w_zf_target|^2)
        error = w_pred_norm - w_zf_target
        # Use tf.math.real because square(abs(complex)) = real^2 + imag^2
        total_loss = tf.reduce_mean(tf.math.real(error * tf.math.conj(error)))
        # Alternative: tf.reduce_mean(tf.square(tf.abs(error))) should be equivalent
        # --- End NEW Loss Function ---

    # 4. Calculate and Apply Gradients based on MSE loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    optimizer.apply_gradients(grads_and_vars)

    # --- Calculate Alignment and SINR *only* for logging/monitoring ---
    # Use the *normalized* prediction for these calculations
    w_pred_for_eval = w_pred_norm # Use the normalized weights for eval consistency

    with tf.device('/cpu:0'): # Optional: Move calculations to CPU
        # Alignment Calculation
        h_norm = tf.nn.l2_normalize(h, axis=-1, epsilon=1e-8)
        complex_dots = tf.reduce_sum(h_norm * tf.math.conj(w_pred_for_eval), axis=-1)
        dot_abs_all = tf.abs(complex_dots)

        # SINR Calculation
        signal_matrix = tf.matmul(h, w_pred_for_eval, adjoint_b=True) # Use normalized w for SINR calc too
        desired_signal = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.abs(desired_signal)**2
        mask = 1.0 - tf.eye(model.num_users, batch_shape=[tf.shape(x)[0]], dtype=tf.float32)
        interference_power_masked = tf.abs(signal_matrix)**2 * mask
        interference_power = tf.reduce_sum(tf.math.real(interference_power_masked), axis=-1)
        sinr_linear = desired_power / (interference_power + noise_power)
        sinr_dB_batch_mean = 10.0 * tf.math.log(tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0)
        sinr_per_user_bps = tf.math.log(1.0 + sinr_linear) / tf.math.log(2.0)
    # --- End Logging Calculations ---

    # --- Per-user Diagnostics Logging ---
    log_lines = []
    # Use the calculated metrics for logging
    sinr_user0 = sinr_per_user_bps[0]; dot_abs_user0 = dot_abs_all[0]; complex_dots_user0 = complex_dots[0]
    for u in range(model.num_users):
        complex_dot_u = complex_dots_user0[u]
        dot_real = tf.math.real(complex_dot_u); dot_abs = tf.abs(complex_dot_u)
        angle = tf.math.angle(complex_dot_u); sinr_u_val = sinr_user0[u].numpy()
        # Log based on alignment criterion, even though loss isn't directly alignment
        if (dot_abs < 0.5) or (tf.abs(angle) > np.pi / 4):
            line = f"User {u:02d} | dot={dot_real:.4f} | angle={angle:.4f} rad | |dot|={dot_abs:.4f} | SINR={sinr_u_val:.2f} bps/Hz"
            log_lines.append(line)

    # Return metrics (Loss is now MSE, SINR/Alignment are for monitoring)
    return {
        "total_loss": total_loss,  # This is now MSE vs ZF
        "sinr": sinr_dB_batch_mean, # SINR is just for reporting
        "log_lines": log_lines,
        "mean_dot_abs": tf.reduce_mean(dot_abs_all) # Also return alignment metric
    }
# --- Simplified Training Loop (No changes needed) ---
def training_loop_single_task(model, optimizer, task, task_idx, num_epochs=50, batch_size=8, noise_power=1e-3):
    """ Trains the model on a single specified task. """
    print(f"\n🧠 Training Single Task: {task['name']} (Index: {task_idx}) for {num_epochs} epochs.")
    task_name = task['name']
    open("summary_kpi.log", "w").close()
    open("per_user_diag.log", "w").close()

    for epoch in tqdm(range(num_epochs), desc=f"Training {task_name}"):
        x_batch, h_batch, channel_feats = model.generate_synthetic_batch(task, batch_size)
        metrics = train_step_single_task(model, optimizer, x_batch, h_batch, channel_feats, task_idx, noise_power)

        if (epoch + 1) % 10 == 0:
             sinr_val = metrics['sinr'].numpy(); loss_val = metrics['total_loss'].numpy()
             try:
                 with open("summary_kpi.log", "a") as f:
                     f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] SINR={sinr_val:.2f} dB | Loss={loss_val:.4f}\n")
             except Exception as e: print(f"Error writing to summary_kpi.log: {e}")

        if (epoch + 1) % 20 == 0:
            log_lines = metrics.get("log_lines", [])
            if log_lines:
                 try:
                     with open("per_user_diag.log", "a") as f:
                         f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] ----\n")
                         for line in log_lines: f.write(line + "\n")
                 except Exception as e: print(f"Error writing to per_user_diag.log: {e}")

    print(f"\n✅ Finished training loop for task: {task_name}")


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Configuration for Single Task Training ---
    CHOSEN_TASK_INDEX = 0  # Index of the task to train on (0: Static, 1: Pedestrian, etc.)
    NUM_EPOCHS = 100      # Number of epochs for training test
    LEARNING_RATE = 1e-4 # Default LR to test with
    NOISE_POWER_LIN = 1e-3 # Linear noise power
    BATCH_SIZE_MAIN = 8    # Training batch size

    if CHOSEN_TASK_INDEX >= len(TASKS):
        print(f"❌ Error: CHOSEN_TASK_INDEX ({CHOSEN_TASK_INDEX}) is out of bounds for TASKS list (size {len(TASKS)}).")
        sys.exit(1)

    single_task = TASKS[CHOSEN_TASK_INDEX]
    print(f"--- Starting Single Task Training ---")
    print(f"Task: {single_task['name']} (Index: {CHOSEN_TASK_INDEX})")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE_MAIN}")
    print(f"Noise Power (Linear): {NOISE_POWER_LIN}")
    print("-" * 36) # Use 36 for consistency

    # --- Optional: Baseline Calculation (MMSE/Random) ---
    print("\n📊 Calculating Baselines for the chosen task...")
    try:
        # Instantiate a temporary model just for data generation for baselines
        temp_model_for_data = BeamformingMetaAttentionModel(NUM_ANTENNAS, NUM_USERS, len(TASKS), False, False)
        # Generate data using the CORRECTED generate_synthetic_batch method
        x_base, h_base, feats_base = temp_model_for_data.generate_synthetic_batch(single_task, batch_size=32) # Larger batch for stable baseline
        # Compute baselines using the CORRECTED eval_kpi method (called internally by compute_*)
        sinr_r, thrpt_r, lat_r = compute_random(h_base, NOISE_POWER_LIN)
        sinr_m, thrpt_m, lat_m = compute_mmse(h_base, NOISE_POWER_LIN) # Contains RZF approximation
        print(f"🎲 Random Beamforming → SINR: {sinr_r:.2f} dB | Thrpt: {thrpt_r:.4f} bps/Hz")
        print(f"🎯 RZF/MMSE Approx. → SINR: {sinr_m:.2f} dB | Thrpt: {thrpt_m:.4f} bps/Hz")
    except Exception as e:
        print(f"⚠️ Could not compute baselines: {e}")
    print("-" * 30)


    # --- Model and Optimizer Initialization ---
    model = BeamformingMetaAttentionModel(
        num_antennas=NUM_ANTENNAS, num_users=NUM_USERS, num_tasks=len(TASKS),
        use_replay=False, use_fisher=False ) # Keep CL flags False for single task
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # --- Build Model and Optimizer ---
    print("Building model...")
    # Use shapes consistent with the model input
    dummy_h = tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64)
    dummy_feats = tf.zeros((1, NUM_USERS, 3), dtype=tf.float32) # Assuming 3 features
    try:
        # Call model once to build layers
        _ = model(dummy_h, dummy_feats, task_idx=CHOSEN_TASK_INDEX, training=False)
        # Build optimizer explicitly
        optimizer.build(model.trainable_variables)
        print("✅ Model and Optimizer built successfully.")
    except Exception as e:
        print(f"❌ Error building model/optimizer: {e}")
        sys.exit(1)

    # ***** CORRECT INSERTION POINT FOR ZF VERIFICATION *****
    # --- Verification Step: Check ZF Weights Alignment ---
    print("\n🔬 Verifying ZF Weight Calculation...")
    try:
        # Generate a small batch
        zf_check_batch_size = 1 # Use batch size 1 for simplicity
        # Re-use the main model instance now that it's built
        _, h_check_zf, _ = model.generate_synthetic_batch(single_task, batch_size=zf_check_batch_size)

        # Calculate ZF weights for this batch
        # Ensure compute_zf_weights function is defined globally or imported
        w_zf_check = compute_zf_weights(h_check_zf) # Shape: [B, U, A]

        # Check alignment for the first sample in the batch (B=0)
        h_sample = h_check_zf[0]   # Shape: [U, A]
        w_zf_sample = w_zf_check[0] # Shape: [U, A]

        tf.print("Checking ZF alignment for User 0 to", NUM_USERS-1, ":", output_stream=sys.stdout)
        total_alignment = 0.0
        all_users_aligned = True
        for u in range(NUM_USERS):
            h_user = h_sample[u] # Shape: [A]
            w_user = w_zf_sample[u] # Shape: [A]

            # Normalize both vectors
            h_user_norm = tf.nn.l2_normalize(h_user, epsilon=1e-8)
            w_user_norm = tf.nn.l2_normalize(w_user, epsilon=1e-8)

            # Calculate dot product: sum(h_norm * conj(w_norm))
            complex_dot = tf.reduce_sum(h_user_norm * tf.math.conj(w_user_norm))

            # Magnitude (|dot|) should be close to 1.0 if aligned
            alignment_magnitude = tf.abs(complex_dot)
            alignment_np = alignment_magnitude.numpy() # Convert to numpy for comparison
            total_alignment += alignment_np

            tf.print("  User", u, "| Alignment |h_norm^H w_zf_norm|:", alignment_magnitude, output_stream=sys.stdout)
            if alignment_np < 0.8: # Check numerical value
                 all_users_aligned = False

        avg_alignment = total_alignment / NUM_USERS
        print(f"  Average ZF Alignment Magnitude: {avg_alignment:.4f}")
        if not all_users_aligned:
             print("  ⚠️ Warning: Average or per-user ZF alignment seems low. Check compute_zf_weights function.")
        else:
             print("  ✅ ZF weights seem reasonably aligned with channels.")

    except Exception as e:
        print(f"❌ Error during ZF weight verification: {e}")
        # Optionally re-raise or exit if ZF verification failure is critical
        # raise e
    print("-" * 30)
    # --- End Verification Step ---
    # ***** END OF INSERTED CODE *****

    # --- Warm-up / Sanity Check (Optional but recommended) ---
    print("\n performing sanity check forward pass...")
    try:
        # Use the main model instance here too
        x_check, h_check, feats_check = model.generate_synthetic_batch(single_task, batch_size=2)
        w_check = model(x_check, feats_check, task_idx=CHOSEN_TASK_INDEX, training=False)
        norm_w = tf.reduce_mean(tf.norm(w_check, axis=-1)) # Model output w is non-normalized now
        norm_h = tf.reduce_mean(tf.norm(h_check, axis=-1)) # h_check is normalized
        # tf.print("Sanity Check: Avg Norm(w) =", norm_w, "| Avg Norm(h) =", norm_h) # Norm(w) won't be 1
        print(f"Sanity Check: Avg Norm(h) = {norm_h.numpy():.4f}") # Just check h norm
        print("✅ Sanity check forward pass completed.")
    except Exception as e:
        print(f"⚠️ Error during sanity check: {e}")

    # --- Start Single Task Training Loop ---
    # Ensure train_step_single_task is using the desired loss (e.g., MSE vs ZF)
    training_loop_single_task(
        model=model, optimizer=optimizer, task=single_task, task_idx=CHOSEN_TASK_INDEX,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE_MAIN, noise_power=NOISE_POWER_LIN
    )

    print("\n--- Single Task Training Finished ---")