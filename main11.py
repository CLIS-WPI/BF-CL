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
        # Use only the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Allow memory growth to avoid allocating all memory at once
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ Using only GPU:0 -> {gpus[0].name}")
    except RuntimeError as e:
        print(f"❌ Error setting GPU visibility: {e}")
else:
    print("ℹ️ No GPU found, using CPU.")

# --- Settings ---
NUM_ANTENNAS = 64
FREQ = 28e9
NUM_SLOTS = 10  # Used in channel generation
BATCH_SIZE = 4  # Reduced batch size for potentially easier debugging
NUM_USERS = 6
# USER_DURATION = 2  # hours (Not directly used in training loop)

# --- Environment Simulation Parameters (Not directly used in simplified training) ---
# ARRIVAL_RATES = {"morning": 50, "noon": 75, "evening": 60}
# PERIOD_HOURS = {"morning": 8, "noon": 4, "evening": 12}
# DAILY_COMPOSITION = {
#     "morning": {"Pedestrian": 0.40, "Static": 0.30, "Vehicular": 0.20, "Aerial": 0.10},
#     "noon": {"Vehicular": 0.50, "Pedestrian": 0.20, "Static": 0.20, "Aerial": 0.10},
#     "evening": {"Aerial": 0.30, "Vehicular": 0.30, "Pedestrian": 0.20, "Static": 0.20}
# }

# --- Task Definitions ---
TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": [30e-9, 50e-9], "doppler": [10, 50], "coherence_time": 0.423 / 50, "channel": "TDL", "model": "A"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": [50e-9, 100e-9], "doppler": [50, 150], "coherence_time": 0.423 / 150, "channel": "Rayleigh"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": [200e-9, 500e-9], "doppler": [500, 2000], "coherence_time": 0.423 / 2000, "channel": "TDL", "model": "C"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": [100e-9, 300e-9], "doppler": [200, 1000], "coherence_time": 0.423 / 1000, "channel": "TDL", "model": "A"},
]

# --- Custom Layers ---
class NormalizedDense(tf.keras.layers.Dense):
    """ Dense layer with kernel normalized along the output axis. """
    # Note: Normalization happens during layer call, might affect training dynamics.
    def call(self, inputs):
        # Normalize weights along the output feature dimension (axis=1 for kernel shape [in, out])
        # Or axis=0 if kernel shape is [out, in] - Check Dense layer implementation detail
        # tf.keras.layers.Dense kernel shape is [input_dim, output_dim]
        norm_weights = self.kernel / tf.norm(self.kernel, axis=0, keepdims=True) + 1e-8 # Add epsilon for stability
        output = tf.matmul(inputs, norm_weights)
        if self.use_bias:
             output = output + self.bias
        # No activation function here by default, relies on subsequent layers
        return output

class LearnableBeamspace(tf.keras.layers.Layer):
    """ Learns a transformation matrix initialized with DFT. """
    def __init__(self, num_antennas, trainable=True, use_pruning=True, pruning_threshold=0.05):
        super().__init__()
        self.num_antennas = num_antennas
        self.use_pruning = use_pruning
        self.pruning_threshold = pruning_threshold
        # Initialize with DFT matrix
        init_dft = tf.signal.fft(tf.eye(num_antennas, dtype=tf.complex64))
        self.transform_matrix = tf.Variable(init_dft, trainable=trainable, name="beamspace_matrix")

    def call(self, h):
        # h shape: [B, U, A]
        # transform_matrix shape: [A, A]
        # einsum: batch, user, antenna ; antenna, beam_coeff -> batch, user, beam_coeff
        res = tf.einsum('bua,ac->buc', h, self.transform_matrix)
        if self.use_pruning:
            # Prune small values based on magnitude
            mask = tf.abs(res) > self.pruning_threshold
            res = tf.where(mask, res, tf.zeros_like(res))
        return res

# --- Beamforming Model ---
class BeamformingMetaAttentionModel(tf.keras.Model):
    def __init__(self, num_antennas, num_users, num_tasks, use_replay=True, use_fisher=True):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.num_tasks = num_tasks
        self.hidden_dim = 128
        self.lambda_reg = 10.0 # EWC Regularization strength (Not used in single-task mode)

        # --- CL Flags (Set to False for single-task training) ---
        self.use_replay = use_replay
        self.use_fisher = use_fisher

        # --- Shared Layers ---
        self.concat_proj = tf.keras.layers.Dense(self.hidden_dim, name="concat_proj")
        self.feat_proj = tf.keras.layers.Dense(64, activation='relu', name="feat_proj")
        self.conv1 = tf.keras.layers.Conv1D(64, kernel_size=1, activation='relu', name="conv1") # Kernel size 1 acts like a Dense layer per user antenna data
        self.mha_shared = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64, name="mha_shared")
        self.norm = tf.keras.layers.LayerNormalization(name="layer_norm")
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', name="fc1")
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', name="fc2")
        # Output layer: Predicts real and imaginary parts for all users' antennas
        self.out = tf.keras.layers.Dense(num_antennas * num_users * 2, name="output_dense") # Use standard Dense for simplicity first
        # self.out = NormalizedDense(num_antennas * num_users * 2, name="output_normalized") # Optional: Try NormalizedDense later

        self.x_feat_proj_layer = tf.keras.layers.Dense(self.hidden_dim, name="x_feat_proj")

        # --- Optional: Learnable beamspace ---
        # Set trainable=False to use fixed DFT initially
        self.beamspace = LearnableBeamspace(num_antennas, trainable=False, use_pruning=False)

        # --- Task-specific Layers (Still created but only one path used in single-task) ---
        for i in range(num_tasks):
            setattr(self, f"gru_task_{i}", tf.keras.layers.GRU(self.hidden_dim, return_sequences=True, name=f"gru_task_{i}"))
            setattr(self, f"gate_task_{i}", tf.keras.layers.Dense(self.hidden_dim, activation="sigmoid", name=f"gate_task_{i}"))
        # Build task-specific layers with dummy input
        dummy_input_gru = tf.random.normal([1, num_users, self.hidden_dim])
        dummy_input_gate = tf.random.normal([1, num_users, self.hidden_dim])
        for i in range(num_tasks):
                _ = getattr(self, f"gru_task_{i}")(dummy_input_gru)
                _ = getattr(self, f"gate_task_{i}")(dummy_input_gate)

        # --- CL Buffers (Only initialized if flags are True) ---
        if self.use_replay:
            self.replay_x = [[] for _ in range(num_tasks)]
            self.replay_h = [[] for _ in range(num_tasks)]
            self.replay_limit = 1000 # Max samples per task in replay buffer

        if self.use_fisher:
            self.old_params = {} # Store parameters after task training
            self.fisher = {}     # Store Fisher information (squared gradients)

    def call(self, x, channel_features, task_idx, training=False):
        # x: Input, assumed to be complex channel H [B, U, A] for now
        # channel_features: Side information like Doppler, delay, SNR [B, U, F]
        # task_idx: Integer indicating the current task (fixed to 0 in single-task mode)
        batch_size = tf.shape(x)[0]

        # --- Feature Processing ---
        # Apply optional beamspace transformation
        x_beamspace = self.beamspace(x)

        # Project channel side features
        channel_feat_proj = self.feat_proj(channel_features)  # [B, U, 64]

        # Convert complex channel input to real representation [Real, Imag]
        real_part = tf.math.real(x_beamspace)
        imag_part = tf.math.imag(x_beamspace)
        x_split = tf.concat([real_part, imag_part], axis=-1)  # [B, U, 2*A]

        # Process channel representation (e.g., with Conv1D)
        x_conv_feat = self.conv1(x_split)  # [B, U, 64]

        # Concatenate processed channel and side features
        x_combined_feat = tf.concat([x_conv_feat, channel_feat_proj], axis=-1)  # [B, U, 64+64=128]
        x_projected_feat = self.concat_proj(x_combined_feat) # [B, U, hidden_dim]

        # --- Shared Attention ---
        # Use projected features as query and key/value for self-attention among users
        x_attn = self.mha_shared(query=x_projected_feat, value=x_projected_feat, key=x_projected_feat, training=training) # [B, U, hidden_dim]

        # --- Task-Specific Path (Selects based on task_idx) ---
        # In single-task mode, task_idx is fixed, so the same GRU and Gate are always used.
        gru_layer = getattr(self, f"gru_task_{task_idx}")
        gate_layer = getattr(self, f"gate_task_{task_idx}")

        gru_out = gru_layer(x_attn, training=training)  # Pass training flag to GRU
        gate = gate_layer(x_attn)
        x_gated = tf.multiply(gru_out, gate) # Element-wise multiplication

        # --- Residual Connection & Normalization ---
        x_residual_proj = self.x_feat_proj_layer(x_projected_feat) # Project original features for residual connection
        x_norm = self.norm(x_gated + x_residual_proj) # Add residual and normalize [B, U, hidden_dim]

        # --- Output MLP ---
        # Aggregate user information (e.g., mean pooling)
        x_flat = tf.reduce_mean(x_norm, axis=1)  # [B, hidden_dim]
        x_fc1_out = self.fc1(x_flat)
        x_fc2_out = self.fc2(x_fc1_out)      # [B, 128]
        out_real_imag = self.out(x_fc2_out)  # [B, U*A*2]

        # --- Reshape to Complex Beamforming Weights ---
        real_w = out_real_imag[:, :self.num_users * self.num_antennas]
        imag_w = out_real_imag[:, self.num_users * self.num_antennas:]

        # Reshape real and imaginary parts
        real_w_reshaped = tf.reshape(real_w, [batch_size, self.num_users, self.num_antennas])
        imag_w_reshaped = tf.reshape(imag_w, [batch_size, self.num_users, self.num_antennas])

        # Combine into complex weights
        w = tf.complex(real_w_reshaped, imag_w_reshaped) # [B, U, A]

        # --- Final Normalization (Per User) ---
        # Ensure each user's beamforming vector has unit norm
        w_normalized = tf.nn.l2_normalize(w, axis=-1, epsilon=1e-8) # Normalize along the antenna axis

        # Debug print for norm (might cause issues in graph mode)
        # norm_mean = tf.reduce_mean(tf.norm(w_normalized, axis=-1))
        # tf.print("📡 Norm(w) =", norm_mean, output_stream=sys.stdout)

        return w_normalized

    # --- CL Helper Methods (Not used in single-task mode) ---
    def update_replay(self, x, h, task_idx):
        if not self.use_replay: return
        # ... (Implementation remains but won't be called if use_replay=False) ...
        if len(self.replay_x[task_idx]) >= self.replay_limit:
            # Simple FIFO replacement
            self.replay_x[task_idx] = self.replay_x[task_idx][self.replay_limit // 2:]
            self.replay_h[task_idx] = self.replay_h[task_idx][self.replay_limit // 2:]

        real = tf.math.real(x)
        imag = tf.math.imag(x)
        x_split = tf.concat([real, imag], axis=-1)
        self.replay_x[task_idx].extend(tf.unstack(x_split)) # Store as float32
        self.replay_h[task_idx].extend(tf.unstack(h))       # Store complex h

    def sample_replay(self, task_idx, num_samples):
        if not self.use_replay or len(self.replay_x[task_idx]) < num_samples:
            return None, None

        # Simple random sampling for now
        indices = np.random.choice(len(self.replay_x[task_idx]), size=num_samples, replace=False)

        x_float = tf.stack([self.replay_x[task_idx][i] for i in indices]) # [B, U, 2*A]
        h_complex = tf.stack([self.replay_h[task_idx][i] for i in indices]) # [B, U, A]

        # Convert float x back to complex
        a = self.num_antennas
        real_part = x_float[..., :a]
        imag_part = x_float[..., a:]
        x_complex = tf.complex(real_part, imag_part) # [B, U, A]

        # Return dummy channel features for replay samples for now
        dummy_features = tf.zeros_like(tf.math.real(h_complex[..., :3])) # Shape [B, U, 3]

        return x_complex, h_complex, dummy_features


    def update_fisher_info(self, x, h, channel_features, task_idx):
        # Note: Uses its own SINR loss for Fisher calculation
        if not self.use_fisher: return
        with tf.GradientTape() as tape:
            # Use model's prediction for Fisher loss calculation
            w_pred = self(x, channel_features, task_idx, training=True)
            # Calculate loss (e.g., negative log SINR based on model's prediction)
            loss = fisher_sinr_loss(w_pred, h) # Use the dedicated Fisher loss function

        grads = tape.gradient(loss, self.trainable_variables)
        new_fisher = {}
        for v, g in zip(self.trainable_variables, grads):
            if g is not None:
                # Ensure Fisher is float32
                g_float = tf.cast(tf.square(g), dtype=tf.float32)
                if v.name in self.fisher:
                     # Accumulate Fisher info (can decay older info if needed)
                    new_fisher[v.name] = self.fisher[v.name] + g_float
                else:
                    new_fisher[v.name] = g_float
        self.fisher.update(new_fisher) # Update the stored Fisher info

        # Store current parameters after processing the task
        self.old_params = {v.name: tf.identity(v) for v in self.trainable_variables}
        print(f"ℹ️ Updated Fisher Information for task {task_idx}. Param count: {len(self.old_params)}")


    def regularization_loss(self):
        if not self.use_fisher or not self.fisher or not self.old_params:
             return tf.constant(0.0, dtype=tf.float32)

        reg_loss = 0.0
        for v in self.trainable_variables:
            if v.name in self.fisher and v.name in self.old_params:
                # Ensure compatible types (e.g., both float32)
                fisher_val = tf.cast(self.fisher[v.name], dtype=v.dtype) # Cast fisher to param type
                param_diff = v - self.old_params[v.name]
                reg_loss += tf.reduce_sum(fisher_val * tf.square(param_diff))

        return self.lambda_reg * tf.cast(reg_loss, dtype=tf.float32) # Ensure final loss is float32


    # --- Data Generation ---
    def generate_synthetic_batch(self, task, batch_size=8):
            """ Generates a batch of channel data and features for a given task. """
            h_users = []
            features = []
            for _ in range(self.num_users):
                # Sample task-specific parameters
                delay = np.random.uniform(*task["delay_spread"])
                doppler = np.random.uniform(*task["doppler"])
                snr_db = np.random.uniform(5, 25) # Example: Add SNR as a feature
                # Estimate sampling frequency based on Doppler/delay (simplified)
                # A higher sampling frequency captures faster fading better.
                sampling_freq = int(max(1 / (delay + 1e-9), 2 * doppler)) * 10 # Heuristic multiplier

                if task["channel"] == "TDL":
                # TDL object created correctly
                tdl = TDL(
                    model=task.get("model", "A"),
                    delay_spread=delay,
                    carrier_frequency=FREQ,
                    num_tx_ant=self.num_antennas,
                    num_rx_ant=1, # Single antenna user
                    min_speed=task["speed_range"][0],
                    max_speed=task["speed_range"][1]
                )
                # Generate channel impulse responses over time
                # Assumed shape: [B, 1, 1, A, P, T] (P=num_paths, T=num_time_steps)
                h_time, _ = tdl(batch_size=batch_size, num_time_steps=NUM_SLOTS, sampling_frequency=sampling_freq)

                # --- Correction Start ---
                # 1. Average over time dimension (axis=-1)
                h_avg_time = tf.reduce_mean(h_time, axis=-1) # Shape -> [B, 1, 1, A, P]

                # 2. Combine paths coherently (sum over path dimension, axis=-2)
                h_comb_paths = tf.reduce_sum(h_avg_time, axis=-2) # Shape -> [B, 1, 1, A]

                # 3. Squeeze unnecessary Rx/Tx dimensions (axes 1 and 2)
                h_user = tf.squeeze(h_comb_paths, axis=[1, 2]) # Shape -> [B, A]
                # --- Correction End ---

            else: # Rayleigh
                rb = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=self.num_antennas)
                # Assumed shape: [B, 1, 1, A, 1] (T=1)
                h_block, _ = rb(batch_size=batch_size, num_time_steps=1)
                # Squeeze singleton dimensions for Rx, RxAnt, Time
                h_user = tf.squeeze(h_block, axis=[1, 2, 4]) # Shape -> [B, A] (This part was likely correct)

            # Reshape for stacking: [B, 1, A]
            # Now h_user should have the correct shape [B, A] before reshape
            h_user_reshaped = tf.reshape(h_user, [batch_size, 1, self.num_antennas])
            h_users.append(h_user_reshaped)

                # Generate features: [B, 3] (Doppler, Delay, SNR_dB) - ensure consistent shape
                f = tf.tile([[doppler, delay, snr_db]], [batch_size, 1]) # Tile features for batch
                features.append(f)

            # Stack user channels and features
            h_stacked = tf.stack(h_users, axis=1)  # [B, U, A]
            feats_stacked = tf.stack(features, axis=1)  # [B, U, 3]

            # --- Normalization ---
            # Normalize channel per user for consistent magnitude (optional, but often helpful)
            h_norm = h_stacked / (tf.cast(tf.norm(h_stacked, axis=-1, keepdims=True), tf.complex64) + 1e-8)
            # Normalize features (e.g., using standardization) - IMPORTANT for NN performance
            # Calculate mean/stddev based on typical ranges or a sample dataset
            # Example simple scaling (adjust based on actual ranges):
            doppler_mean, doppler_std = 1000, 800 # Placeholder values
            delay_mean, delay_std = 200e-9, 150e-9 # Placeholder values
            snr_mean, snr_std = 15, 7 # Placeholder values
            means = tf.constant([[[doppler_mean, delay_mean, snr_mean]]], dtype=tf.float32) # Shape [1, 1, 3]
            stds = tf.constant([[[doppler_std, delay_std, snr_std]]], dtype=tf.float32) # Shape [1, 1, 3]
            feats_norm = (feats_stacked - means) / (stds + 1e-8)

            # Return normalized channel (used as 'x' and 'h') and normalized features
            # Using h_norm for both x and h input based on previous assumption
            return tf.identity(h_norm), tf.identity(h_norm), tf.cast(feats_norm, tf.float32)
    # --- Evaluation ---
    def eval_kpi(self, w, h, noise_power_lin=1e-3):
        """ Evaluates SINR and Throughput for given weights and channels. """
        # w: beamforming weights [B, U, A]
        # h: channel [B, U, A]
        batch_size = tf.shape(w)[0]
        num_users = self.num_users

        # Ensure w is normalized (redundant if model output is already normalized)
        w_norm = tf.nn.l2_normalize(w, axis=-1, epsilon=1e-8)

        # Calculate received signal matrix: y_ij = h_i^H * w_j
        # h_adj = tf.linalg.adjoint(h) # Adjoint/Hermitian transpose [B, A, U] - WRONG shape for matmul
        # We need h_i^H w_j. Let's compute signal for each user i.
        # signal = h_i * w_i^H (element-wise product then sum) ? No.
        # signal = matmul(h, w, adjoint_b=True) ? -> [B, U, U] y_ik = sum_a h_ia * conj(w_ka) <= Correct

        signal_matrix = tf.matmul(h, w_norm, adjoint_b=True) # [B, U, U]

        # Desired signal power for user i: |h_i^H w_i|^2 = |y_ii|^2
        desired_signal = tf.linalg.diag_part(signal_matrix) # [B, U]
        desired_power = tf.abs(desired_signal)**2 # [B, U]

        # Interference power for user i: sum_{j!=i} |h_i^H w_j|^2 = sum_{j!=i} |y_ij|^2
        # Create mask to zero out diagonal elements
        mask = 1.0 - tf.eye(num_users, batch_shape=[batch_size], dtype=tf.float32) # [B, U, U]
        interference_power_matrix = tf.abs(signal_matrix)**2 * tf.cast(mask, dtype=tf.complex64) # [B, U, U]
        interference_power = tf.reduce_sum(tf.math.real(interference_power_matrix), axis=-1) # Sum interference from other users [B, U]

        # SINR calculation (linear)
        sinr_linear = desired_power / (interference_power + noise_power_lin) # [B, U]

        # Average SINR over batch and users (in dB)
        sinr_dB = 10.0 * tf.math.log(tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0) # Scalar

        # Throughput (Shannon capacity, average over batch and users)
        # log2(1 + SINR)
        throughput = tf.math.log(1.0 + sinr_linear) / tf.math.log(2.0) # [B, U]
        avg_throughput = tf.reduce_mean(throughput) # Scalar

        # Latency (Placeholder - not meaningful here)
        latency_ms = tf.constant(0.0, dtype=tf.float32) # Replace with actual measurement if needed

        # Return scalar numpy values
        return sinr_dB.numpy(), avg_throughput.numpy(), latency_ms.numpy()

# --- Helper Functions (Loss, Benchmarks) ---
def fisher_sinr_loss(w, h, noise_power_lin=1e-3):
    """ SINR-based loss for Fisher Information calculation. Target: Maximize SINR. """
    # w: predicted beamforming weights [B, U, A]
    # h: channel [B, U, A]
    num_users = tf.shape(w)[1]
    batch_size = tf.shape(w)[0]

    signal_matrix = tf.matmul(h, w, adjoint_b=True) # [B, U, U], y_ik = h_i * w_k^H

    desired_signal = tf.linalg.diag_part(signal_matrix) # [B, U]
    desired_power = tf.abs(desired_signal)**2 # [B, U]

    mask = 1.0 - tf.eye(num_users, batch_shape=[batch_size], dtype=tf.float32) # [B, U, U]
    interference_power_matrix = tf.abs(signal_matrix)**2 * tf.cast(mask, dtype=tf.complex64) # [B, U, U]
    interference_power = tf.reduce_sum(tf.math.real(interference_power_matrix), axis=-1) # [B, U]

    sinr = desired_power / (interference_power + noise_power_lin) # [B, U]

    # Loss is negative log SINR (or log(1+SINR)) to maximize SINR
    loss = -tf.reduce_mean(tf.math.log(1.0 + sinr))
    return loss

# --- Benchmarking Functions (MMSE, Random) --- Not strictly needed for single-task debug but kept for reference ---
def compute_mmse(h, noise_power=1e-3, power=1.0):
    """ Computes MMSE beamforming and evaluates performance. """
    start = time.time()
    B, U, A = h.shape
    # MMSE precoder: P = H^H (H H^H + noise/signal * I)^-1
    # Or ZF: P = H^H (H H^H)^-1
    # Assuming noise_power is variance sigma^2. MMSE uses sigma^2/P per user? Let's assume noise_power includes P scaling.
    # Or simply noise variance relative to normalized signal power (P=1).

    H_herm = tf.linalg.adjoint(h) # [B, A, U]
    HH_herm = tf.matmul(h, H_herm) # [B, U, U] -- This is wrong dimension for standard MMSE precoder calc

    # Let's try the formulation W = (H^H H + U/rho * I)^-1 H^H where rho = P/sigma^2 (SNR)
    # Or simpler: W = H^H (H H^H + sigma^2 I_U)^-1 -- Check Sionna docs if needed
    # Let's use RZF as an approximation/alternative to MMSE as CBF/RZF are already imported
    # alpha_rzf = noise_power # Or related to noise power
    # w_rzf = tf.linalg.adjoint(rzf_precoding_matrix(h, alpha=alpha_rzf)) # [B, U, A] <= Assuming rzf output is [B, A, U]

    # Re-implementing basic MMSE receiver logic (treating BS as Rx, Users as Tx)
    # W = (H H^H + sigma^2 I)^-1 H ? No, that's MMSE equalizer
    # Precoding: W_MMSE = beta * (H^H H + tr(sigma^2 I) / P * I_A)^-1 H^H ? Complex...

    # --- Using Sionna's RZF as a stand-in for MMSE/regularized beamformer ---
    try:
        # Assuming rzf_precoding_matrix returns shape [batch_size, num_tx_ant, num_streams] = [B, A, U]
        # Need to adjust noise regularization parameter 'alpha'
        # Alpha in RZF is often U * noise_power / total_power. Assuming total_power=1.
        alpha_rzf = U * noise_power
        w_precoder = rzf_precoding_matrix(h, alpha=alpha_rzf) # Shape [B, A, U]
        # We need weights per user for transmission: [B, U, A]
        w_mmse_approx = tf.linalg.adjoint(w_precoder) # Shape [B, U, A]
        w_mmse_approx = normalize_precoding_power(w_mmse_approx) # Normalize power

    except Exception as e:
         print(f"Error calculating RZF, using random: {e}")
         # Fallback to random if RZF fails
         real = tf.random.normal([B, U, A])
         imag = tf.random.normal([B, U, A])
         w_mmse_approx = tf.complex(real, imag)
         w_mmse_approx = tf.nn.l2_normalize(w_mmse_approx, axis=-1)

    # Evaluate performance
    # Need a consistent eval function. Let's use the model's eval_kpi
    # Instantiate a dummy model just to call eval_kpi
    dummy_model = BeamformingMetaAttentionModel(NUM_ANTENNAS, NUM_USERS, len(TASKS), False, False)
    sinr_m, thrpt_m, _ = dummy_model.eval_kpi(w_mmse_approx, h, noise_power)
    latency_m = (time.time() - start) * 1000  # ms

    return float(sinr_m), float(thrpt_m), latency_m

def compute_random(h, noise_power=1e-3):
    """ Computes random beamforming performance. """
    start = time.time()
    B, U, A = h.shape
    real = tf.random.normal([B, U, A])
    imag = tf.random.normal([B, U, A])
    w_random = tf.complex(real, imag)
    w_random = tf.nn.l2_normalize(w_random, axis=-1) # Normalize per user

    # Evaluate performance
    dummy_model = BeamformingMetaAttentionModel(NUM_ANTENNAS, NUM_USERS, len(TASKS), False, False)
    sinr_r, thrpt_r, _ = dummy_model.eval_kpi(w_random, h, noise_power)
    latency_r = (time.time() - start) * 1000  # ms

    return float(sinr_r), float(thrpt_r), latency_r


# --- Simplified Training Step (Single Task, Basic Loss) ---
# @tf.function # Optional: Decorate with tf.function for potential speedup after debugging
def train_step_single_task(model, optimizer, x, h, channel_features, task_idx, noise_power=1e-3):
    """
    Performs one training step focused on a single task.
    Uses a simplified loss: Maximizing SINR based on model prediction `w_pred`.
    CL mechanisms (Replay, EWC, Teacher) are disabled here.
    """
    with tf.GradientTape() as tape:
        # 1. Get model's prediction
        # Input 'x' is assumed to be channel 'h' based on generate_synthetic_batch
        w_pred = model(x, channel_features, task_idx, training=True) # task_idx is fixed

        # 2. Calculate SINR based *only* on w_pred
        # h: [B, U, A], w_pred: [B, U, A]
        signal_matrix = tf.matmul(h, w_pred, adjoint_b=True) # [B, U, U], y_ik = h_i * w_k^H

        desired_signal = tf.linalg.diag_part(signal_matrix) # [B, U]
        desired_power = tf.abs(desired_signal)**2 # [B, U]

        # Interference Mask
        mask = 1.0 - tf.eye(model.num_users, batch_shape=[tf.shape(x)[0]], dtype=tf.float32) # [B, U, U]
        # Calculate interference power
        interference_power_matrix = tf.abs(signal_matrix)**2 * tf.cast(mask, dtype=tf.complex64)
        interference_power = tf.reduce_sum(tf.math.real(interference_power_matrix), axis=-1) # [B, U]

        # Linear SINR
        sinr_linear = desired_power / (interference_power + noise_power) # [B, U]

        # 3. Calculate Loss: Maximize SINR = Minimize Negative Log(1 + SINR)
        # Add small epsilon for numerical stability inside log
        total_loss = -tf.reduce_mean(tf.math.log(1.0 + sinr_linear + 1e-9))

    # 4. Calculate and Apply Gradients
    grads = tape.gradient(total_loss, model.trainable_variables)
    # Optional: Gradient Clipping
    # grads, _ = tf.clip_by_global_norm(grads, 1.0)
    grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]

    # Debug: Print gradient norms for key layers (optional)
    # for g, v in grads_and_vars:
    #     if "output_dense/kernel" in v.name or "gru_task_0/gru_cell/kernel" in v.name: # Check specific layers
    #          tf.print("📉 Grad norm:", v.name, tf.norm(g), output_stream=sys.stdout)

    optimizer.apply_gradients(grads_and_vars)

    # 5. Calculate Metrics for Logging
    # Average SINR over batch in dB
    sinr_dB_batch_mean = 10.0 * tf.math.log(tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0)
    # Per-user SINR in bps/Hz for diagnostics log
    sinr_per_user_bps = tf.math.log(1.0 + sinr_linear) / tf.math.log(2.0) # [B, U]

    # --- Per-user Diagnostics Logging ---
    log_lines = []
    # Log diagnostics for the first sample in the batch
    w0 = w_pred[0] # [U, A]
    h0 = h[0]     # [U, A]
    sinr_user0 = sinr_per_user_bps[0] # [U]

    for u in range(model.num_users):
        w_u = tf.nn.l2_normalize(w0[u], axis=-1) # Ensure unit norm for dot product calc
        h_u = tf.nn.l2_normalize(h0[u], axis=-1)
        # Complex dot product: sum(h_u * conj(w_u))
        complex_dot_u = tf.reduce_sum(h_u * tf.math.conj(w_u))
        dot_real = tf.math.real(complex_dot_u)
        dot_abs = tf.abs(complex_dot_u) # Cosine similarity magnitude
        angle = tf.math.angle(complex_dot_u)
        sinr_u_val = sinr_user0[u].numpy()

        # Log if alignment is poor or angle is large
        if (dot_abs < 0.5) or (tf.abs(angle) > np.pi / 4): # Adjusted thresholds
            line = f"User {u:02d} | dot={dot_real:.4f} | angle={angle:.4f} rad | |dot|={dot_abs:.4f} | SINR={sinr_u_val:.2f} bps/Hz"
            log_lines.append(line)

    # Append to log file periodically
    # if epoch % 20 == 0 and log_lines: # Need epoch info passed or managed externally
    #     try:
    #         with open("per_user_diag.log", "a") as f:
    #              f.write(f"[Task {task_idx}][Step ??] ----\n") # Add step/epoch info
    #              for line in log_lines:
    #                  f.write(line + "\n")
    #     except Exception as e:
    #         print(f"Error writing to per_user_diag.log: {e}")

    return {
        "total_loss": total_loss,
        "sinr": sinr_dB_batch_mean,
        "w_pred": w_pred, # Return predictions for potential external logging
        "h": h,         # Return channel for potential external logging
        "sinr_per_user_bps": sinr_per_user_bps, # For external diag log
        "log_lines": log_lines # For external diag log
    }

# --- Simplified Training Loop (Single Task) ---
def training_loop_single_task(model, optimizer, task, task_idx, num_epochs=50, batch_size=8, noise_power=1e-3):
    """ Trains the model on a single specified task. """
    print(f"\n🧠 Training Single Task: {task['name']} (Index: {task_idx}) for {num_epochs} epochs.")
    task_name = task['name']

    # Clear log files at the start of training
    open("summary_kpi.log", "w").close()
    open("per_user_diag.log", "w").close()

    for epoch in tqdm(range(num_epochs), desc=f"Training {task_name}"):
        # 1. Generate Data for the chosen task
        x_batch, h_batch, channel_feats = model.generate_synthetic_batch(task, batch_size)

        # 2. Perform one training step
        metrics = train_step_single_task(model, optimizer, x_batch, h_batch, channel_feats, task_idx, noise_power)

        # 3. Logging
        # Log summary KPIs periodically
        if (epoch + 1) % 10 == 0: # Log every 10 epochs
             sinr_val = metrics['sinr'].numpy()
             loss_val = metrics['total_loss'].numpy()
             try:
                 with open("summary_kpi.log", "a") as f:
                     f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] SINR={sinr_val:.2f} dB | Loss={loss_val:.4f}\n")
             except Exception as e:
                 print(f"Error writing to summary_kpi.log: {e}")

        # Log per-user diagnostics periodically
        if (epoch + 1) % 20 == 0: # Log every 20 epochs
            log_lines = metrics.get("log_lines", [])
            if log_lines:
                 try:
                     with open("per_user_diag.log", "a") as f:
                         f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch+1}] ----\n")
                         for line in log_lines:
                             f.write(line + "\n")
                 except Exception as e:
                     print(f"Error writing to per_user_diag.log: {e}")

        # Optional: Add evaluation on a separate validation set periodically

    print(f"\n✅ Finished training loop for task: {task_name}")


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Configuration for Single Task Training ---
    CHOSEN_TASK_INDEX = 0  # Index of the task to train on (0: Static, 1: Pedestrian, etc.)
    NUM_EPOCHS = 100       # Increase epochs for single task convergence
    LEARNING_RATE = 1e-4
    NOISE_POWER_LIN = 1e-3 # Linear noise power (adjust based on assumed signal power/SNR)
    BATCH_SIZE_MAIN = 8    # Potentially larger batch size for stable gradients

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
    print(f"------------------------------------")

    # --- Optional: Baseline Calculation (MMSE/Random) ---
    # You might want to run this on the *chosen* task's data
    print("\n📊 Calculating Baselines for the chosen task...")
    try:
        x_base, h_base, feats_base = BeamformingMetaAttentionModel(NUM_ANTENNAS, NUM_USERS, len(TASKS), False, False).generate_synthetic_batch(single_task, batch_size=32) # Larger batch for stable baseline
        sinr_r, thrpt_r, lat_r = compute_random(h_base, NOISE_POWER_LIN)
        sinr_m, thrpt_m, lat_m = compute_mmse(h_base, NOISE_POWER_LIN) # RZF approximation

        print(f"🎲 Random Beamforming → SINR: {sinr_r:.2f} dB | Thrpt: {thrpt_r:.4f} bps/Hz")
        print(f"🎯 RZF/MMSE Approx. → SINR: {sinr_m:.2f} dB | Thrpt: {thrpt_m:.4f} bps/Hz")
    except Exception as e:
        print(f"⚠️ Could not compute baselines: {e}")
    print("-" * 30)


    # --- Model and Optimizer Initialization ---
    model = BeamformingMetaAttentionModel(
        num_antennas=NUM_ANTENNAS,
        num_users=NUM_USERS,
        num_tasks=len(TASKS), # Keep full num_tasks for layer creation consistency
        use_replay=False,     # <<<--- CL Disabled --->>>
        use_fisher=False      # <<<--- CL Disabled --->>>
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # --- Build Model and Optimizer ---
    # Call the model once with dummy data to build layers
    print("Building model...")
    dummy_h = tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64)
    dummy_feats = tf.zeros((1, NUM_USERS, 3), dtype=tf.float32) # 3 features: Doppler, Delay, SNR
    try:
        _ = model(dummy_h, dummy_feats, task_idx=CHOSEN_TASK_INDEX, training=False)
        # Build optimizer explicitly (might help with some TF issues)
        optimizer.build(model.trainable_variables)
        print("✅ Model and Optimizer built successfully.")
    except Exception as e:
        print(f"❌ Error building model/optimizer: {e}")
        sys.exit(1)

    # --- Warm-up / Sanity Check (Optional but recommended) ---
    print("\n performing sanity check forward pass...")
    try:
        x_check, h_check, feats_check = model.generate_synthetic_batch(single_task, batch_size=2)
        w_check = model(x_check, feats_check, task_idx=CHOSEN_TASK_INDEX, training=False)
        norm_w = tf.reduce_mean(tf.norm(w_check, axis=-1))
        norm_h = tf.reduce_mean(tf.norm(h_check, axis=-1))
        tf.print("Sanity Check: Avg Norm(w) =", norm_w, "| Avg Norm(h) =", norm_h)
        print("✅ Sanity check forward pass completed.")
    except Exception as e:
        print(f"⚠️ Error during sanity check: {e}")


    # --- Start Single Task Training Loop ---
    training_loop_single_task(
        model=model,
        optimizer=optimizer,
        task=single_task,
        task_idx=CHOSEN_TASK_INDEX,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE_MAIN,
        noise_power=NOISE_POWER_LIN
    )

    print("\n--- Single Task Training Finished ---")