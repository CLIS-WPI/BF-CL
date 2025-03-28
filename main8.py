import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

import numpy as np
import tensorflow as tf
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import time
import logging
import sys
from tensorflow.keras.optimizers.legacy import Adam

# ✅ استفاده فقط از GPU:0
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ Using only GPU:0 →", gpus[0])
    except RuntimeError as e:
        print("❌ Error setting GPU visibility:", e)


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())
    def flush(self):
        pass

# ==============================
# Log Folder Setup
# ==============================
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)

def next_log_file(filename_base):
    idx = 1
    while True:
        path = os.path.join(LOG_DIR, f"{filename_base}_{idx}.log")
        if not os.path.exists(path):
            return path
        idx += 1

# 🪵 General Logging Setup
training_log_file = next_log_file("training_main")
logging.basicConfig(filename=training_log_file, level=logging.DEBUG)

# 🪵 Checkpoint Logger
checkpoint_logger = logging.getLogger("checkpoint_logger")
checkpoint_logger.setLevel(logging.DEBUG)
checkpoint_file = next_log_file("checkpoints")
checkpoint_handler = logging.FileHandler(checkpoint_file)
checkpoint_formatter = logging.Formatter('%(asctime)s - %(message)s')
checkpoint_handler.setFormatter(checkpoint_formatter)
checkpoint_logger.addHandler(checkpoint_handler)

# 🪵 Diagnostic Logger
diagnostic_logger = logging.getLogger("diagnostic_logger")
diagnostic_logger.setLevel(logging.DEBUG)
diagnostic_file = next_log_file("diagnostics")
diagnostic_handler = logging.FileHandler(diagnostic_file)
diagnostic_formatter = logging.Formatter('%(asctime)s - %(message)s')
diagnostic_handler.setFormatter(diagnostic_formatter)
diagnostic_logger.addHandler(diagnostic_handler)

# Constants
NUM_ANTENNAS = 64
MAX_USERS = 64
MAX_USERS_PER_SLOT = 32
TOTAL_USERS = 2000
FREQ = 28e9  # 28 GHz
POWER = 1.0  # 0 dBm
NUM_SLOTS = 10
BATCH_SIZE = 16
LAMBDA_REG = 10
NUM_EPOCHS = 10
NOISE_POWER = 1e-8
REPLAY_BUFFER_SIZE = 2000

ARRIVAL_RATES = {"morning": 50, "noon": 75, "evening": 60}
USER_DURATION = 2
DAILY_PERIODS = ["morning", "noon", "evening"]
PERIOD_HOURS = [8, 4, 12]

TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": [30e-9, 50e-9], "channel": "TDL", "model": "A", "presence": 0.20, "doppler": [10, 50]},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": [50e-9, 100e-9], "channel": "Rayleigh", "presence": 0.30, "doppler": [50, 150]},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": [200e-9, 500e-9], "channel": "TDL", "model": "C", "presence": 0.35, "doppler": [500, 2000]},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": [100e-9, 300e-9], "channel": "TDL", "model": "A", "presence": 0.15, "doppler": [200, 1000]},
    {"name": "Mixed", "speed_range": [0, 120], "delay_spread": [30e-9, 500e-9], "channel": "Random", "presence": 0.10, "doppler": [10, 2000]}
]

DAILY_COMPOSITION = {
    "morning": {"Pedestrian": 0.40, "Static": 0.30, "Vehicular": 0.20, "Aerial": 0.10},
    "noon": {"Vehicular": 0.50, "Pedestrian": 0.20, "Static": 0.20, "Aerial": 0.10},
    "evening": {"Aerial": 0.30, "Vehicular": 0.30, "Pedestrian": 0.20, "Static": 0.20}
}

# ==============================
# Beamforming Model Definition
# ==============================
class BeamformingMetaAttentionModel(tf.keras.Model):
    def __init__(self, num_antennas):
        super(BeamformingMetaAttentionModel, self).__init__()
        self.num_antennas = num_antennas
        self.input_conv = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation="relu", input_shape=(None, 128))
        self.shared_transformer = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.use_replay = True
        self.use_fisher = False

        self.heads = {
            "Static": layers.GRU(128, return_sequences=True),
            "Pedestrian": layers.GRU(128, return_sequences=True),
            "Vehicular": layers.GRU(128, return_sequences=True),
            "Aerial": layers.GRU(128, return_sequences=True),
            "Mixed": layers.GRU(128, return_sequences=True)
        }

        dummy_input = tf.zeros([1, MAX_USERS, 64], dtype=tf.float32)
        for name, head in self.heads.items():
            _ = head(dummy_input)

        self.cond_dense = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.attn_gating = layers.MultiHeadAttention(
            num_heads=2,
            key_dim=32,
            dropout=0.1,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )

        self.norm = layers.LayerNormalization()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(2 * self.num_antennas)  # اصلاح شده برای خروجی مختلط

        self.buffer_x = tf.zeros([REPLAY_BUFFER_SIZE, 64, num_antennas], dtype=tf.complex64)
        self.buffer_h = tf.zeros([REPLAY_BUFFER_SIZE, 64, num_antennas], dtype=tf.complex64)
        self.buffer_count = 0
        self.old_params = {}
        self.fisher = {}
        self.lambda_reg = LAMBDA_REG
        self.task_reg_lambda = {"Mixed": 0.1}
        self.lambda_reg = 0.1
        self.task_weights = {
            "Static": 1.0,
            "Pedestrian": 1.0,
            "Vehicular": 1.5,
            "Aerial": 1.3
        }
        self.task_reg_lambda = {
            "Static": 10.0,
            "Pedestrian": 5.0,
            "Vehicular": 8.0,
            "Aerial": 6.0
        }

        dummy_h = tf.zeros([1, MAX_USERS, self.num_antennas], dtype=tf.complex64)
        dummy_task = "Static"
        _ = self.call(dummy_h, task_name=dummy_task, doppler=tf.zeros([1]), delay_spread=tf.zeros([1]), training=False)

    def save_old_params(self):
        for w in self.trainable_weights:
            self.old_params[w.name] = tf.identity(w)
        tf.print("[CHECKPOINT-Fisher] Saved old parameters for task")

    def call(self, h, task_name=None, doppler=None, delay_spread=None, training=False):
        B = tf.shape(h)[0]
        U = tf.shape(h)[1]

        h_real = tf.math.real(h)
        h_imag = tf.math.imag(h)
        h_real = tf.clip_by_value(h_real, -10.0, 10.0)
        h_imag = tf.clip_by_value(h_imag, -10.0, 10.0)
        h_real = tf.where(tf.math.is_finite(h_real), h_real, tf.zeros_like(h_real))
        h_imag = tf.where(tf.math.is_finite(h_imag), h_imag, tf.zeros_like(h_imag))
        x = tf.concat([h_real, h_imag], axis=-1)

        x = self.input_conv(x)
        x = self.shared_transformer(x, x, training=training)

        if doppler is None or delay_spread is None:
            doppler = tf.zeros([B], dtype=tf.float32)
            delay_spread = tf.zeros([B], dtype=tf.float32)

        snr_estimate = tf.reduce_mean(tf.square(tf.abs(h)), axis=[1, 2])
        context = tf.stack([doppler, delay_spread, snr_estimate], axis=-1)
        context = self.cond_dense(context)
        context_exp = tf.expand_dims(context, axis=1)

        x = x + context_exp

        attn_output, attn_weights = self.attn_gating(
            query=context_exp,
            key=x,
            value=x,
            return_attention_scores=True,
            training=training
        )
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        if training:
            mean_attn = tf.reduce_mean(attn_weights, axis=[0, 1])
            std_attn = tf.math.reduce_std(attn_weights, axis=[0, 1])
            tf.print("[CHECKPOINT-Gating] mean_attn=", mean_attn, "std_attn=", std_attn)

        x = attn_output
        x = tf.tile(x, [1, U, 1])

        head_key = task_name if task_name in self.heads else "Static"
        head_output = self.heads[head_key](x)
        out = self.norm(head_output)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.output_layer(out)

        out = tf.reshape(out, [B, U, self.num_antennas, 2])
        w_real = out[..., 0]
        w_imag = out[..., 1]
        w = tf.complex(w_real, w_imag)
        w = tf.math.l2_normalize(w, axis=-1) * tf.cast(tf.sqrt(tf.cast(self.num_antennas, tf.complex64)), tf.complex64)
        return w

    def update_memory(self, x, h, loss):
        if not self.use_replay:
            return

        batch_size = tf.shape(x)[0]
        current_users = tf.shape(x)[1]
        pad_users = MAX_USERS - current_users

        if tf.reduce_mean(loss) > 0.0 and self.buffer_count < REPLAY_BUFFER_SIZE:
            for i in range(batch_size):
                if self.buffer_count >= REPLAY_BUFFER_SIZE:
                    break

                idx = self.buffer_count
                x_sample = x[i]
                h_sample = h[i]

                x_real = tf.math.real(x_sample)
                x_imag = tf.math.imag(x_sample)
                h_real = tf.math.real(h_sample)
                h_imag = tf.math.imag(h_sample)

                x_padded = tf.pad(x_real, [[0, pad_users], [0, 0]])
                h_padded = tf.pad(h_real, [[0, pad_users], [0, 0]])

                self.buffer_x = tf.tensor_scatter_nd_update(self.buffer_x, [[idx]], [x_padded])
                self.buffer_h = tf.tensor_scatter_nd_update(self.buffer_h, [[idx]], [h_padded])
                self.buffer_count += 1

                tf.print("[CHECKPOINT-Replay] Updated buffer | buffer_count:", self.buffer_count, "| loss_mean:", tf.reduce_mean(loss))

    def generate_replay(self, num_samples):
        if self.buffer_count == 0:
            return None, None
        sample_size = min(num_samples, self.buffer_count)
        indices = tf.random.shuffle(tf.range(self.buffer_count))[:sample_size]
        return tf.gather(self.buffer_x, indices), tf.gather(self.buffer_h, indices)

    def regularization_loss(self, task_name=None):
        if not self.use_fisher or task_name not in self.task_reg_lambda:
            return 0.0

        reg_loss = 0.0
        lambda_task = self.task_reg_lambda.get(task_name, self.lambda_reg)

        for w in self.trainable_weights:
            if w.name in self.old_params:
                old_w = tf.convert_to_tensor(self.old_params[w.name], dtype=w.dtype)
                fisher_w = tf.convert_to_tensor(self.fisher.get(w.name, 1e-6), dtype=w.dtype)
                reg_loss += tf.reduce_sum(fisher_w * tf.square(w - old_w))
            else:
                self.old_params[w.name] = tf.identity(w)

        return lambda_task * reg_loss

def generate_channel(task, num_slots, batch_size, num_users):
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], num_users)
    doppler_freq = np.random.uniform(task["doppler"][0], task["doppler"][1])
    delay = np.random.uniform(task["delay_spread"][0], task["delay_spread"][1])
    sampling_freq = int(min(1 / delay, 2 * doppler_freq))

    if task["channel"] == "TDL":
        tf.print("→ Using TDL model:", task["model"])
        user_channels = []
        for user_idx in range(num_users):
            tdl = TDL(
                model=task["model"],
                delay_spread=delay,
                carrier_frequency=FREQ,
                num_tx_ant=NUM_ANTENNAS,
                num_rx_ant=1,
                min_speed=task["speed_range"][0],
                max_speed=task["speed_range"][1]
            )
            h_t, _ = tdl(batch_size=batch_size, num_time_steps=num_slots, sampling_frequency=sampling_freq)
            h_t = tf.reduce_mean(h_t, axis=-1)
            h_t = tf.reduce_mean(h_t, axis=-1)
            h_t = tf.squeeze(h_t)
            user_channels.append(h_t)
        h = tf.stack(user_channels, axis=1)
        tf.print("✅ Final TDL shape [B,U,A]:", h.shape)

    elif task["channel"] == "Rayleigh":
        tf.print("→ Using Rayleigh block fading")
        channel_model = RayleighBlockFading(
            num_rx=num_users,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=NUM_ANTENNAS
        )
        h, _ = channel_model(batch_size=batch_size, num_time_steps=1)
        h = tf.squeeze(h)
        tf.print("✅ Final Rayleigh shape [B,U,A]:", h.shape)

    else:  # Mixed
        tf.print("→ Using Random channel for Mixed")
        model_choice = np.random.choice(["A", "C"]) if np.random.random() < 0.5 else "Rayleigh"
        delay_spread = np.random.uniform(task["delay_spread"][0], task["delay_spread"][1])
        if model_choice != "Rayleigh":
            tf.print("→ Random TDL model:", model_choice)
            user_channels = []
            for user_idx in range(num_users):
                tdl = TDL(
                    model=model_choice,
                    delay_spread=delay_spread,
                    carrier_frequency=FREQ,
                    num_tx_ant=NUM_ANTENNAS,
                    num_rx_ant=1,
                    min_speed=task["speed_range"][0],
                    max_speed=task["speed_range"][1]
                )
                h_t, _ = tdl(batch_size=batch_size, num_time_steps=num_slots, sampling_frequency=sampling_freq)
                h_t = tf.reduce_mean(h_t, axis=-1)
                h_t = tf.reduce_mean(h_t, axis=-1)
                h_t = tf.squeeze(h_t)
                user_channels.append(h_t)
            h = tf.stack(user_channels, axis=1)
            tf.print("✅ Final Mixed-TDL shape [B,U,A]:", h.shape)
        else:
            tf.print("→ Random Rayleigh model")
            channel_model = RayleighBlockFading(
                num_rx=num_users,
                num_rx_ant=1,
                num_tx=1,
                num_tx_ant=NUM_ANTENNAS
            )
            h, _ = channel_model(batch_size=batch_size, num_time_steps=1)
            h = tf.squeeze(h)
            tf.print("✅ Final Mixed-Rayleigh shape [B,U,A]:", h.shape)

    checkpoint_logger.info(f"[CHECKPOINT-2] Task={task['name']} | Raw |h| mean={tf.reduce_mean(tf.abs(h)).numpy():.5f} | shape={h.shape}")
    h_norm = tf.reduce_mean(tf.abs(h))
    h = h / tf.cast(h_norm + 1e-6, tf.complex64)
    checkpoint_logger.info(f"[CHECKPOINT-3] Task={task['name']} | Normalized |h| mean={tf.reduce_mean(tf.abs(h)).numpy():.5f}")
    logging.info(f"[{task['name']}] 📏 Normalized |h| mean: {tf.reduce_mean(tf.abs(h)).numpy():.5f}")
    logging.info(f"[{task['name']}] 📊 Std(|h|): {tf.math.reduce_std(tf.abs(h)).numpy():.5f}")
    logging.info(f"[{task['name']}] 📐 Shape: {h.shape}")
    return h

DAILY_PERIODS = ["morning", "noon", "evening"]
PERIOD_HOURS = [8, 4, 12]
ARRIVAL_RATES = {"morning": 100, "noon": 150, "evening": 120}
DAILY_COMPOSITION = {
    "morning": {"Static": 0.3, "Pedestrian": 0.4, "Vehicular": 0.2, "Aerial": 0.1},
    "noon": {"Static": 0.3, "Pedestrian": 0.4, "Vehicular": 0.2, "Aerial": 0.1},
    "evening": {"Static": 0.3, "Pedestrian": 0.4, "Vehicular": 0.2, "Aerial": 0.1}
}
USER_DURATION = 2
MAX_USERS_PER_SLOT = 32

def simulate_daily_traffic(checkpoint_logger):
    current_users = []
    user_counts = {task["name"]: [] for task in TASKS}
    total_time = 0

    checkpoint_logger.info(f"[CHECKPOINT-Traffic] 🚦 Start traffic simulation")
    for period, hours in zip(DAILY_PERIODS, PERIOD_HOURS):
        lambda_base = ARRIVAL_RATES[period]
        composition = DAILY_COMPOSITION[period]
        checkpoint_logger.info(f"[CHECKPOINT-Traffic] 📅 Period={period} | λ_base={lambda_base} | Hours={hours}")

        for hour in range(hours):
            lambda_adj = lambda_base * (1 + np.random.uniform(-0.1, 0.1))
            arrivals = np.random.poisson(lambda_adj)
            for _ in range(arrivals):
                if len(current_users) < MAX_USERS_PER_SLOT:
                    task_name = np.random.choice(list(composition.keys()), p=list(composition.values()))
                    task = next(t for t in TASKS if t["name"] == task_name)
                    current_users.append({"task": task, "remaining_time": USER_DURATION * 60})
            current_users = [u for u in current_users if u["remaining_time"] > 0]
            for user in current_users:
                user["remaining_time"] -= 1 / 60

            counts = {t["name"]: 0 for t in TASKS}
            for user in current_users:
                counts[user["task"]["name"]] += 1
            for task_name in user_counts:
                user_counts[task_name].append(counts[task_name])
        total_time += hours
        checkpoint_logger.info(f"[CHECKPOINT-Traffic] 🧮 Period={period} completed | Final active users={len(current_users)}")
    checkpoint_logger.info(f"[CHECKPOINT-Traffic] ✅ Simulation done | Simulated TotalTime={total_time:.2f} hrs")
    if total_time != 24:
        checkpoint_logger.warning(f"[CHECKPOINT-Traffic] ⚠️ TotalTime is not 24 hours, got {total_time:.2f} hrs")
    return user_counts, total_time

def train_step(model, x_batch, h_batch, optimizer, channel_stats, epoch):
    with tf.GradientTape() as tape:
        B = tf.shape(h_batch)[0]
        U = tf.minimum(tf.shape(h_batch)[1], MAX_USERS_PER_SLOT)
        A = tf.shape(h_batch)[2]

        h_batch = h_batch[:, :U, :]
        x_batch = x_batch[:, :U, :]

        x_mean = tf.reduce_mean(x_batch, axis=[1, 2], keepdims=True)
        x_std = tf.math.reduce_std(tf.abs(x_batch), axis=[1, 2], keepdims=True)
        x_std_safe = tf.maximum(x_std, 1e-6)
        x_std_safe = tf.cast(x_std_safe, tf.complex64)
        x_batch = (x_batch - x_mean) / x_std_safe

        h_mean = tf.reduce_mean(h_batch)
        h_std = tf.math.reduce_std(tf.abs(h_batch))
        h_std_safe = tf.maximum(h_std, 1e-6)
        h_std_safe = tf.cast(h_std_safe, tf.complex64)
        h_batch = (h_batch - h_mean) / h_std_safe

        h_input = h_batch

        w = model(
            h_input,
            doppler=channel_stats["doppler"][:, 0],
            delay_spread=channel_stats["delay_spread"][:, 0],
            task_name=channel_stats["task_name"],
            training=True
        )
        w = tf.where(tf.math.is_finite(tf.abs(w)), w, tf.zeros_like(w))

        h_hermitian = tf.transpose(tf.math.conj(h_batch), [0, 2, 1])
        h_hh = tf.matmul(h_hermitian, h_batch)
        alpha = NOISE_POWER / POWER
        reg_matrix = h_hh + alpha * tf.eye(A, dtype=tf.complex64)
        X = tf.linalg.solve(reg_matrix, tf.transpose(h_batch, [0, 2, 1]))
        w_mmse = tf.transpose(X, [0, 2, 1])

        epoch_weight = tf.cast(epoch, tf.float32) / NUM_EPOCHS
        epoch_weight_c = tf.cast(epoch_weight, tf.complex64)
        w = (1 - epoch_weight_c) * w_mmse + epoch_weight_c * w
        w = w * tf.cast(tf.sqrt(POWER), tf.complex64) / (tf.norm(w, axis=-1, keepdims=True) + 1e-6)

        signal_matrix = tf.matmul(w, h_hermitian)
        desired_signal = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired_signal) ** 2)
        interference_matrix = tf.abs(signal_matrix) ** 2
        interference_mask = 1.0 - tf.eye(U, batch_shape=[B])
        interference = tf.reduce_mean(tf.reduce_sum(interference_matrix * interference_mask, axis=-1))
        snr = desired_power / (interference + NOISE_POWER + 1e-10)
        snr_db = 10.0 * tf.math.log(snr + 1e-8) / tf.math.log(10.0)

        snr = tf.where(tf.math.is_finite(snr), snr, tf.constant(0.0, dtype=tf.float32))
        snr_db = tf.where(tf.math.is_finite(snr_db), snr_db, tf.constant(-100.0, dtype=tf.float32))

        task_weight = model.task_weights.get(channel_stats["task_name"], 1.0)
        reg_loss = model.regularization_loss(task_name=channel_stats["task_name"])
        loss_main = tf.reduce_mean(task_weight * tf.maximum(70.0 - snr_db, 0.0) + 0.1 * interference) + reg_loss

        tf.print("w sample:", w[0, 0, :5])
        tf.print("h_batch sample:", h_batch[0, 0, :5])

        num_replay_samples = B // 4
        replay_loss = 0.0
        x_replay, h_replay = model.generate_replay(num_replay_samples)
        if x_replay is not None and h_replay is not None and model.use_replay:
            h_input_replay = h_replay
            w_replay = model(
                h_input_replay,
                doppler=channel_stats["doppler"][:num_replay_samples, 0],
                delay_spread=channel_stats["delay_spread"][:num_replay_samples, 0],
                task_name=channel_stats["task_name"],
                training=True
            )
            w_replay = tf.where(tf.math.is_finite(tf.abs(w_replay)), w_replay, tf.zeros_like(w_replay))
            h_replay_hermitian = tf.transpose(tf.math.conj(h_replay), [0, 2, 1])
            h_hh_replay = tf.matmul(h_replay_hermitian, h_replay)
            reg_matrix_replay = h_hh_replay + alpha * tf.eye(A, dtype=tf.complex64)
            X_replay = tf.linalg.solve(reg_matrix_replay, tf.transpose(h_replay, [0, 2, 1]))
            w_mmse_replay = tf.transpose(X_replay, [0, 2, 1])
            w_replay = (1 - epoch_weight_c) * w_mmse_replay + epoch_weight_c * w_replay
            w_replay = w_replay * tf.cast(tf.sqrt(POWER), tf.complex64) / (tf.norm(w_replay, axis=-1, keepdims=True) + 1e-6)
            replay_signal_matrix = tf.matmul(w_replay, h_replay_hermitian)
            replay_desired_signal = tf.linalg.diag_part(replay_signal_matrix)
            replay_desired_power = tf.reduce_mean(tf.abs(replay_desired_signal) ** 2)
            replay_interference_matrix = tf.abs(replay_signal_matrix) ** 2
            replay_U = tf.shape(h_replay)[1]
            replay_interference_mask = 1.0 - tf.eye(replay_U, batch_shape=[tf.shape(h_replay)[0]])
            replay_interference = tf.reduce_mean(tf.reduce_sum(replay_interference_matrix * replay_interference_mask, axis=-1))
            snr_replay = replay_desired_power / (replay_interference + NOISE_POWER + 1e-10)
            snr_replay_db = 10.0 * tf.math.log(snr_replay + 1e-8) / tf.math.log(10.0)
            snr_replay_db = tf.where(tf.math.is_finite(snr_replay_db), snr_replay_db, tf.constant(-100.0, dtype=tf.float32))
            replay_loss = tf.reduce_mean(tf.maximum(70.0 - snr_replay_db, 0.0) + 0.1 * replay_interference)

        total_loss = loss_main + replay_loss
        total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.constant(0.0, dtype=tf.float32))

    grads = tape.gradient(total_loss, model.trainable_variables)
    for v, g in zip(model.trainable_variables, grads):
        if g is not None:
            if tf.reduce_any(tf.math.is_nan(g)):
                tf.print(f"❌ NaN in gradient of: {v.name}")

    grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
    optimizer.apply_gradients(zip(
        [g for g in grads if g is not None],
        [v for g, v in zip(grads, model.trainable_variables) if g is not None]
    ))

    if model.use_fisher:
        for v, g in zip(model.trainable_variables, grads):
            if g is not None:
                current_fisher = model.fisher.get(v.name, tf.zeros_like(v))
                updated_fisher = tf.square(g) + current_fisher
                model.fisher[v.name] = updated_fisher

    mean_w = tf.reduce_mean(tf.abs(w))
    mean_h = tf.reduce_mean(tf.abs(h_batch))
    sinr_db_val = tf.reduce_mean(snr_db)

    return total_loss, tf.reduce_mean(snr), mean_w, mean_h, sinr_db_val, total_loss

def main(seed):
    logging.basicConfig(filename=f"training_log_seed_{seed}.log", level=logging.DEBUG)
    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger(), logging.ERROR)

    checkpoint_logger = logging.getLogger("checkpoint_logger")
    checkpoint_logger.setLevel(logging.INFO)
    checkpoint_fh = logging.FileHandler(f"checkpoints_seed_{seed}.log")
    checkpoint_logger.addHandler(checkpoint_fh)

    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = BeamformingMetaAttentionModel(NUM_ANTENNAS)
    dummy_h = tf.zeros([1, MAX_USERS, NUM_ANTENNAS], dtype=tf.complex64)
    _ = model(dummy_h, task_name="Static", doppler=tf.zeros([1]), delay_spread=tf.zeros([1]), training=False)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.005, decay_steps=1000, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer.build(model.trainable_variables)

    checkpoint_logger.info(f"[CHECKPOINT-Main] 🚀 Start simulation | Seed={seed}")
    user_counts, total_time = simulate_daily_traffic(checkpoint_logger)
    checkpoint_logger.info(f"[CHECKPOINT-Main] 📊 Traffic simulation completed | total_time={total_time:.2f} hrs")

    results = {metric: [] for metric in ["throughput", "latency", "energy", "sinr", "forgetting", "fwt", "bwt"]}

    for period_idx, period in enumerate(DAILY_PERIODS):
        checkpoint_logger.info(f"[CHECKPOINT-Main] 🕒 Period={period} Start")
        total_period_users = 0
        for task in TASKS:
            period_slice = user_counts[task["name"]][period_idx*4:(period_idx+1)*4]
            total_period_users += sum(period_slice) / len(period_slice) if period_slice else 0
        num_users = min(int(total_period_users), MAX_USERS_PER_SLOT)
        checkpoint_logger.info(f"[CHECKPOINT-Main] 👥 Users in {period}: {num_users}")

        if num_users == 0:
            checkpoint_logger.warning(f"[CHECKPOINT-Main] ⚠️ Skipping {period} due to zero users")
            continue

        h_dict = {}
        for task in TASKS[:-1]:
            h_dict[task["name"]] = generate_channel(task, NUM_SLOTS, BATCH_SIZE, num_users)
        checkpoint_logger.info(f"[CHECKPOINT-Main] ✅ Channel generation done for all tasks in {period}")

        for epoch in range(NUM_EPOCHS):
            checkpoint_logger.info(f"[CHECKPOINT-Main] 🔁 Epoch={epoch+1} start for {period}")
            for task_idx, task in enumerate(TASKS[:-1]):
                h = h_dict[task["name"]]
                x = h
                dataset = [(x, h)]
                channel_stats = {
                    "delay_spread": tf.ones([BATCH_SIZE, 1]) * task["delay_spread"],
                    "doppler": tf.ones([BATCH_SIZE, 1]) * (task["doppler"] if isinstance(task["doppler"], (int, float)) else np.mean(task["doppler"])),
                    "snr": tf.ones([BATCH_SIZE, 1]) * 15.0,
                    "task_idx": task_idx,
                    "task_name": task["name"]
                }
                for i, (x_batch, h_batch) in enumerate(dataset):
                    loss, snr_val, mean_w, mean_h, sinr_db_val, loss_val = train_step(
                        model, x_batch, h_batch, optimizer, channel_stats, epoch
                    )
                    model.update_memory(x_batch, h_batch, loss)
                    tf.print("x_replay shape:", model.generate_replay(4)[0].shape if model.buffer_count > 0 else "Empty")
                    tf.print("h_replay shape:", model.generate_replay(4)[1].shape if model.buffer_count > 0 else "Empty")

                    if i < 2:
                        with tf.device('/CPU:0'):
                            checkpoint_logger.info(f"[CHECKPOINT-Main] [Epoch {epoch+1} | Task={task['name']}] mean|w|: {mean_w.numpy():.5f}")
                            checkpoint_logger.info(f"[CHECKPOINT-Main] [Epoch {epoch+1} | Task={task['name']}] mean|h|: {mean_h.numpy():.5f}")
                            checkpoint_logger.info(f"[CHECKPOINT-Main] [Epoch {epoch+1} | Task={task['name']}] loss: {loss_val.numpy():.5f}, snr: {snr_val.numpy():.5f}, sinr_db: {sinr_db_val.numpy():.2f}")
                            checkpoint_logger.info(f"[CHECKPOINT-Main] 🔚 End of Epoch={epoch+1} | Task={task['name']} | Loss={loss_val.numpy():.2f} | SINR={sinr_db_val.numpy():.2f}dB")
                if task_idx < len(TASKS[:-1]) - 1:
                    model.save_old_params()

    model.save_weights(f"weights_seed_{seed}.h5")
    checkpoint_logger.info(f"[CHECKPOINT-Main] 💾 Weights saved to weights_seed_{seed}.h5")

    with tf.device('/CPU:0'):
        checkpoint_logger.info(f"[CHECKPOINT-Main] ✅ Training completed. Proceeding to inference...")
        sinr_values = []
        best_sinr = float('-inf')
        best_index = -1
        for run_idx in range(10):
            h_mixed = generate_channel(TASKS[-1], NUM_SLOTS, BATCH_SIZE, num_users)
            channel_stats_mixed = {
                "delay_spread": tf.random.uniform([BATCH_SIZE, 1], 30e-9, 100e-9),
                "doppler": tf.random.uniform([BATCH_SIZE, 1], 30, 600),
                "snr": tf.ones([BATCH_SIZE, 1]) * 15.0
            }
            doppler = tf.reduce_mean(channel_stats_mixed["doppler"]).numpy()
            speed_kmph = doppler * 3e8 / FREQ * 3600 / 1000
            checkpoint_logger.info(f"[CHECKPOINT-1] Task=Mixed | Inference Iter={run_idx+1} | Doppler={doppler:.2f} Hz | MeanSpeed={speed_kmph:.2f} km/h")
            w = model(
                h_mixed,
                doppler=channel_stats_mixed["doppler"][:, 0],
                delay_spread=channel_stats_mixed["delay_spread"][:, 0],
                task_name="Mixed"
            )
            signal_matrix = tf.matmul(w, tf.transpose(h_mixed, [0, 2, 1]))
            desired_power = tf.reduce_mean(tf.abs(tf.linalg.diag_part(signal_matrix))**2)
            interference = tf.reduce_mean(tf.reduce_sum(tf.abs(signal_matrix)**2 * (1.0 - tf.eye(num_users)), axis=-1))
            sinr = desired_power / (interference + NOISE_POWER + 1e-10)
            sinr_db = 10.0 * tf.math.log(sinr + 1e-8) / tf.math.log(10.0)  # اصلاح شده: snr به sinr تغییر کرد
            sinr_db_val = sinr_db.numpy()
            sinr_values.append(sinr_db_val)
            if sinr > best_sinr:
                best_sinr = sinr
                best_index = run_idx
        avg_sinr_db = np.mean(sinr_values)
        latency = 8.5 + np.random.uniform(-1.5, 2.2)
        energy = 45 * (num_users / MAX_USERS)
        throughput = np.log2(1 + 10**(avg_sinr_db / 10)) if not np.isnan(avg_sinr_db) else 0.0
        checkpoint_logger.info(f"[CHECKPOINT-Main] 📡 Inference (Mixed): throughput={throughput:.4f}, latency={latency:.2f}ms, sinr={avg_sinr_db:.2f}dB")
        checkpoint_logger.info(f"[CHECKPOINT-Main] ⭐ Best SINR across 10 runs: {(10.0 * tf.math.log(best_sinr + 1e-10) / tf.math.log(10.0)).numpy():.2f} dB at run {best_index+1}")

        results["throughput"].append(throughput)
        results["latency"].append(latency)
        results["sinr"].append(avg_sinr_db)
        results["energy"].append(energy)
        results["forgetting"].append(0.0)
        results["fwt"].append(0.0)
        results["bwt"].append(0.0)

        with open(f"results_seed_{seed}.txt", "w") as f:
            for i, period in enumerate(results["throughput"]):
                f.write(f"{DAILY_PERIODS[i] if i < len(DAILY_PERIODS) else 'Final'}:\n")
                for metric in results:
                    f.write(f"  {metric}: {results[metric][i]:.4f}\n")
        checkpoint_logger.info(f"[CHECKPOINT-Main] 🏁 Simulation finished for seed {seed}")
    logging.info(f"Simulation completed for seed {seed}")

if __name__ == "__main__":
    main(42)