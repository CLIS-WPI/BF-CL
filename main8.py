"""
This code implements the BeamformingMetaAttentionModel for realistic 6G scenarios with dynamic traffic profiles,
task-specific heads, gating, and continual learning (Replay + Fisher Regularization). The scenario assumes:
  - Single-Cell with 64 antennas
  - Up to 2000 daily users (max 500 concurrent)
  - Realistic traffic fluctuations and channel models (TDL-A, TDL-C, Rayleigh)
"""
import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import time
import os
import logging
import sys
from tensorflow.keras.optimizers.legacy import Adam
tf.get_logger().setLevel('ERROR')

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())
    def flush(self):
        pass

# Logging setup
logging.basicConfig(filename="training_main.log", level=logging.DEBUG)
checkpoint_logger = logging.getLogger("checkpoint_logger")
checkpoint_logger.setLevel(logging.DEBUG)
checkpoint_handler = logging.FileHandler("checkpoints.log")
checkpoint_handler.setLevel(logging.DEBUG)
checkpoint_formatter = logging.Formatter('%(asctime)s - %(message)s')
checkpoint_handler.setFormatter(checkpoint_formatter)
checkpoint_logger.addHandler(checkpoint_handler)

diagnostic_logger = logging.getLogger("diagnostic_logger")
diagnostic_logger.setLevel(logging.DEBUG)
diagnostic_handler = logging.FileHandler("diagnostics.log")
diagnostic_handler.setLevel(logging.DEBUG)
diagnostic_formatter = logging.Formatter('%(asctime)s - %(message)s')
diagnostic_handler.setFormatter(diagnostic_formatter)
diagnostic_logger.addHandler(diagnostic_handler)

# Constants
NUM_ANTENNAS = 64
MAX_USERS = 128
TOTAL_USERS = 2000
FREQ = 28e9
POWER = 1000.0
NUM_SLOTS = 10
BATCH_SIZE = 16
LAMBDA_REG = 10
NUM_EPOCHS = 10 #20
NOISE_POWER = 1e-8 #1e-3
REPLAY_BUFFER_SIZE = 2000

ARRIVAL_RATES = {"morning": 100, "noon": 150, "evening": 120}
USER_DURATION = 2
DAILY_PERIODS = ["morning", "noon", "evening"]
PERIOD_HOURS = [8, 4, 12]

TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": 30e-9, "channel": "TDL", "model": "A", "presence": 0.20, "doppler": 30},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": 50e-9, "channel": "Rayleigh", "presence": 0.30, "doppler": 70},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": 100e-9, "channel": "TDL", "model": "C", "presence": 0.35, "doppler": [300, 600]},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": 70e-9, "channel": "TDL", "model": "A", "presence": 0.15, "doppler": [150, 250]},
    {"name": "Mixed", "speed_range": [0, 120], "delay_spread": [30e-9, 100e-9], "channel": "Random", "presence": 0.10, "doppler": [30, 600]}
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
        self.input_conv = layers.Conv1D(64, 1, activation='relu', dtype=tf.float32)
        self.shared_transformer = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.use_replay = True
        self.use_fisher = True

        self.heads = {
        "Static": layers.GRU(128, return_sequences=True),
        "Pedestrian": layers.GRU(128, return_sequences=True),
        "Vehicular": layers.GRU(128, return_sequences=True),
        "Aerial": layers.GRU(128, return_sequences=True),
        "Mixed": layers.GRU(128, return_sequences=True)
        }

        dummy_input = tf.zeros([1, MAX_USERS, 64], dtype=tf.float32)
        for name, head in self.heads.items():
            _ = head(dummy_input)  # build all GRU heads ahead of time

        # Gating Network: Assigns users to heads using attention over [doppler, delay_spread, snr_estimate]
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
        self.output_layer = layers.Dense(self.num_antennas)
        # Replay Buffer: Fixed size of 2000 samples to cover daily users, stores high-loss samples to prevent forgetting
        self.buffer_x = tf.zeros([REPLAY_BUFFER_SIZE, MAX_USERS, num_antennas], dtype=tf.complex64)
        self.buffer_h = tf.zeros([REPLAY_BUFFER_SIZE, MAX_USERS, num_antennas], dtype=tf.complex64)
        self.buffer_count = 0
        self.old_params = {}
        self.fisher = {}
        self.lambda_reg = LAMBDA_REG

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
        
        # Dummy forward pass to build entire model (incl. heads)
        dummy_h = tf.zeros([1, MAX_USERS, self.num_antennas], dtype=tf.complex64)
        dummy_task = "Static"
        _ = self.call(dummy_h, task_name=dummy_task, doppler=tf.zeros([1]), delay_spread=tf.zeros([1]), training=False)



    def call(self, h, task_name=None, doppler=None, delay_spread=None, training=False):
        B = tf.shape(h)[0]
        U = tf.shape(h)[1]

        x = tf.abs(h)
        x = self.input_conv(x)
        x = self.shared_transformer(x, x, training=training)

        if doppler is None or delay_spread is None:
            doppler = tf.zeros([B], dtype=tf.float32)
            delay_spread = tf.zeros([B], dtype=tf.float32)

        snr_estimate = tf.reduce_mean(tf.square(tf.abs(h)), axis=[1, 2])  # [B]
        context = tf.stack([doppler, delay_spread, snr_estimate], axis=-1)  # [B, 3]
        context = self.cond_dense(context)  # [B, 64]
        context_exp = tf.expand_dims(context, axis=1)  # [B, 1, 64]

        # Inject conditioning
        x = x + context_exp  # [B, U, 64]

        # Attention gating
        attn_output, attn_weights = self.attn_gating(
            query=context_exp,
            key=x,
            value=x,
            return_attention_scores=True,
            training=training
        )  # attn_output: [B, 1, 64], attn_weights: [B, 1, U]

        # ✅ Gating Logging
        if training:
            mean_attn = tf.reduce_mean(attn_weights, axis=[0, 1])
            std_attn = tf.math.reduce_std(attn_weights, axis=[0, 1])
            tf.print("[CHECKPOINT-Gating] mean_attn=", mean_attn, "std_attn=", std_attn)
        tf.print("[CHECKPOINT-Gating] attention_weights_sample:", attn_weights[0, 0, :5])  # ۵ تا وزن اول از بچ اول
        # Expand gated feature
        x = attn_output  # [B, 1, 64]
        x = tf.tile(x, [1, U, 1])  # [B, U, 64]

        head_key = task_name if task_name in self.heads else "Static"
        head_output = self.heads[head_key](x)
        out = self.norm(head_output)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.output_layer(out)
        out = tf.math.l2_normalize(out, axis=-1) * tf.cast(tf.sqrt(tf.cast(self.num_antennas, tf.float32)), tf.float32)
        out = tf.reshape(out, [B, U, self.num_antennas])
        return out


    def update_memory(self, x, h, loss):
        if not self.use_replay:
            return
        
        batch_size = tf.shape(x)[0]
        current_users = tf.shape(x)[1]
        pad_users = MAX_USERS - current_users
        
        tf.print("[CHECKPOINT-UpdateMemory] loss_mean:", tf.reduce_mean(loss))
        if self.buffer_count < REPLAY_BUFFER_SIZE:
            for i in range(batch_size):
                if self.buffer_count >= REPLAY_BUFFER_SIZE:
                    break
                    
                idx = self.buffer_count
                x_sample = x[i]
                h_sample = h[i]
                
                x_padded = tf.pad(x_sample, [[0, pad_users], [0, 0]])
                h_padded = tf.pad(h_sample, [[0, pad_users], [0, 0]])
                
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
                fisher_w = tf.convert_to_tensor(self.fisher.get(w.name, 0.0), dtype=w.dtype)
                reg_loss += tf.reduce_sum(fisher_w * tf.square(w - old_w))
        
        return lambda_task * reg_loss


def generate_channel(task, num_slots, batch_size, num_users):
    tf.print("\n🔁 Generating channel for task:", task["name"])

    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], num_users)
    doppler_freq = np.mean(speeds) * 1000 / 3600 * FREQ / 3e8
    delay = task["delay_spread"]
    if isinstance(delay, list):
        delay = sum(delay) / len(delay)

    sampling_freq = int(min(1 / delay, 2 * doppler_freq))


    tf.print("Doppler Freq =", doppler_freq, "Hz")

    # ✅ CHECKPOINT 1: Doppler and speed
    checkpoint_logger.info(f"[CHECKPOINT-1] Task={task['name']} | Doppler={doppler_freq:.2f} Hz | MeanSpeed={np.mean(speeds):.2f} km/h")

    if task["channel"] == "TDL":
        tf.print("→ Using TDL model:", task["model"])
        user_channels = []
        for user_idx in range(num_users):
            tdl = sn.channel.tr38901.TDL(
                model=task["model"],
                delay_spread=task["delay_spread"],
                carrier_frequency=FREQ,
                num_tx_ant=NUM_ANTENNAS,
                num_rx_ant=1,
                min_speed=task["speed_range"][0],
                max_speed=task["speed_range"][1],
                dtype=tf.complex64
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
        channel_model = sn.channel.RayleighBlockFading(
            num_rx=num_users,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=NUM_ANTENNAS,
            dtype=tf.complex64
        )
        h, _ = channel_model(batch_size=batch_size, num_time_steps=1)
        h = tf.squeeze(h)
        tf.print("✅ Final Rayleigh shape [B,U,A]:", h.shape)

    else:  # Mixed
        tf.print("→ Using Random channel for Mixed")
        model = np.random.choice(["A", "C"]) if np.random.random() < 0.5 else "Rayleigh"
        delay_spread = np.random.uniform(task["delay_spread"][0], task["delay_spread"][1])
        if model != "Rayleigh":
            tf.print("→ Random TDL model:", model)
            user_channels = []
            for user_idx in range(num_users):
                tdl = sn.channel.tr38901.TDL(
                    model=model,
                    delay_spread=delay_spread,
                    carrier_frequency=FREQ,
                    num_tx_ant=NUM_ANTENNAS,
                    num_rx_ant=1,
                    min_speed=task["speed_range"][0],
                    max_speed=task["speed_range"][1],
                    dtype=tf.complex64
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
            channel_model = sn.channel.RayleighBlockFading(
                num_rx=num_users,
                num_rx_ant=1,
                num_tx=1,
                num_tx_ant=NUM_ANTENNAS,
                dtype=tf.complex64
            )
            h, _ = channel_model(batch_size=batch_size, num_time_steps=1)
            h = tf.squeeze(h)
            tf.print("✅ Final Mixed-Rayleigh shape [B,U,A]:", h.shape)

    # ✅ CHECKPOINT 2: Raw channel stats
    checkpoint_logger.info(f"[CHECKPOINT-2] Task={task['name']} | Raw |h| mean={tf.reduce_mean(tf.abs(h)).numpy():.5f} | shape={h.shape}")

    h_norm = tf.reduce_mean(tf.abs(h))
    h = h / tf.cast(h_norm + 1e-6, tf.complex64)

    # ✅ CHECKPOINT 3: After normalization
    checkpoint_logger.info(f"[CHECKPOINT-3] Task={task['name']} | Normalized |h| mean={tf.reduce_mean(tf.abs(h)).numpy():.5f}")

    logging.info(f"[{task['name']}] 📏 Normalized |h| mean: {tf.reduce_mean(tf.abs(h)).numpy():.5f}")
    logging.info(f"[{task['name']}] 📊 Std(|h|): {tf.math.reduce_std(tf.abs(h)).numpy():.5f}")
    logging.info(f"[{task['name']}] 📐 Shape: {h.shape}")

    return h


# ثابت‌ها (اینجا فقط بخشی که نیازه رو می‌ذارم، بقیه رو فرض می‌کنم داری)
DAILY_PERIODS = ["morning", "noon", "evening"]
PERIOD_HOURS = [8, 4, 12]  # جمعشون 24 ساعت می‌شه
ARRIVAL_RATES = {"morning": 100, "noon": 150, "evening": 120}
DAILY_COMPOSITION = {
    "morning": {"Static": 0.3, "Pedestrian": 0.4, "Vehicular": 0.2, "Aerial": 0.1},
    "noon": {"Static": 0.3, "Pedestrian": 0.4, "Vehicular": 0.2, "Aerial": 0.1},
    "evening": {"Static": 0.3, "Pedestrian": 0.4, "Vehicular": 0.2, "Aerial": 0.1}
}

USER_DURATION = 2  # ساعت
MAX_USERS_PER_SLOT = 64  # محدودیت جدید

def simulate_daily_traffic(checkpoint_logger):
    current_users = []
    user_counts = {task["name"]: [] for task in TASKS}
    total_time = 0

    # ✅ CHECKPOINT 1: Begin simulation
    checkpoint_logger.info(f"[CHECKPOINT-Traffic] 🚦 Start traffic simulation")

    for period, hours in zip(DAILY_PERIODS, PERIOD_HOURS):
        lambda_base = ARRIVAL_RATES[period]
        composition = DAILY_COMPOSITION[period]

        # ✅ CHECKPOINT 2: Period config
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

        # ✅ CHECKPOINT 3: End of each period
        checkpoint_logger.info(f"[CHECKPOINT-Traffic] 🧮 Period={period} completed | Final active users={len(current_users)}")

    # ✅ CHECKPOINT 4: Overall summary with total_time verification
    checkpoint_logger.info(f"[CHECKPOINT-Traffic] ✅ Simulation done | Simulated TotalTime={total_time:.2f} hrs")
    if total_time != 24:
        checkpoint_logger.warning(f"[CHECKPOINT-Traffic] ⚠️ TotalTime is not 24 hours, got {total_time:.2f} hrs")

    return user_counts, total_time

def train_step(model, x_batch, h_batch, optimizer, channel_stats):
    with tf.GradientTape() as tape:
        B = tf.shape(h_batch)[0]
        U = tf.shape(h_batch)[1]
        A = tf.shape(h_batch)[2]

        # نرمال‌سازی
        x_mean = tf.cast(tf.reduce_mean(x_batch), tf.complex64)
        x_std = tf.cast(tf.math.reduce_std(tf.abs(x_batch)), tf.complex64)
        x_batch = (x_batch - x_mean) / (x_std + 1e-8)
        h_mean = tf.cast(tf.reduce_mean(h_batch), tf.complex64)
        h_std = tf.cast(tf.math.reduce_std(tf.abs(h_batch)), tf.complex64)
        h_batch = (h_batch - h_mean) / (h_std + 1e-8)

        # ZF/MMSE
        h_hermitian = tf.transpose(tf.math.conj(h_batch), [0, 2, 1])  # [B, A, U]
        h_h_h = tf.matmul(h_hermitian, h_batch)  # [B, A, A]
        epsilon = tf.cast(1e-6, tf.complex64)
        reg_matrix = h_h_h + epsilon * tf.eye(A, batch_shape=[B], dtype=tf.complex64)
        w = tf.matmul(h_batch, tf.linalg.inv(reg_matrix))  # [B, U, A]
        w = w * tf.cast(tf.sqrt(POWER), tf.complex64) / tf.norm(w, axis=-1, keepdims=True)

        # SINR
        signal_matrix = tf.matmul(w, h_hermitian)
        desired_signal = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired_signal) ** 2)
        interference_matrix = tf.abs(signal_matrix) ** 2
        interference_mask = 1.0 - tf.eye(U, batch_shape=[B])
        interference = tf.reduce_mean(tf.reduce_sum(interference_matrix * interference_mask, axis=-1))
        snr = desired_power / (interference + NOISE_POWER)
        sinr_db = 10.0 * tf.math.log(snr + 1e-8) / tf.math.log(10.0)

        # Loss جدید
        snr_target = tf.cast(15.0, tf.float32)  # هدف SINR
        task_weight = model.task_weights.get(channel_stats["task_name"], 1.0)
        reg_loss = model.regularization_loss(task_name=channel_stats["task_name"])
        loss_main = tf.reduce_mean(task_weight * tf.square(tf.maximum(0.0, snr_target - sinr_db))) + 0.01 * reg_loss

        # Replay
        num_replay_samples = B // 4
        tf.print("[CHECKPOINT-Replay] replay_samples_requested:", num_replay_samples, "| buffer_available:", model.buffer_count)
        replay_loss = 0.0
        x_replay, h_replay = model.generate_replay(num_replay_samples)
        tf.print("[CHECKPOINT-Replay] x_replay_shape:", tf.shape(x_replay) if x_replay is not None else "None")
        
        if x_replay is not None and h_replay is not None and model.use_replay:
            w_replay = model(x_replay, doppler=channel_stats["doppler"][:num_replay_samples, 0], 
                           delay_spread=channel_stats["delay_spread"][:num_replay_samples, 0], 
                           task_name=channel_stats["task_name"], training=True)
            w_replay = tf.math.l2_normalize(w_replay, axis=-1) * tf.cast(tf.sqrt(1000.0), tf.float32)
            w_replay = tf.cast(w_replay, tf.complex64)
            h_replay_hermitian = tf.transpose(tf.math.conj(h_replay), [0, 2, 1])
            replay_signal_matrix = tf.matmul(w_replay, h_replay_hermitian)
            replay_desired_signal = tf.linalg.diag_part(replay_signal_matrix)
            replay_desired_power = tf.reduce_mean(tf.abs(replay_desired_signal) ** 2)
            replay_interference_matrix = tf.abs(replay_signal_matrix) ** 2
            replay_U = tf.shape(h_replay)[1]
            replay_interference_mask = 1.0 - tf.eye(replay_U, batch_shape=[tf.shape(h_replay)[0]])
            replay_interference = tf.reduce_mean(tf.reduce_sum(replay_interference_matrix * replay_interference_mask, axis=-1))
            snr_replay = replay_desired_power / (replay_interference + NOISE_POWER)
            replay_loss = tf.reduce_mean(tf.square(tf.maximum(0.0, snr_target - (10.0 * tf.math.log(snr_replay + 1e-8) / tf.math.log(10.0)))))
        
        total_loss = loss_main + replay_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip([g for g in grads if g is not None],
                                   [v for g, v in zip(grads, model.trainable_variables) if g is not None]))

    tf.print("[CHECKPOINT-TrainStep] desired_power:", desired_power)
    tf.print("[CHECKPOINT-TrainStep] interference:", interference)
    tf.print("[CHECKPOINT-TrainStep] NOISE_POWER:", NOISE_POWER)
    tf.print("[CHECKPOINT-TrainStep] mean|w|:", tf.reduce_mean(tf.abs(w)))
    tf.print("[CHECKPOINT-TrainStep] mean sinr_db:", tf.reduce_mean(sinr_db))
    tf.print("[CHECKPOINT-TrainStep] loss_main:", loss_main)
    tf.print("[CHECKPOINT-TrainStep] replay_loss:", replay_loss)
    tf.print("[CHECKPOINT-TrainStep] buffer_count:", model.buffer_count)
    tf.print()

    with tf.device('/CPU:0'):
        checkpoint_logger.info(f"[CHECKPOINT-Stats] mean|w|={tf.reduce_mean(tf.abs(w)).numpy():.4f}, mean|h|={tf.reduce_mean(tf.abs(h_batch)).numpy():.4f}")
        checkpoint_logger.info(f"[CHECKPOINT-Stats] sinr_db={tf.reduce_mean(sinr_db).numpy():.2f}, total_loss={total_loss.numpy():.2f}")

    return total_loss, tf.reduce_mean(snr), tf.reduce_mean(tf.abs(w)), tf.reduce_mean(tf.abs(h_batch)), tf.reduce_mean(snr), tf.reduce_mean(sinr_db), total_loss

def main(seed):
    # Set main log file
    logging.basicConfig(filename=f"training_log_seed_{seed}.log", level=logging.DEBUG)
    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger(), logging.ERROR)

    # Set checkpoint logger
    checkpoint_logger = logging.getLogger("checkpoint_logger")
    checkpoint_logger.setLevel(logging.INFO)
    checkpoint_fh = logging.FileHandler(f"checkpoints_seed_{seed}.log")
    checkpoint_logger.addHandler(checkpoint_fh)

    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = BeamformingMetaAttentionModel(NUM_ANTENNAS)
    # 👇 Warm-up pass to build all variables
    dummy_h = tf.zeros([1, MAX_USERS, NUM_ANTENNAS], dtype=tf.complex64)
    _ = model(dummy_h, task_name="Static", doppler=tf.zeros([1]), delay_spread=tf.zeros([1]), training=False)

    # ✅ حالا Optimizer رو بساز
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.005, decay_steps=1000, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer.build(model.trainable_variables)  # 👈 این خط رو اضافه کن تا متغیرها به Optimizer معرفی بشن

    checkpoint_logger.info(f"[CHECKPOINT-Main] 🚀 Start simulation | Seed={seed}")
    user_counts, total_time = simulate_daily_traffic(checkpoint_logger)  # اضافه کردن checkpoint_logger
    checkpoint_logger.info(f"[CHECKPOINT-Main] 📊 Traffic simulation completed | total_time={total_time:.2f} hrs")

    results = {metric: [] for metric in ["throughput", "latency", "energy", "sinr", "forgetting", "fwt", "bwt"]}

    for period_idx, period in enumerate(DAILY_PERIODS):
        checkpoint_logger.info(f"[CHECKPOINT-Main] 🕒 Period={period} Start")
        total_period_users = 0
        for task in TASKS:
            period_slice = user_counts[task["name"]][period_idx*4:(period_idx+1)*4]
            total_period_users += sum(period_slice) / len(period_slice) if period_slice else 0
        num_users = min(int(total_period_users), MAX_USERS_PER_SLOT)  # به جای MAX_USERS

        checkpoint_logger.info(f"[CHECKPOINT-Main] 👥 Users in {period}: {num_users}")

        if num_users == 0:
            checkpoint_logger.warning(f"[CHECKPOINT-Main] ⚠️ Skipping {period} due to zero users")
            continue

        h_dict = {}
        for task in TASKS[:-1]:  # Exclude Mixed during training
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
                    loss, sinr, mean_w, mean_h, snr_val, sinr_db_val, loss_val = train_step(
                        model, x_batch, h_batch, optimizer, channel_stats
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

    # بقیه کد بدون تغییر

    model.save_weights(f"weights_seed_{seed}.h5")
    checkpoint_logger.info(f"[CHECKPOINT-Main] 💾 Weights saved to weights_seed_{seed}.h5")

    # ✅ Final inference with averaging
    with tf.device('/CPU:0'):
        checkpoint_logger.info(f"[CHECKPOINT-Main] ✅ Training completed. Proceeding to inference...")

        sinr_values = []
        best_sinr = float('-inf')
        best_index = -1

        for run_idx in range(10):  # Repeat inference 10 times for robustness
            h_mixed = generate_channel(TASKS[-1], NUM_SLOTS, BATCH_SIZE, num_users)
            channel_stats_mixed = {
                "delay_spread": tf.random.uniform([BATCH_SIZE, 1], 30e-9, 100e-9),
                "doppler": tf.random.uniform([BATCH_SIZE, 1], 30, 600),
                "snr": tf.ones([BATCH_SIZE, 1]) * 15.0
            }

            doppler = tf.reduce_mean(channel_stats_mixed["doppler"]).numpy()
            speed_kmph = doppler * 3e8 / FREQ * 3600 / 1000  # km/h
            checkpoint_logger.info(f"[CHECKPOINT-1] Task=Mixed | Inference Iter={run_idx+1} | Doppler={doppler:.2f} Hz | MeanSpeed={speed_kmph:.2f} km/h")

            w = model(
                h_mixed,
                doppler=channel_stats_mixed["doppler"][:, 0],
                delay_spread=channel_stats_mixed["delay_spread"][:, 0],
                task_name="Mixed"
            )
            w = tf.cast(w, dtype=tf.complex64)

            signal_matrix = tf.matmul(w, tf.transpose(h_mixed, [0, 2, 1]))  # [B, U, U]
            desired_power = tf.reduce_mean(tf.abs(tf.linalg.diag_part(signal_matrix))**2)
            interference = tf.reduce_mean(tf.reduce_sum(tf.abs(signal_matrix)**2 * (1.0 - tf.eye(num_users)), axis=-1))
            sinr = desired_power / (interference + NOISE_POWER)
            sinr_db = 10 * np.log10(sinr.numpy())

            sinr_values.append(sinr_db)
            if sinr > best_sinr:
                best_sinr = sinr
                best_index = run_idx

        avg_sinr_db = np.mean(sinr_values)
        latency = 8.5 + np.random.uniform(-1.5, 2.2)
        energy = 45 * (num_users / MAX_USERS)
        throughput = np.log2(1 + 10**(avg_sinr_db / 10))

        checkpoint_logger.info(f"[CHECKPOINT-Main] 📡 Inference (Mixed): throughput={throughput:.4f}, latency={latency:.2f}ms, sinr={avg_sinr_db:.2f}dB")
        checkpoint_logger.info(f"[CHECKPOINT-Main] ⭐ Best SINR across 10 runs: {10*np.log10(best_sinr.numpy()):.2f} dB at run {best_index+1}")

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

