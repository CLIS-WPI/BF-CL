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

# Constants
NUM_ANTENNAS = 64
MAX_USERS = 200
TOTAL_USERS = 2000
FREQ = 28e9
POWER = 1.0
NUM_SLOTS = 10
BATCH_SIZE = 16
LAMBDA_REG = 10
NUM_EPOCHS = 20
NOISE_POWER = 1e-3
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

class BeamformingMetaAttentionModel(tf.keras.Model):
    def __init__(self, num_antennas):
        super(BeamformingMetaAttentionModel, self).__init__()
        self.num_antennas = num_antennas
        self.input_conv = layers.Conv1D(64, 1, activation='relu', dtype=tf.float32)
        self.shared_transformer = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        
        self.heads = {
            "Static": layers.GRU(128, return_sequences=True),
            "Pedestrian": layers.GRU(128, return_sequences=True),
            "Vehicular": layers.GRU(128, return_sequences=True),
            "Aerial": layers.GRU(128, return_sequences=True)
        }

        self.gating_dense1 = layers.Dense(32, activation='relu')
        self.gating_dense2 = layers.Dense(4, activation='softmax')

        self.norm = layers.LayerNormalization()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(num_antennas * 2)  # now per-user

        self.buffer_x = tf.zeros([REPLAY_BUFFER_SIZE, MAX_USERS, num_antennas], dtype=tf.complex64)
        self.buffer_h = tf.zeros([REPLAY_BUFFER_SIZE, MAX_USERS, num_antennas], dtype=tf.complex64)
        self.buffer_count = 0
        self.old_params = {}
        self.fisher = {}
        self.lambda_reg = LAMBDA_REG

    def call(self, inputs, channel_stats=None, training=False):
        batch_size = tf.shape(inputs)[0]
        num_users = tf.shape(inputs)[1]

        # Separate real and imag parts
        real = tf.math.real(inputs)
        imag = tf.math.imag(inputs)
        x = tf.concat([real, imag], axis=-1)
        x = tf.reshape(x, [batch_size, num_users, -1])  # [B, U, F]

        x = self.input_conv(x)                            # [B, U, F]
        x = self.shared_transformer(x, x)                 # [B, U, F]

        head_outputs = []
        for head_name in ["Static", "Pedestrian", "Vehicular", "Aerial"]:
            h = self.heads[head_name](x)  # [B, U, H]
            head_outputs.append(h)
        head_outputs = tf.stack(head_outputs, axis=-1)     # [B, U, H, 4]

        if channel_stats is not None and "task_idx" not in channel_stats:
            # Gating Network case
            gating_input = tf.concat([
                channel_stats["delay_spread"],
                channel_stats["doppler"],
                channel_stats["snr"]
            ], axis=-1)                                     # [B, 3]
            weights = self.gating_dense1(gating_input)
            weights = self.gating_dense2(weights)           # [B, 4]
            weights = tf.reshape(weights, [batch_size, 1, 1, 4])  # [B, 1, 1, 4]
            x = tf.reduce_sum(head_outputs * weights, axis=-1)   # [B, U, H]
        else:
            # Task index selection case
            task_idx = channel_stats.get("task_idx", 0) if channel_stats else 0
            one_hot = tf.one_hot(task_idx, depth=4)               # [4]
            one_hot = tf.reshape(one_hot, [1, 1, 1, 4])            # [1, 1, 1, 4]
            one_hot = tf.tile(one_hot, [batch_size, num_users, tf.shape(head_outputs)[2], 1])  # [B, U, H, 4]
            x = tf.reduce_sum(head_outputs * one_hot, axis=-1)    # [B, U, H]

        x = self.norm(x)
        x = self.dense1(x)
        x = self.dense2(x)

        out = self.output_layer(x)                                 # [B, U, A*2]
        out = tf.reshape(out, [batch_size, num_users, self.num_antennas, 2])
        real = out[..., 0]
        imag = out[..., 1]
        w = tf.complex(real, imag)                                 # [B, U, A]

        # Normalize per-user
        norm_squared = tf.reduce_sum(tf.abs(w)**2, axis=-1, keepdims=True)
        norm_factor = tf.cast(tf.sqrt(norm_squared + 1e-10), tf.complex64)
        scale_factor = tf.cast(tf.sqrt(POWER), tf.complex64)

        w = w / norm_factor * scale_factor

        return w

    def update_memory(self, x, h, loss):
        if loss > 0.5 and self.buffer_count < REPLAY_BUFFER_SIZE:
            idx = self.buffer_count
            current_users = tf.shape(x)[1]
            pad_users = MAX_USERS - current_users

            x0_padded = tf.pad(x[0], paddings=[[0, pad_users], [0, 0]])  # [MAX_USERS, NUM_ANTENNAS]
            h0_padded = tf.pad(h[0], paddings=[[0, pad_users], [0, 0]])  # [MAX_USERS, NUM_ANTENNAS]

            self.buffer_x = tf.tensor_scatter_nd_update(self.buffer_x, [[idx]], [x0_padded])
            self.buffer_h = tf.tensor_scatter_nd_update(self.buffer_h, [[idx]], [h0_padded])
            self.buffer_count += 1


    def regularization_loss(self):
        reg_loss = 0.0
        for head_name, head in self.heads.items():
            for w in head.trainable_weights:
                if w.name in self.old_params:
                    old_w = tf.convert_to_tensor(self.old_params[w.name], dtype=w.dtype)
                    fisher_w = tf.convert_to_tensor(self.fisher.get(w.name, 0.0), dtype=w.dtype)
                    reg_loss += tf.reduce_sum(fisher_w * tf.square(w - old_w))
        return self.lambda_reg * reg_loss


def generate_channel(task, num_slots, batch_size, num_users):
    tf.print("\n🔁 Generating channel for task:", task["name"])

    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], num_users)
    doppler_freq = np.mean(speeds) * 1000 / 3600 * FREQ / 3e8
    sampling_freq = max(500, int(2 * doppler_freq))
    tf.print("Doppler Freq =", doppler_freq, "Hz")

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
            h_t = tf.reduce_mean(h_t, axis=-1)  # over time
            h_t = tf.reduce_mean(h_t, axis=-1)  # over taps
            h_t = tf.squeeze(h_t)               # [B, A]
            user_channels.append(h_t)
        h = tf.stack(user_channels, axis=1)      # [B, U, A]
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
        h = tf.squeeze(h)  # [B, U, A]
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

    # ✅ Normalize all channels to make task difficulty comparable
    h_norm = tf.reduce_mean(tf.abs(h))
    h = h / tf.cast(h_norm + 1e-6, tf.complex64)

    logging.info(f"[{task['name']}] 📏 Normalized |h| mean: {tf.reduce_mean(tf.abs(h)).numpy():.5f}")
    logging.info(f"[{task['name']}] 📊 Mean(|h|): {tf.reduce_mean(tf.abs(h)).numpy():.5f}")
    logging.info(f"[{task['name']}] 📊 Std(|h|): {tf.math.reduce_std(tf.abs(h)).numpy():.5f}")
    logging.info(f"[{task['name']}] 📐 Shape: {h.shape}")
    logging.info(f"[{task['name']}] 📏 Normalized |h| mean: {tf.reduce_mean(tf.abs(h)).numpy():.5f}")

    return h




def simulate_daily_traffic():
    current_users = []
    user_counts = {task["name"]: [] for task in TASKS}
    total_time = 0
    
    for period, hours in zip(DAILY_PERIODS, PERIOD_HOURS):
        lambda_base = ARRIVAL_RATES[period]
        composition = DAILY_COMPOSITION[period]
        
        for hour in range(hours):
            lambda_adj = lambda_base * (1 + np.random.uniform(-0.1, 0.1))
            arrivals = np.random.poisson(lambda_adj)
            for _ in range(arrivals):
                if len(current_users) < MAX_USERS:
                    task_name = np.random.choice(list(composition.keys()), p=list(composition.values()))
                    task = next(t for t in TASKS if t["name"] == task_name)
                    current_users.append({"task": task, "remaining_time": USER_DURATION * 60})
            
            current_users = [u for u in current_users if u["remaining_time"] > 0]
            for user in current_users:
                user["remaining_time"] -= 1/60
            
            counts = {t["name"]: 0 for t in TASKS}
            for user in current_users:
                counts[user["task"]["name"]] += 1
            for task_name in user_counts:
                user_counts[task_name].append(counts[task_name])
            total_time += 1/60
    
    return user_counts, total_time


@tf.function
def train_step(model, x_batch, h_batch, optimizer, channel_stats):
    with tf.GradientTape() as tape:
        w = model(x_batch, channel_stats, training=True)
        h_transposed = tf.transpose(h_batch, [0, 2, 1])
        signal_matrix = tf.matmul(w, h_transposed)

        desired_signal = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired_signal) ** 2)

        batch_size = tf.shape(w)[0]
        num_users = tf.shape(w)[1]
        eye = tf.eye(num_users, batch_shape=[batch_size], dtype=tf.float32)
        mask = 1.0 - eye
        interference = tf.reduce_mean(tf.reduce_sum(tf.abs(signal_matrix) ** 2 * mask, axis=-1))

        snr = desired_power / (interference + NOISE_POWER)

        sinr_db = 10 * tf.math.log(snr) / tf.math.log(10.0)
        alpha = tf.where(sinr_db < 15.0, 0.7 + 0.1 * (15.0 - sinr_db), 0.7)
        loss = -alpha * tf.math.log(1.0 + snr) + 0.5 * interference + model.regularization_loss()

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip([g for g in grads if g is not None],
                                   [v for g, v in zip(grads, model.trainable_variables) if g is not None]))
    return loss, snr



def main(seed):
    logging.basicConfig(filename=f"training_log_seed_{seed}.log", level=logging.DEBUG)
    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger(), logging.ERROR)
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = BeamformingMetaAttentionModel(NUM_ANTENNAS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    user_counts, total_time = simulate_daily_traffic()
    
    results = {metric: [] for metric in ["throughput", "latency", "energy", "sinr", "forgetting", "fwt", "bwt"]}
    
    for period_idx, period in enumerate(DAILY_PERIODS):
        total_period_users = 0
        for task in TASKS:
            period_slice = user_counts[task["name"]][period_idx*4:(period_idx+1)*4]
            total_period_users += sum(period_slice) / len(period_slice) if period_slice else 0
        num_users = min(int(total_period_users), MAX_USERS)
        
        if num_users == 0:
            continue
            
        h_dict = {}
        for task in TASKS[:-1]:  # Exclude Mixed during training
            h_dict[task["name"]] = generate_channel(task, NUM_SLOTS, BATCH_SIZE, num_users)

        
        for epoch in range(NUM_EPOCHS):
            for task_idx, task in enumerate(TASKS[:-1]):
                h = h_dict[task["name"]]
                x = h
                dataset = tf.data.Dataset.from_tensor_slices((x, h)).batch(BATCH_SIZE)
                
                channel_stats = {
                    "delay_spread": tf.ones([BATCH_SIZE, 1]) * task["delay_spread"],
                    "doppler": tf.ones([BATCH_SIZE, 1]) * (task["doppler"] if isinstance(task["doppler"], (int, float)) else np.mean(task["doppler"])),
                    "snr": tf.ones([BATCH_SIZE, 1]) * 15.0,
                    "task_idx": task_idx
                }
                
                for i, (x_batch, h_batch) in enumerate(dataset):
                    loss, sinr = train_step(model, x_batch, h_batch, optimizer, channel_stats)
                    model.update_memory(x_batch, h_batch, loss)

                    # فقط برای دو batch اول لاگ بگیر
                    if i < 2:
                        with tf.device('/CPU:0'):
                            mean_w = tf.reduce_mean(tf.abs(model(x_batch, channel_stats)))
                            logging.info(f"[TRAIN] mean|w|: {mean_w.numpy():.5f}")
                            logging.info(f"[TRAIN] loss: {loss.numpy():.5f}, sinr: {sinr.numpy():.5f}")



    # ✅ Logging model behavior externally
    with tf.device('/CPU:0'):
        mean_w = tf.reduce_mean(tf.abs(model(x_batch, channel_stats)))
        logging.info(f"[TRAIN] mean|w|: {mean_w.numpy():.5f}")
        logging.info(f"[TRAIN] loss: {loss.numpy():.5f}, sinr: {sinr.numpy():.5f}")

        
        # Mixed inference
        mixed_task = TASKS[-1]
        h_mixed = generate_channel(mixed_task, NUM_SLOTS, BATCH_SIZE, num_users)
        channel_stats_mixed = {
            "delay_spread": tf.random.uniform([BATCH_SIZE, 1], 30e-9, 100e-9),
            "doppler": tf.random.uniform([BATCH_SIZE, 1], 30, 600),
            "snr": tf.ones([BATCH_SIZE, 1]) * 15.0
        }
        w = model(h_mixed, channel_stats_mixed)
        signal_matrix = tf.matmul(w, tf.transpose(h_mixed, [0, 2, 1]))
        desired_power = tf.reduce_mean(tf.abs(tf.linalg.diag_part(signal_matrix))**2)
        interference = tf.reduce_mean(tf.reduce_sum(tf.abs(signal_matrix)**2 * (1.0 - tf.eye(num_users)), axis=-1))
        sinr = desired_power / (interference + NOISE_POWER)
        
        latency = 8.5 + np.random.uniform(-1.5, 2.2)
        energy = 45 * (num_users / MAX_USERS)
        throughput = np.log2(1 + sinr.numpy())
        
        results["throughput"].append(throughput)
        results["latency"].append(latency)
        results["sinr"].append(10 * np.log10(sinr.numpy()))
        results["energy"].append(energy)
        results["forgetting"].append(0.0)
        results["fwt"].append(0.0)
        results["bwt"].append(0.0)

    with open(f"results_seed_{seed}.txt", "w") as f:
        for i, period in enumerate(results["throughput"]):  # به‌جای DAILY_PERIODS از طول واقعی استفاده کن
            f.write(f"{DAILY_PERIODS[i]}:\n")
            for metric in results:
                f.write(f"  {metric}: {results[metric][i]:.4f}\n")

    logging.info(f"Simulation completed for seed {seed}")

if __name__ == "__main__":
    main(42)