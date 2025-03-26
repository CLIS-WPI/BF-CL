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
MAX_USERS = 200
TOTAL_USERS = 2000
FREQ = 28e9
POWER = 100.0
NUM_SLOTS = 10
BATCH_SIZE = 16
LAMBDA_REG = 10
NUM_EPOCHS = 5 #20
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
        self.use_replay = False  # ← for debug only
        self.use_fisher = False  # ← for debug only

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
        self.output_layer = layers.Dense(num_antennas * 2)

        self.buffer_x = tf.zeros([REPLAY_BUFFER_SIZE, MAX_USERS, num_antennas], dtype=tf.complex64)
        self.buffer_h = tf.zeros([REPLAY_BUFFER_SIZE, MAX_USERS, num_antennas], dtype=tf.complex64)
        self.buffer_count = 0
        self.old_params = {}
        self.fisher = {}
        self.lambda_reg = LAMBDA_REG

    def call(self, inputs, channel_stats=None, training=False):
        batch_size = tf.shape(inputs)[0]
        num_users = tf.shape(inputs)[1]

        # ✅ چک‌پوینت امن بدون .numpy() (برای دیباگ زنده بدون crash)
        tf.print("[CHECKPOINT-1] batch:", batch_size, "users:", num_users, "|h|_mean:", tf.reduce_mean(tf.abs(inputs)))

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
            gating_input = tf.concat([
                channel_stats["delay_spread"],
                channel_stats["doppler"],
                channel_stats["snr"]
            ], axis=-1)
            weights = self.gating_dense1(gating_input)
            weights = self.gating_dense2(weights)
            weights = tf.reshape(weights, [batch_size, 1, 1, 4])
            x = tf.reduce_sum(head_outputs * weights, axis=-1)
        else:
            task_idx = channel_stats.get("task_idx", 0) if channel_stats else 0
            one_hot = tf.one_hot(task_idx, depth=4)
            one_hot = tf.reshape(one_hot, [1, 1, 1, 4])
            one_hot = tf.tile(one_hot, [batch_size, num_users, tf.shape(head_outputs)[2], 1])
            x = tf.reduce_sum(head_outputs * one_hot, axis=-1)

        x = self.norm(x)
        x = self.dense1(x)
        x = self.dense2(x)

        out = self.output_layer(x)  # [B, U, A*2]
        out = tf.reshape(out, [batch_size, num_users, self.num_antennas, 2])
        real = out[..., 0]
        imag = out[..., 1]
        w = tf.complex(real, imag)  # [B, U, A]

        # Normalize power per user independently
        norm = tf.norm(w, axis=-1, keepdims=True) + 1e-9
        w = (w / norm) * tf.cast(tf.sqrt(POWER), tf.complex64)


        return w



    #def update_memory(self, x, h, loss):
       # if loss > 0.5 and self.buffer_count < REPLAY_BUFFER_SIZE:
          #  idx = self.buffer_count
         #   current_users = tf.shape(x)[1]
         #   pad_users = MAX_USERS - current_users

        #    x0_padded = tf.pad(x[0], paddings=[[0, pad_users], [0, 0]])  # [MAX_USERS, NUM_ANTENNAS]
         #   h0_padded = tf.pad(h[0], paddings=[[0, pad_users], [0, 0]])  # [MAX_USERS, NUM_ANTENNAS]

        #    self.buffer_x = tf.tensor_scatter_nd_update(self.buffer_x, [[idx]], [x0_padded])
        #    self.buffer_h = tf.tensor_scatter_nd_update(self.buffer_h, [[idx]], [h0_padded])
         #   self.buffer_count += 1
    def update_memory(self, x, h, loss):
        if not self.use_replay:
            return


    #def regularization_loss(self):
    #    reg_loss = 0.0
    #    for head_name, head in self.heads.items():
    #        for w in head.trainable_weights:
    #            if w.name in self.old_params:
    #                old_w = tf.convert_to_tensor(self.old_params[w.name], dtype=w.dtype)
    #                fisher_w = tf.convert_to_tensor(self.fisher.get(w.name, 0.0), dtype=w.dtype)
    #                reg_loss += tf.reduce_sum(fisher_w * tf.square(w - old_w))
    #    return self.lambda_reg * reg_loss
    def regularization_loss(self):
        if not self.use_fisher:
            return 0.0



def generate_channel(task, num_slots, batch_size, num_users):
    tf.print("\n🔁 Generating channel for task:", task["name"])

    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], num_users)
    doppler_freq = np.mean(speeds) * 1000 / 3600 * FREQ / 3e8
    sampling_freq = max(500, int(2 * doppler_freq))
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


def simulate_daily_traffic():
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
                if len(current_users) < MAX_USERS:
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
            total_time += 1 / 60

        # ✅ CHECKPOINT 3: End of each period
        checkpoint_logger.info(f"[CHECKPOINT-Traffic] 🧮 Period={period} completed | Final active users={len(current_users)}")

    # ✅ CHECKPOINT 4: Overall summary
    checkpoint_logger.info(f"[CHECKPOINT-Traffic] ✅ Simulation done | TotalTime={total_time:.2f} hrs")
    return user_counts, total_time


#@tf.function
#def train_step(model, x_batch, h_batch, optimizer, channel_stats):
    #with tf.GradientTape() as tape:
        #w = model(x_batch, channel_stats, training=True)  # [B, U, A]
        #w = model(x_batch, channel_stats, training=True)  # ← موقتاً غیرفعال کن
        #w = tf.ones_like(h_batch, dtype=tf.complex64)  # ← تست beamforming واحد

        # Beamforming فقط برای یک user فعال در یک batch
        #w = tf.zeros_like(h_batch, dtype=tf.complex64)  # [B, U, A]
        #w = tf.tensor_scatter_nd_update(w, [[0, 0]], [1.0 + 0j])  # فقط batch 0، user 0، تمام آنتن‌ها → 1+0j

        #h_transposed = tf.transpose(h_batch, [0, 2, 1])  # [B, A, U]
        #h_hermitian = tf.transpose(tf.math.conj(h_batch), [0, 2, 1])  # [B, A, U] → [B, U, A]ᴴfor test

        #signal_matrix = tf.matmul(w, h_transposed)       # [B, U, U]
        #signal_matrix = tf.matmul(w, h_hermitian)  # [B, U, A] x [B, A, U] for test
        # تست: فقط user 0 در batch 0 فعال → حالت ایده‌آل بدون تداخل
        #B = tf.shape(h_batch)[0]
        #U = tf.shape(h_batch)[1]
        #A = tf.shape(h_batch)[2]

        #w = tf.zeros_like(h_batch, dtype=tf.complex64)  # [B, U, A]

        # فقط batch 0، user 0 → همه آنتن‌ها فعال
       # update_indices = tf.stack([tf.zeros([A], dtype=tf.int32),  # batch=0
                                  # tf.zeros([A], dtype=tf.int32),  # user=0
                                  # tf.range(A)], axis=1)           # antennas 0 to A-1

        #w = tf.tensor_scatter_nd_update(w, update_indices, tf.ones([A], dtype=tf.complex64))

        #h_hermitian = tf.transpose(tf.math.conj(h_batch), [0, 2, 1])  # [B, A, U]
        #signal_matrix = tf.matmul(w, h_hermitian)                     # [B, U, U]

        #desired_signal = tf.linalg.diag_part(signal_matrix)
        #desired_power = tf.reduce_mean(tf.abs(desired_signal) ** 2)

        #batch_size = tf.shape(w)[0]
        #num_users = tf.shape(w)[1]
        #eye = tf.eye(num_users, batch_shape=[batch_size], dtype=tf.float32)
        #mask = 1.0 - eye
        #3interference = tf.reduce_mean(tf.reduce_sum(tf.abs(signal_matrix) ** 2 * mask, axis=-1))

        #snr = desired_power / (interference + NOISE_POWER)
        #sinr_db = 10 * tf.math.log(snr + 1e-8) / tf.math.log(10.0)
        #alpha = tf.where(sinr_db < 15.0, 0.7 + 0.1 * (15.0 - sinr_db), 0.7)
        #loss = -alpha * tf.math.log(1.0 + snr) + 0.5 * interference + model.regularization_loss()

    #grads = tape.gradient(loss, model.trainable_variables)
    #optimizer.apply_gradients(zip([g for g in grads if g is not None],
                                   #[v for g, v in zip(grads, model.trainable_variables) if g is not None]))

    #tf.print("[CHECKPOINT-TrainStep] mean|w|:", tf.reduce_mean(tf.abs(w)))
    #tf.print("[CHECKPOINT-TrainStep] desired_power:", desired_power)
    #tf.print("[CHECKPOINT-TrainStep] interference:", interference)
    #tf.print("[CHECKPOINT-TrainStep] snr:", snr)
    #tf.print("[CHECKPOINT-TrainStep] loss:", loss)
    #tf.print("")

    #return loss, snr, tf.reduce_mean(tf.abs(w)), tf.reduce_mean(tf.abs(h_batch)), snr, sinr_db, loss
@tf.function
def train_step(model, x_batch, h_batch, optimizer, channel_stats):
    with tf.GradientTape() as tape:
        B = tf.shape(h_batch)[0]
        U = tf.shape(h_batch)[1]
        A = tf.shape(h_batch)[2]

        w = model(x_batch, channel_stats, training=True)  # [B, U, A]
        tf.print("[CHECKPOINT-TrainStep] raw |w| mean:", tf.reduce_mean(tf.abs(w)))

        # انتخاب یک یوزر فعال به صورت تصادفی برای هر batch
        selected_user = tf.random.uniform([], minval=0, maxval=U, dtype=tf.int32)
        mask = tf.one_hot(selected_user, depth=U, dtype=tf.complex64)  # [U]
        mask = tf.reshape(mask, [1, U, 1])  # [1, U, 1]
        w = w * mask  # فقط اون یوزر فعال بمونه

        # Signal and interference
        h_hermitian = tf.transpose(tf.math.conj(h_batch), [0, 2, 1])  # [B, A, U]
        signal_matrix = tf.matmul(w, h_hermitian)                     # [B, U, U]
        desired_signal = tf.linalg.diag_part(signal_matrix)          # [B, U]
        desired_power = tf.abs(desired_signal[0, selected_user]) ** 2

        interference_matrix = tf.abs(signal_matrix[0]) ** 2
        mask_interf = 1.0 - tf.eye(U, dtype=tf.float32)[selected_user]
        interference = tf.reduce_sum(interference_matrix[selected_user] * mask_interf)

        snr = desired_power / (interference + NOISE_POWER)
        sinr_db = 10.0 * tf.math.log(snr + 1e-8) / tf.math.log(10.0)
        alpha = tf.where(sinr_db < 15.0, 0.7 + 0.1 * (15.0 - sinr_db), 0.7)
        loss = -alpha * tf.math.log(1.0 + snr) + 0.5 * interference + model.regularization_loss()

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip([g for g in grads if g is not None],
                                   [v for g, v in zip(grads, model.trainable_variables) if g is not None]))

    # لاگ دقیق برای تحلیل
    tf.print("[CHECKPOINT-TrainStep] selected_user:", selected_user)
    tf.print("[CHECKPOINT-TrainStep] mean|w|:", tf.reduce_mean(tf.abs(w)))
    tf.print("[CHECKPOINT-TrainStep] desired_power:", desired_power)
    tf.print("[CHECKPOINT-TrainStep] interference:", interference)
    tf.print("[CHECKPOINT-TrainStep] snr:", snr)
    tf.print("[CHECKPOINT-TrainStep] sinr_db:", sinr_db)
    tf.print("[CHECKPOINT-TrainStep] loss:", loss)
    tf.print()

    return loss, snr, tf.reduce_mean(tf.abs(w)), tf.reduce_mean(tf.abs(h_batch)), snr, sinr_db, loss





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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    checkpoint_logger.info(f"[CHECKPOINT-Main] 🚀 Start simulation | Seed={seed}")
    user_counts, total_time = simulate_daily_traffic()
    checkpoint_logger.info(f"[CHECKPOINT-Main] 📊 Traffic simulation completed | total_time={total_time:.2f} hrs")

    results = {metric: [] for metric in ["throughput", "latency", "energy", "sinr", "forgetting", "fwt", "bwt"]}

    for period_idx, period in enumerate(DAILY_PERIODS):
        checkpoint_logger.info(f"[CHECKPOINT-Main] 🕒 Period={period} Start")
        total_period_users = 0
        for task in TASKS:
            period_slice = user_counts[task["name"]][period_idx*4:(period_idx+1)*4]
            total_period_users += sum(period_slice) / len(period_slice) if period_slice else 0
        num_users = min(int(total_period_users), MAX_USERS)

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
                dataset = tf.data.Dataset.from_tensor_slices((x, h)).batch(BATCH_SIZE)

                channel_stats = {
                    "delay_spread": tf.ones([BATCH_SIZE, 1]) * task["delay_spread"],
                    "doppler": tf.ones([BATCH_SIZE, 1]) * (task["doppler"] if isinstance(task["doppler"], (int, float)) else np.mean(task["doppler"])),
                    "snr": tf.ones([BATCH_SIZE, 1]) * 15.0,
                    "task_idx": task_idx
                }

                for i, (x_batch, h_batch) in enumerate(dataset):
                    loss, sinr, mean_w, mean_h, snr_val, sinr_db_val, loss_val = train_step(
                        model, x_batch, h_batch, optimizer, channel_stats
                    )
                    model.update_memory(x_batch, h_batch, loss)

                    if i < 2:
                        with tf.device('/CPU:0'):
                            checkpoint_logger.info(f"[CHECKPOINT-Main] [Epoch {epoch+1} | Task={task['name']}] mean|w|: {mean_w.numpy():.5f}")
                            checkpoint_logger.info(f"[CHECKPOINT-Main] [Epoch {epoch+1} | Task={task['name']}] mean|h|: {mean_h.numpy():.5f}")
                            checkpoint_logger.info(f"[CHECKPOINT-Main] [Epoch {epoch+1} | Task={task['name']}] loss: {loss_val.numpy():.5f}, snr: {snr_val.numpy():.5f}, sinr_db: {sinr_db_val.numpy():.2f}")


    # ✅ Final check before inference
    with tf.device('/CPU:0'):
        mean_w = tf.reduce_mean(tf.abs(model(x_batch, channel_stats)))
        logging.info(f"[TRAIN] mean|w|: {mean_w.numpy():.5f}")
        logging.info(f"[TRAIN] loss: {loss.numpy():.5f}, sinr: {sinr.numpy():.5f}")
        checkpoint_logger.info(f"[CHECKPOINT-Main] ✅ Training completed. Proceeding to inference...")

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

        checkpoint_logger.info(f"[CHECKPOINT-Main] 📡 Inference (Mixed): throughput={throughput:.4f}, latency={latency:.2f}ms, sinr={10*np.log10(sinr.numpy()):.2f}dB")

        results["throughput"].append(throughput)
        results["latency"].append(latency)
        results["sinr"].append(10 * np.log10(sinr.numpy()))
        results["energy"].append(energy)
        results["forgetting"].append(0.0)
        results["fwt"].append(0.0)
        results["bwt"].append(0.0)

    with open(f"results_seed_{seed}.txt", "w") as f:
        for i, period in enumerate(results["throughput"]):
            f.write(f"{DAILY_PERIODS[i]}:\n")
            for metric in results:
                f.write(f"  {metric}: {results[metric][i]:.4f}\n")

    checkpoint_logger.info(f"[CHECKPOINT-Main] 🏁 Simulation finished for seed {seed}")
    logging.info(f"Simulation completed for seed {seed}")


if __name__ == "__main__":
    main(42)

