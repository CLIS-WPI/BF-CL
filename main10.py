
import tensorflow as tf
import numpy as np
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
from tqdm import tqdm
import time
from tensorflow.keras import layers
import os
import sys

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ Using only GPU:0 →", gpus[0])
    except RuntimeError as e:
        print("❌ Error setting GPU visibility:", e)

# Settings
NUM_ANTENNAS = 64
FREQ = 28e9
NUM_SLOTS = 10
BATCH_SIZE = 4
NUM_USERS = 6
USER_DURATION = 2  # hours

ARRIVAL_RATES = {"morning": 50, "noon": 75, "evening": 60}
PERIOD_HOURS = {"morning": 8, "noon": 4, "evening": 12}
DAILY_COMPOSITION = {
    "morning": {"Pedestrian": 0.40, "Static": 0.30, "Vehicular": 0.20, "Aerial": 0.10},
    "noon": {"Vehicular": 0.50, "Pedestrian": 0.20, "Static": 0.20, "Aerial": 0.10},
    "evening": {"Aerial": 0.30, "Vehicular": 0.30, "Pedestrian": 0.20, "Static": 0.20}
}

TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": [30e-9, 50e-9], "doppler": [10, 50], "coherence_time": 0.423 / 50, "channel": "TDL", "model": "A"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": [50e-9, 100e-9], "doppler": [50, 150], "coherence_time": 0.423 / 150, "channel": "Rayleigh"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": [200e-9, 500e-9], "doppler": [500, 2000], "coherence_time": 0.423 / 2000, "channel": "TDL", "model": "C"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": [100e-9, 300e-9], "doppler": [200, 1000], "coherence_time": 0.423 / 1000, "channel": "TDL", "model": "A"},
]

class BeamformingMetaAttentionModel(tf.keras.Model):
    def __init__(self, num_antennas, num_users, num_tasks, use_replay=True, use_fisher=True):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.num_tasks = num_tasks
        self.use_replay = use_replay
        self.use_fisher = use_fisher
        self.hidden_dim = 128
        self.lambda_reg = 10.0
        self.concat_proj = tf.keras.layers.Dense(self.hidden_dim)
        self.feat_proj = tf.keras.layers.Dense(64, activation='relu')

        self.conv1 = tf.keras.layers.Conv1D(64, 1, activation='relu')
        self.mha_shared = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.norm = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.out = tf.keras.layers.Dense(num_antennas * num_users * 2)
        self.x_feat_proj_layer = tf.keras.layers.Dense(self.hidden_dim)
        
        dummy_input = tf.random.normal([1, num_users, self.hidden_dim])

        # Create task-specific GRU and gating layers as attributes
        for i in range(num_tasks):
            setattr(self, f"gru_task_{i}", tf.keras.layers.GRU(self.hidden_dim, return_sequences=True))
            setattr(self, f"gate_task_{i}", tf.keras.layers.Dense(self.hidden_dim, activation="sigmoid"))
            _ = getattr(self, f"gru_task_{i}")(dummy_input)
            _ = getattr(self, f"gate_task_{i}")(dummy_input)

        # Optional buffers
        if self.use_replay:
            self.replay_x = [[] for _ in range(num_tasks)]
            self.replay_h = [[] for _ in range(num_tasks)]
            self.replay_limit = 1000

        if self.use_fisher:
            self.old_params = {}
            self.fisher = {}

    def call(self, x, channel_features, task_idx, training=False):
        batch_size = tf.shape(x)[0]

        # Convert complex → real
        channel_feat_proj = self.feat_proj(channel_features)  # [B, U, F]
        real = tf.math.real(x)
        imag = tf.math.imag(x)
        x_split = tf.concat([real, imag], axis=-1)  # [B, U, 2*A]
        x_feat = self.conv1(x_split)  # [B, U, D]

        # Concatenate channel features
        x_feat = tf.concat([x_feat, channel_feat_proj], axis=-1)  # [B, U, D+F]
        x_feat = self.concat_proj(x_feat)  # [B, U, hidden_dim]

        # Shared Multi-Head Attention
        x_attn = self.mha_shared(x_feat, x_feat)  # [B, U, hidden_dim]

        # 🧠 Task-specific GRU and Gate (hard selection)
        gru_layer = getattr(self, f"gru_task_{task_idx}")
        gate_layer = getattr(self, f"gate_task_{task_idx}")
        gru_out = gru_layer(x_attn)  # [B, U, hidden_dim]
        gate = gate_layer(x_attn)   # [B, U, hidden_dim]
        x_gated = tf.multiply(gru_out, gate)  # [B, U, hidden_dim]

        # Residual + LayerNorm
        x_proj = self.x_feat_proj_layer(x_feat)  # [B, U, hidden_dim]
        x_norm = self.norm(x_gated + x_proj)     # [B, U, hidden_dim]

        # MLP → Beamforming Weights
        x_flat = tf.reduce_mean(x_norm, axis=1)  # [B, hidden_dim]
        x_out = self.fc2(self.fc1(x_flat))       # [B, 128]
        out = self.out(x_out)                    # [B, U*A*2]

        # Convert to complex weights
        real = out[:, :self.num_users * self.num_antennas]
        imag = out[:, self.num_users * self.num_antennas:]
        w = tf.complex(real, imag)
        w = tf.reshape(w, [batch_size, self.num_users, self.num_antennas])  # [B, U, A]

        # Normalize (per-user)
        w = tf.nn.l2_normalize(w, axis=-1)
        # Add this right after w = tf.nn.l2_normalize(...):
        norm_mean = tf.reduce_mean(tf.norm(w, axis=-1))
        tf.print("📡 Norm(w) =", norm_mean, output_stream=sys.stdout)
        _ = tf.identity(norm_mean)  # Force execution

        norm_mean = tf.reduce_mean(tf.norm(w, axis=-1))
        tf.print("📡 Norm(w) =", norm_mean, output_stream=sys.stdout)
        _ = tf.identity(norm_mean)  # ✅ force execution

        return w

    def update_replay(self, x, h, task_idx):
        if not self.use_replay:
            return
        if len(self.replay_x[task_idx]) >= self.replay_limit:
            self.replay_x[task_idx] = self.replay_x[task_idx][-self.replay_limit // 2:]
            self.replay_h[task_idx] = self.replay_h[task_idx][-self.replay_limit // 2:]

        # ✅ Split real & imag parts and store as float32
        real = tf.math.real(x)
        imag = tf.math.imag(x)
        x_split = tf.concat([real, imag], axis=-1)  # shape: [B, U, 2*A]
        self.replay_x[task_idx].extend(tf.unstack(x_split))
        self.replay_h[task_idx].extend(tf.unstack(h))


    def sample_replay(self, task_idx, num_samples):
        if not self.use_replay or len(self.replay_x[task_idx]) == 0:
            return None, None
        idx = np.random.choice(len(self.replay_x[task_idx]), size=num_samples)
        x = tf.stack([self.replay_x[task_idx][i] for i in idx])  # shape: [B, U, 2*A]
        h = tf.stack([self.replay_h[task_idx][i] for i in idx])  # shape: [B, U, A]

        # ✅ Convert back to complex
        a = self.num_antennas
        real = x[..., :a]
        imag = x[..., a:]
        x_complex = tf.complex(real, imag)  # shape: [B, U, A]

        return x_complex, h

    def update_fisher_info(self, x, h, task_idx):
        if not self.use_fisher: return
        with tf.GradientTape() as tape:
            channel_dummy = tf.zeros([tf.shape(x)[0], tf.shape(x)[1], 3], dtype=tf.float32)

            w = self(x, channel_dummy, task_idx, training=True)

            loss = tf.reduce_mean(tf.abs(w)**2)
        grads = tape.gradient(loss, self.trainable_variables)
        for v, g in zip(self.trainable_variables, grads):
            if g is not None:
                self.fisher[v.name] = tf.square(g)
        self.old_params = {v.name: v.numpy() for v in self.trainable_variables}

    def regularization_loss(self):
        if not self.use_fisher: return 0.0
        reg = 0.0
        for v in self.trainable_variables:
            if v.name in self.fisher and v.name in self.old_params:
                reg += tf.reduce_sum(self.fisher[v.name] * tf.square(v - self.old_params[v.name]))
        return self.lambda_reg * reg
    
    def generate_synthetic_batch(self, task, batch_size=8):
        h_users = []
        features = []
        for _ in range(self.num_users):
            delay = np.random.uniform(*task["delay_spread"])
            doppler = np.random.uniform(*task["doppler"])
            snr = np.random.uniform(5, 25)
            sampling_freq = int(min(1 / delay, 2 * doppler))

            if task["channel"] == "TDL":
                tdl = TDL(
                    model=task.get("model", "A"),
                    delay_spread=delay,
                    carrier_frequency=FREQ,
                    num_tx_ant=self.num_antennas,
                    num_rx_ant=1,
                    min_speed=task["speed_range"][0],
                    max_speed=task["speed_range"][1],
                )
                h, _ = tdl(batch_size=batch_size, num_time_steps=NUM_SLOTS, sampling_frequency=sampling_freq)
                h = tf.reduce_mean(h, axis=[-1, -2])
            else:
                rb = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=self.num_antennas)
                h, _ = rb(batch_size=batch_size, num_time_steps=1)
                h = tf.squeeze(h, axis=1)

            h = tf.reshape(h, [batch_size, 1, self.num_antennas])
            h_b = apply_beamspace(h)[:, 0, :]
            h_users.append(h_b)

            f = tf.tile([[doppler, delay, snr]], [batch_size, 1])  # [B, 3]
            features.append(f)

        h = tf.stack(h_users, axis=1)  # [B, U, A]
        h = h / tf.cast(tf.norm(h, axis=-1, keepdims=True) + 1e-8, tf.complex64)
        feats = tf.stack(features, axis=1)  # [B, U, 3]
        return tf.identity(h), tf.identity(h), feats  # ولی شاید بهتر باشه x یه چیز جدای از h باشه.

    def eval_kpi(self, w_pred, h_true):
        h_adj = tf.transpose(h_true, [0, 2, 1])
        signal_matrix = tf.matmul(w_pred, h_adj)
        desired = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired)**2)
        mask = 1.0 - tf.eye(self.num_users, dtype=tf.float32)
        mask = tf.tile(mask[None, :, :], [tf.shape(h_true)[0], 1, 1])
        interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
        interference_power = tf.reduce_mean(interference)

        sinr = desired_power / (interference_power + 1e-3)
        throughput = tf.reduce_mean(tf.math.log(1.0 + sinr) / tf.math.log(2.0))  # bps/Hz
        latency = tf.constant(2.0) * (1.0 / (1.0 + sinr)) * 10  # dummy model → scale if needed
        sinr_db = 10.0 * tf.math.log(sinr) / tf.math.log(10.0)
        return float(tf.reduce_mean(sinr_db)), float(throughput), float(tf.reduce_mean(latency))

    def get_task_weights(self, task_idx):
        return [v.numpy() for v in self.trainable_variables]

    def compute_forgetting(self, prev_weights, task_idx):
        current_weights = self.get_task_weights(task_idx)
        diffs = [np.mean((cw - pw)**2) for cw, pw in zip(current_weights, prev_weights)]
        return np.mean(diffs)

def apply_beamspace(h):
    dft_matrix = tf.signal.fft(tf.eye(NUM_ANTENNAS, dtype=tf.complex64))
    return tf.einsum("bua,ac->buc", h, dft_matrix)

def generate_user_channel(task):
    delay = np.random.uniform(*task["delay_spread"])
    doppler = np.random.uniform(*task["doppler"])
    sampling_freq = int(min(1 / delay, 2 * doppler))
    if task["channel"] == "TDL":
        tdl = TDL(
            model=task.get("model", "A"),
            delay_spread=delay,
            carrier_frequency=FREQ,
            num_tx_ant=NUM_ANTENNAS,
            num_rx_ant=1,
            min_speed=task["speed_range"][0],
            max_speed=task["speed_range"][1]
        )
        h, _ = tdl(batch_size=BATCH_SIZE, num_time_steps=NUM_SLOTS, sampling_frequency=sampling_freq)
        h = tf.reduce_mean(h, axis=[-1, -2])
    else:
        rb = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=NUM_ANTENNAS)
        h, _ = rb(batch_size=BATCH_SIZE, num_time_steps=1)
        h = tf.squeeze(h, axis=1)
    h = tf.reshape(h, [BATCH_SIZE, 1, NUM_ANTENNAS])
    return apply_beamspace(h)[:, 0, :]

def generate_batch(period):
    comp = DAILY_COMPOSITION[period]
    names = list(comp.keys())
    probs = list(comp.values())
    task_map = {t["name"]: t for t in TASKS}
    users = [task_map[np.random.choice(names, p=probs)] for _ in range(NUM_USERS)]
    h_users = [generate_user_channel(task) for task in users]
    h = tf.stack(h_users, axis=1)
    norm = tf.reduce_mean(tf.norm(h, axis=-1, keepdims=True))
    return h / tf.cast(norm + 1e-6, tf.complex64)

def compute_metrics(h, w=None, power=1.0, noise_power=1e-3):
    start = time.time()
    B, U, A = h.shape
    if w is None:
        real = tf.random.normal([B, U, A])
        imag = tf.random.normal([B, U, A])
        w = tf.complex(real, imag)
        norm = tf.sqrt(tf.reduce_sum(tf.abs(w)**2, axis=-1, keepdims=True) + 1e-10)
        w = w / tf.cast(norm, tf.complex64) * tf.complex(tf.sqrt(power * U), 0.0)
    h_adj = tf.transpose(h, [0, 2, 1])
    signal_matrix = tf.matmul(w, h_adj)
    desired = tf.linalg.diag_part(signal_matrix)
    desired_power = tf.reduce_mean(tf.abs(desired)**2)
    mask = 1.0 - tf.eye(U, dtype=tf.float32)
    mask = tf.tile(tf.expand_dims(mask, 0), [B, 1, 1])
    interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
    interference_power = tf.reduce_mean(interference)
    sinr = desired_power / (interference_power + noise_power)
    throughput = tf.reduce_mean(tf.math.log(1.0 + sinr) / tf.math.log(2.0))
    latency = (time.time() - start) * 1000  # ms
    return float(sinr), float(throughput), latency

def compute_mmse_weights(h, noise_power=1e-3):
    B, U, A = h.shape
    h_herm = tf.transpose(h, [0, 2, 1])
    H_H = tf.matmul(h_herm, h, adjoint_b=False)
    identity = tf.eye(A, batch_shape=[B], dtype=tf.complex64)
    inv_term = tf.linalg.inv(H_H + tf.cast(noise_power, tf.complex64) * identity)
    w_mmse = tf.matmul(inv_term, h_herm)
    w_mmse = tf.transpose(w_mmse, [0, 2, 1])  # [B, U, A]
    return w_mmse

def compute_mmse(h, noise_power=1e-3, power=1.0):
    start = time.time()
    B, U, A = h.shape
    h_herm = tf.transpose(h, [0, 2, 1])
    H_H = tf.matmul(h_herm, h, adjoint_b=False)
    identity = tf.eye(A, batch_shape=[B], dtype=tf.complex64)
    inv_term = tf.linalg.inv(H_H + tf.cast(noise_power, tf.complex64) * identity)
    w_mmse = tf.matmul(inv_term, h_herm)
    w_mmse = tf.transpose(w_mmse, [0, 2, 1])
    norm = tf.sqrt(tf.reduce_sum(tf.abs(w_mmse)**2, axis=-1, keepdims=True) + 1e-10)
    w = w_mmse / tf.cast(norm, tf.complex64) * tf.cast(tf.sqrt(power * U), tf.complex64)
    h_adj = tf.transpose(h, [0, 2, 1])
    signal_matrix = tf.matmul(w, h_adj)
    desired = tf.linalg.diag_part(signal_matrix)
    desired_power = tf.reduce_mean(tf.abs(desired)**2)
    mask = 1.0 - tf.eye(U, dtype=tf.float32)
    mask = tf.tile(tf.expand_dims(mask, 0), [B, 1, 1])
    interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
    interference_power = tf.reduce_mean(interference)
    sinr = desired_power / (interference_power + noise_power)
    throughput = tf.reduce_mean(tf.math.log(1.0 + sinr) / tf.math.log(2.0))
    latency = (time.time() - start) * 1000  # ms
    return float(sinr), float(throughput), latency

#@tf.function
def train_step(model, optimizer, x, h, channel_features, task_idx, epoch, loss_weights):
    alpha = tf.minimum(tf.cast(epoch, tf.float32) / 50.0, 1.0)
    alpha_c = tf.complex(alpha, 0.0)

    with tf.GradientTape() as tape:
        w_mmse = compute_mmse_weights(h)  # 🎯 Teacher
        w_pred = model(x, channel_features, task_idx, training=True)

        w_blend = alpha_c * w_pred + (1.0 - alpha_c) * w_mmse

        # --- SINR Loss ---
        h_adj = tf.transpose(h, [0, 2, 1])
        signal_matrix = tf.matmul(w_blend, h_adj)
        desired = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired)**2)
        mask = 1.0 - tf.eye(model.num_users, dtype=tf.float32)
        mask = tf.tile(mask[None, :, :], [tf.shape(x)[0], 1, 1])
        interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
        interference_power = tf.reduce_mean(interference)
        sinr = desired_power / (interference_power + 1e-3)
        main_loss = -tf.reduce_mean(tf.math.log(1.0 + sinr))

        # --- Alignment Loss ---
        h_norm = tf.nn.l2_normalize(h, axis=-1)
        w_norm = tf.nn.l2_normalize(w_blend, axis=-1)
        complex_dot = tf.reduce_sum(h_norm * tf.math.conj(w_norm), axis=-1)
        abs_dot = tf.abs(complex_dot)
        align_loss = tf.reduce_mean(1.0 - abs_dot)

        # --- Phase Loss ---
        angle_penalty = tf.reduce_mean(tf.abs(tf.math.angle(complex_dot)))

        # --- Orthogonality Loss ---
        w_orth = tf.matmul(w_blend, w_blend, adjoint_b=True)
        eye = tf.eye(model.num_users, dtype=tf.complex64)
        orth_loss = tf.math.log(1. + tf.reduce_mean(tf.abs(w_orth * (1.0 - eye))))

        # --- Replay Loss ---
        replay_loss = 0.0
        if model.use_replay:
            replay_x, replay_h = model.sample_replay(task_idx, 8)
            if replay_x is not None:
                dummy = tf.zeros_like(tf.math.real(replay_x[..., :3]))
                w_r = model(replay_x, dummy, task_idx, training=True)
                s_r = tf.matmul(w_r, tf.transpose(replay_h, [0, 2, 1]))
                desired_r = tf.linalg.diag_part(s_r)
                power_r = tf.reduce_mean(tf.abs(desired_r)**2)
                interf_r = tf.reduce_mean(tf.reduce_sum(tf.abs(s_r)**2, axis=-1)) - power_r
                sinr_r = power_r / (interf_r + 1e-3)
                replay_loss = -tf.reduce_mean(tf.math.log(1.0 + sinr_r))

        fisher_reg = model.regularization_loss()

        total_loss = (
            main_loss +
            loss_weights["align"] * align_loss +
            loss_weights["orth"] * orth_loss +
            loss_weights["phase"] * angle_penalty +
            loss_weights["replay"] * replay_loss +
            loss_weights["fisher"] * fisher_reg
        )

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # --- Logging to terminal every 20 epochs ---
    if epoch % 20 == 0:
        sinr_db = (10.0 * tf.math.log(sinr) / tf.math.log(10.0)).numpy()
        tf.print(f"➡️  SINR[dB]= {sinr_db:.2f} | Align={align_loss:.4f} | |dot|={tf.reduce_mean(abs_dot):.4f} | Interf={interference_power:.4f}")

    # --- Logging to file ---
    if epoch % 20 == 0:
        # Per-user log
        w0, h0 = w_pred[0], h[0]
        lines = []
        for u in range(model.num_users):
            w_u = tf.nn.l2_normalize(w0[u])
            h_u = tf.nn.l2_normalize(h0[u])
            c_dot = tf.reduce_sum(h_u * tf.math.conj(w_u))
            if tf.abs(tf.math.angle(c_dot)) > 1.0 or tf.abs(c_dot) < 0.3:
                lines.append(f"User {u:02d} | dot={tf.math.real(c_dot):.4f} | angle={tf.math.angle(c_dot):.4f} rad | |dot|={tf.abs(c_dot):.4f}")
        if lines:
            with open("per_user_diag.log", "a") as f:
                f.write(f"[Epoch {task_idx}:{epoch}] ----\n")
                for line in lines:
                    f.write(line + "\n")

        # KPI log
        sinr_val, thrpt_val, lat_val = model.eval_kpi(w_pred, h)
        with open("summary_kpi.log", "a") as f:
            f.write(f"[Epoch {task_idx}:{epoch}] SINR={sinr_val:.2f} dB | Thrpt={thrpt_val:.4f} bps/Hz | "
                    f"Latency={lat_val:.2f} ms | Loss={total_loss.numpy():.4f} | "
                    f"Align={align_loss.numpy():.4f} | Orth={orth_loss.numpy():.4f} | "
                    f"Replay={replay_loss:.4f} | Fisher={fisher_reg:.4f}\n")

    return {
        "total_loss": total_loss,
        "main_loss": main_loss,
        "align_loss": align_loss,
        "orth_loss": orth_loss,
        "replay_loss": replay_loss,
        "fisher_reg": fisher_reg,
        "sinr": 10.0 * tf.math.log(sinr) / tf.math.log(10.0)
    }


    
def training_loop(model, optimizer, tasks, num_epochs=5, batch_size=8):
    for task_idx, task in enumerate(tasks):
        print(f"\n🧠 Training Task: {task['name']} (idx={task_idx})")
        for epoch in range(num_epochs):
            x_batch, h_batch, channel_feats = model.generate_synthetic_batch(task, batch_size)
            metrics = train_step(model, optimizer, x_batch, h_batch, channel_feats, task_idx, epoch, {
                "align": 1.0,
                "orth": 0.5,
                "phase": 0.3,
                "replay": 1.0,
                "fisher": 1.0
            })
            model.update_replay(x_batch, h_batch, task_idx)

            print(f"[{task['name']}][Epoch {epoch}] 🎯 SINR: {metrics['sinr'].numpy():.2f} dB | Loss: {metrics['total_loss'].numpy():.4f}")

        if model.use_fisher:
            model.update_fisher_info(x_batch, h_batch, task_idx)


# Main
if __name__ == "__main__":
    results = {
        "hour": [], "period": [],
        "sinr_random": [], "thrpt_random": [], "latency_random": [],
        "sinr_mmse": [], "thrpt_mmse": [], "latency_mmse": []
    }
    hour = 0
    for period in ["morning", "noon", "evening"]:
        for _ in tqdm(range(PERIOD_HOURS[period]), desc=f"{period.upper()}"):
            h = generate_batch(period)
            sinr_r, thrpt_r, lat_r = compute_metrics(h)
            sinr_m, thrpt_m, lat_m = compute_mmse(h)

            results["hour"].append(hour)
            results["period"].append(period)
            results["sinr_random"].append(sinr_r)
            results["thrpt_random"].append(thrpt_r)
            results["latency_random"].append(lat_r)
            results["sinr_mmse"].append(sinr_m)
            results["thrpt_mmse"].append(thrpt_m)
            results["latency_mmse"].append(lat_m)

            hour += 1

    print("\n📊 Daily Summary (mean over 24h):")
    print(f"🎲 Random Beamforming → SINR: {np.mean(results['sinr_random']):.2f} dB | Throughput: {np.mean(results['thrpt_random']):.4f} bps/Hz | Latency: {np.mean(results['latency_random']):.2f} ms")
    print(f"🎯 MMSE Beamforming   → SINR: {10*np.log10(np.mean(results['sinr_mmse'])):.2f} dB | Throughput: {np.mean(results['thrpt_mmse']):.4f} bps/Hz | Latency: {np.mean(results['latency_mmse']):.2f} ms")
    print("\n🚀 Starting Continual Learning Training...")
    model = BeamformingMetaAttentionModel(
    num_antennas=NUM_ANTENNAS,
    num_users=NUM_USERS,
    num_tasks=len(TASKS),
    use_replay=False,
    use_fisher=False
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # 👇 Warmup GRUs to avoid tf.function variable creation error
    for task_idx in range(len(TASKS)):
        dummy_input = tf.zeros((1, NUM_USERS, model.hidden_dim), dtype=tf.float32)
        gru_layer = getattr(model, f"gru_task_{task_idx}")
        gate_layer = getattr(model, f"gate_task_{task_idx}")
        _ = gru_layer(dummy_input)
        _ = gate_layer(dummy_input)

    # ✅ Test Norm(w) and Norm(h) before training loop
    print("\n🧪 Testing Norm(w) and Norm(h) before training loop...\n")
    task = TASKS[0]  # e.g. Static
    x_batch, h_batch, feats = model.generate_synthetic_batch(task, batch_size=8)
    w = model(x_batch, feats, task_idx=0, training=True)

    # Print norms
    
    tf.print("📡 Norm(w) =", tf.reduce_mean(tf.norm(w, axis=-1)), output_stream=sys.stdout)
    tf.print("📡 Norm(h) =", tf.reduce_mean(tf.norm(h_batch, axis=-1)), output_stream=sys.stdout)

    kpis = training_loop(model, optimizer, TASKS, num_epochs=50, batch_size=8)
