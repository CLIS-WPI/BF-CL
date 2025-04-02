
import tensorflow as tf
import numpy as np
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
from tqdm import tqdm
import time
from tensorflow.keras import layers
import os

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
        real = tf.math.real(x)
        imag = tf.math.imag(x)
        x_split = tf.concat([real, imag], axis=-1)
        x_split = tf.cast(x_split, tf.float32)

        # Conv + Channel Feature concat
        x_feat = self.conv1(x_split)
        x_feat = tf.concat([x_feat, channel_features], axis=-1)
        x_feat = self.concat_proj(x_feat)  # 🔁 تبدیل به [B, U, 128]


        # Shared attention
        x_attn = self.mha_shared(x_feat, x_feat)

        # Task-specific GRU & Gate (must use getattr to avoid variable creation mid-call)
        gru_layer = getattr(self, f"gru_task_{task_idx}")
        gate_layer = getattr(self, f"gate_task_{task_idx}")

        gru_out = gru_layer(x_attn)
        gate = gate_layer(x_attn)
        x_gated = tf.multiply(gate, gru_out)

        # Projection + Norm
        x_feat_proj = self.x_feat_proj_layer(x_feat)
        x_norm = self.norm(x_gated + x_feat_proj)

        # MLP → Beamforming Weights
        x_flat = tf.reduce_mean(x_norm, axis=1)
        x_out = self.fc2(self.fc1(x_flat))
        out = self.out(x_out)

        # Convert to complex weights
        real = out[:, :self.num_users * self.num_antennas]
        imag = out[:, self.num_users * self.num_antennas:]
        w = tf.complex(real, imag)
        w = tf.reshape(w, [batch_size, self.num_users, self.num_antennas])

        # Normalize
        norm_sq = tf.reduce_sum(tf.abs(w) ** 2, axis=-1, keepdims=True)
        norm = tf.sqrt(norm_sq + 1e-10)
        norm = tf.cast(norm, tf.complex64)

        scale = tf.cast(tf.sqrt(tf.cast(self.num_users, tf.float32)), tf.complex64)
        w = w / norm * scale

        return w



    def update_replay(self, x, h, task_idx):
        if not self.use_replay: return
        if len(self.replay_x[task_idx]) >= self.replay_limit:
            self.replay_x[task_idx] = self.replay_x[task_idx][-self.replay_limit//2:]
            self.replay_h[task_idx] = self.replay_h[task_idx][-self.replay_limit//2:]
        self.replay_x[task_idx].extend(tf.unstack(x))
        self.replay_h[task_idx].extend(tf.unstack(h))

    def sample_replay(self, task_idx, num_samples):
        if not self.use_replay or len(self.replay_x[task_idx]) == 0:
            return None, None
        idx = np.random.choice(len(self.replay_x[task_idx]), size=num_samples)
        x = tf.stack([self.replay_x[task_idx][i] for i in idx])
        h = tf.stack([self.replay_h[task_idx][i] for i in idx])
        return x, h

    def update_fisher_info(self, x, h, task_idx):
        if not self.use_fisher: return
        with tf.GradientTape() as tape:
            channel_dummy = tf.zeros_like(tf.math.real(x[..., :3]))  # cast to float32
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
        h = h / (tf.reduce_mean(tf.norm(h, axis=-1, keepdims=True)) + 1e-6)
        feats = tf.stack(features, axis=1)  # [B, U, 3]
        return h, h, tf.cast(feats, tf.float32)  # x, h, features


    def eval_kpi(self, w, h, noise_power=1e-3):
        B, U, A = h.shape
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
        return float(10*np.log10(sinr)), float(throughput)


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

@tf.function
def train_step(model, optimizer, x, h, channel_features, task_idx, loss_weights):
    with tf.GradientTape() as tape:
        w_pred = model(x, channel_features, task_idx, training=True)

        # --- MMSE fallback ---
        h_adj = tf.transpose(h, [0, 2, 1])  # [B, A, U]
        signal_matrix = tf.matmul(w_pred, h_adj)
        desired = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired)**2)

        mask = 1.0 - tf.eye(model.num_users, dtype=tf.float32)
        mask = tf.tile(mask[None, :, :], [tf.shape(x)[0], 1, 1])
        interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
        interference_power = tf.reduce_mean(interference)

        sinr = desired_power / (interference_power + 1e-3)
        main_loss = -tf.reduce_mean(tf.math.log(1.0 + sinr))  # maximize SINR

        # --- Alignment loss (cosine similarity) ---
        h_norm = tf.nn.l2_normalize(h, axis=-1)
        w_norm = tf.nn.l2_normalize(w_pred, axis=-1)
        dot = tf.reduce_sum(tf.math.real(h_norm * tf.math.conj(w_norm)), axis=-1)
        align_loss = 1.0 - tf.reduce_mean(dot)

        # --- Orthogonality loss (between beams) ---
        w_orth = tf.matmul(w_pred, w_pred, adjoint_b=True)  # [B, U, U]
        eye = tf.eye(model.num_users, dtype=tf.complex64)
        off_diag = tf.reduce_mean(tf.abs(w_orth * (1.0 - eye)))  # ✅ compute only off-diagonal terms
        orth_loss = off_diag  # ✅ assign correctly


        # --- Replay loss ---
                # --- Replay loss ---
                # --- Replay loss ---
        replay_loss = 0.0
        if model.use_replay:
            replay_x, replay_h = model.sample_replay(task_idx, 8)
            if replay_x is not None:
                channel_dummy = tf.zeros_like(tf.math.real(replay_x[..., :3]))  # ensures float32
                w_replay = model(replay_x, channel_dummy, task_idx, training=True)

                signal_matrix_r = tf.matmul(w_replay, tf.transpose(replay_h, [0, 2, 1]))
                desired_r = tf.linalg.diag_part(signal_matrix_r)

                desired_power_r = tf.reduce_mean(tf.abs(desired_r) ** 2)
                desired_power_r = tf.cast(desired_power_r, tf.complex64)

                eye = tf.eye(model.num_users, dtype=tf.complex64)
                mask = tf.constant(1.0, dtype=tf.complex64) - eye

                abs_sq = tf.cast(tf.abs(signal_matrix_r) ** 2, tf.complex64)
                interference_r = tf.reduce_sum(abs_sq * mask, axis=-1)
                interference_power_r = tf.reduce_mean(interference_r)
                interference_power_r = tf.cast(interference_power_r, tf.complex64)

                sinr_r = desired_power_r / (interference_power_r + tf.constant(1e-3, dtype=tf.complex64))
                replay_loss_complex = -tf.reduce_mean(tf.math.log(tf.cast(1.0, tf.complex64) + sinr_r))
                replay_loss = tf.math.real(replay_loss_complex)  # ✅ make sure it's float32


        # --- Fisher regularization ---
        fisher_reg = model.regularization_loss()

        # --- Total loss ---
        total_loss = (
            main_loss +
            loss_weights["align"] * align_loss +
            loss_weights["orth"] * orth_loss +
            loss_weights["replay"] * replay_loss +
            loss_weights["fisher"] * fisher_reg
        )

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    

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
    kpi_log = []
    prev_task_weights = None

    for task_idx, task in enumerate(tasks):
        print(f"\n🧠 Training Task: {task['name']} ({task_idx})")
        losses = {"sinr": [], "thrpt": [], "forget": []}
        model.reset_metrics()

        for epoch in range(num_epochs):
            # Generate synthetic batch for this task
            x_batch, h_batch, channel_feats = model.generate_synthetic_batch(task, batch_size)
            metrics = train_step(model, optimizer, x_batch, h_batch, channel_feats, task_idx, {
                "align": 1.0,
                "orth": 0.1,
                "replay": 1.0,
                "fisher": 0.5 if model.use_fisher else 0.0,
            })
            model.update_replay(x_batch, h_batch, task_idx)
            print(f"[{task['name']}][Epoch {epoch+1}] 🎯 SINR: {metrics['sinr'].numpy():.2f} dB | Loss: {metrics['total_loss'].numpy():.4f}")
            losses["sinr"].append(metrics["sinr"].numpy())

        # Save Fisher matrix for task if applicable
        if model.use_fisher:
            model.update_fisher_info(x_batch, h_batch, task_idx)


        # Save weights for forgetting calculation
        if prev_task_weights is not None:
            forget = model.compute_forgetting(prev_task_weights, task_idx)
            losses["forget"].append(forget)
            print(f"🧠 Forgetting vs last task: {forget:.4f}")
        prev_task_weights = model.get_task_weights(task_idx)

        # Inference on Mixed users
        mixed_task = TASKS[task_idx]  # or random.choice(TASKS)
        x_mix, h_mix, mix_feats = model.generate_synthetic_batch(mixed_task, batch_size=16)

        w_pred = model(x_mix, mix_feats, task_idx, training=False)
        sinr, thrpt = model.eval_kpi(w_pred, h_mix)
        print(f"📶 Mixed Eval → SINR: {sinr:.2f} dB | Throughput: {thrpt:.4f} bps/Hz")
        kpi_log.append({
            "task": task["name"],
            "avg_sinr": np.mean(losses["sinr"]),
            "forgetting": np.mean(losses["forget"]) if losses["forget"] else 0.0,
            "mixed_sinr": sinr,
            "throughput": thrpt,
        })

    return kpi_log

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
    model = BeamformingMetaAttentionModel(num_antennas=NUM_ANTENNAS, num_users=NUM_USERS, num_tasks=len(TASKS))
    optimizer = tf.keras.optimizers.Adam(1e-3)

    # 👇 Warmup GRUs to avoid tf.function variable creation error
    for task_idx in range(len(TASKS)):
        dummy_input = tf.zeros((1, NUM_USERS, model.hidden_dim), dtype=tf.float32)
        gru_layer = getattr(model, f"gru_task_{task_idx}")
        gate_layer = getattr(model, f"gate_task_{task_idx}")
        _ = gru_layer(dummy_input)
        _ = gate_layer(dummy_input)

    kpis = training_loop(model, optimizer, TASKS, num_epochs=3, batch_size=8)
