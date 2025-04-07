
import tensorflow as tf
import numpy as np
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
from tqdm import tqdm
import time
from tensorflow.keras import layers
import os
import sys
from sionna.phy.mimo import cbf_precoding_matrix,rzf_precoding_matrix, normalize_precoding_power
tf.config.run_functions_eagerly(True)
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

class NormalizedDense(tf.keras.layers.Dense):
    def call(self, inputs):
        norm_weights = self.kernel / tf.norm(self.kernel, axis=0)
        return tf.matmul(inputs, norm_weights) + self.bias

class LearnableBeamspace(tf.keras.layers.Layer):
    def __init__(self, num_antennas):
        super().__init__()
        init_dft = tf.signal.fft(tf.eye(num_antennas, dtype=tf.complex64))
        self.dft = tf.Variable(init_dft, trainable=True)

    def call(self, h):
        res = tf.einsum('bua,ac->buc', h, self.dft)
        mask = tf.abs(res) > 0.05  # Prune small values
        return tf.where(mask, res, tf.zeros_like(res))


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
        self.out = NormalizedDense(num_antennas * num_users * 2)
        self.x_feat_proj_layer = tf.keras.layers.Dense(self.hidden_dim)

        # Learnable beamspace module (optional)
        self.beamspace = LearnableBeamspace(num_antennas)

        dummy_input = tf.random.normal([1, num_users, self.hidden_dim])
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

        # 🧠 Task-specific GRUs and Gating (average combination across all tasks)
        # 🧠 فقط GRU مربوط به task فعلی
        gru_layer = getattr(self, f"gru_task_{task_idx}")
        gate_layer = getattr(self, f"gate_task_{task_idx}")
        gru_out = gru_layer(x_attn)
        gate = gate_layer(x_attn)
        x_gated = tf.multiply(gru_out, gate)



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
        # ✅ Print mean norm to force execution
        norm_mean = tf.reduce_mean(tf.norm(w, axis=-1))
        tf.print("📡 Norm(w) =", norm_mean, output_stream=sys.stdout)
        _ = tf.identity(norm_mean)

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
        # گزینه پیشرفته‌تر (بعداً اضافه کن):
        # - برای هر sample یک SINR ذخیره کن.
        # - با توجه به پایین‌ترین SINRها انتخاب کن.

        # جدید: SINR-based sampling
        def _sinr(xi, hi):
            hi_adj = tf.transpose(hi, [0, 2, 1])  # [1, A, U]
            xi_complex = tf.complex(xi[..., :self.num_antennas], xi[..., self.num_antennas:])  # [1, U, A]
            signal = tf.matmul(xi_complex, hi_adj)  # [1, U, U]
            desired = tf.linalg.diag_part(signal)
            p_d = tf.reduce_mean(tf.abs(desired) ** 2)
            interference = tf.reduce_sum(tf.abs(signal) ** 2) - p_d
            return (p_d / (interference + 1e-3)).numpy()


        sinrs = np.array([_sinr(tf.expand_dims(xi,0), tf.expand_dims(hi,0)) for xi, hi in zip(self.replay_x[task_idx], self.replay_h[task_idx])])
        idx = np.argsort(sinrs)[:num_samples]  # select worst SINR examples


        # الان فقط یک استراتژی ساده‌تر:
        idx = np.random.choice(len(self.replay_x[task_idx]), size=num_samples, replace=False)

        x = tf.stack([self.replay_x[task_idx][i] for i in idx])  # shape: [B, U, 2*A]
        h = tf.stack([self.replay_h[task_idx][i] for i in idx])  # shape: [B, U, A]

        # ✅ Convert back to complex
        a = self.num_antennas
        real = x[..., :a]
        imag = x[..., a:]
        x_complex = tf.complex(real, imag)  # shape: [B, U, A]

        return x_complex, h

    def update_fisher_info(self, x, h, task_idx):
        if not self.use_fisher:
            return
        with tf.GradientTape() as tape:
            channel_dummy = tf.random.uniform([tf.shape(x)[0], tf.shape(x)[1], 3], dtype=tf.float32)
            w = self(x, channel_dummy, task_idx, training=True)
            loss = fisher_sinr_loss(w, h)  # ✅ حالا بدون مشکل

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
            h_b = self.beamspace(h)[:, 0, :]

            h_users.append(h_b)

            f = tf.tile([[doppler, delay, snr]], [batch_size, 1])  # [B, 3]
            features.append(f)

        h = tf.stack(h_users, axis=1)  # [B, U, A]
        h = h / tf.cast(tf.norm(h, axis=-1, keepdims=True) + 1e-8, tf.complex64)
        feats = tf.stack(features, axis=1)  # [B, U, 3]
        return tf.identity(h), tf.identity(h), feats  # ولی شاید بهتر باشه x یه چیز جدای از h باشه.

    
    def eval_kpi(self, w, h):
        """
        Evaluate SINR, throughput, and latency.
        - w: beamforming weights [B, U, A]
        - h: channel [B, U, A]
        Returns: (SINR_dB, throughput_bps_per_Hz, latency_ms)
        """
        # --- SINR ---
        h_adj = tf.transpose(h, [0, 2, 1])  # [B, A, U]
        signal_matrix = tf.matmul(w, h_adj)  # [B, U, U]
        desired = tf.linalg.diag_part(signal_matrix)  # [B, U]
        desired_power = tf.reduce_mean(tf.abs(desired)**2)

        # --- SINR ---
        h_adj = tf.transpose(h, [0, 2, 1])  # [B, A, U]
        signal_matrix = tf.matmul(w, h_adj)  # [B, U, U]
        desired = tf.linalg.diag_part(signal_matrix)  # [B, U]
        desired_power = tf.reduce_mean(tf.abs(desired)**2)

        # ✅ Fixed mask dtype
        mask = tf.ones_like(tf.abs(signal_matrix)) - tf.eye(self.num_users)[None, :, :]
        interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
        interference_power = tf.reduce_mean(interference)
        
        sinr_linear = desired_power / (interference_power + 1e-3)
        sinr_dB = tf.cast(10.0 * tf.math.log(sinr_linear) / tf.math.log(10.0), tf.float32)

        # --- Throughput ---
        thrpt = tf.math.log(1.0 + sinr_linear) / tf.math.log(2.0)
        thrpt_mean = tf.reduce_mean(thrpt)

        # --- Latency (fake model) ---
        latency_ms = tf.cast(1000.0 / (1.0 + thrpt_mean), tf.float32)  # just for eval

        return sinr_dB.numpy().item(), thrpt_mean.numpy().item(), latency_ms.numpy().item()

    def get_task_weights(self, task_idx):
        return [v.numpy() for v in self.trainable_variables]

    def compute_forgetting(self, prev_weights, task_idx):
        current_weights = self.get_task_weights(task_idx)
        diffs = [np.mean((cw - pw)**2) for cw, pw in zip(current_weights, prev_weights)]
        return np.mean(diffs)

def apply_beamspace(h):
    dft_matrix = tf.signal.fft(tf.eye(NUM_ANTENNAS, dtype=tf.complex64))  # [A, A]
    res = tf.einsum('bua,ac->buc', h, dft_matrix)  # Apply FFT across antennas
    mask = tf.abs(res) > 0.05  # Prune small values
    return tf.where(mask, res, tf.zeros_like(res))

def fisher_sinr_loss(w, h):
        """
        SINR-based loss برای Fisher اطلاعات.
        """
        B, U, A = h.shape
        h_adj = tf.transpose(h, [0, 2, 1])
        signal_matrix = tf.matmul(w, h_adj)
        desired = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired)**2)

        mask = tf.ones_like(tf.abs(signal_matrix)) - tf.eye(U)[None, :, :]
        interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
        interference_power = tf.reduce_mean(interference)

        sinr = desired_power / (interference_power + 1e-3)
        return -tf.reduce_mean(tf.math.log(1.0 + sinr))  # negative SINR (minimize)

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

class AlphaSelector(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, channel_feats):
        # channel_feats: [B, U, 3] → mean over users → [B, 3]
        x = tf.reduce_mean(channel_feats, axis=1)
        alpha = self.net(x)  # [B, 1]
        return tf.cast(alpha[:, None, None], tf.complex64)  # [B, 1, 1]

alpha_selector = AlphaSelector()

def teacher_blend(h, channel_feats, method="adaptive", alpha_rzf=0.05):
    """
    ترکیب RZF و CBF با توجه به شرایط کانال.
    - h: channel [B, U, A]
    - channel_feats: [B, U, 3] → [doppler, delay, snr]
    Returns: w_teacher [B, U, A]
    """
    w_cbf = tf.transpose(cbf_precoding_matrix(h), [0, 2, 1])  # [B, U, A]
    w_rzf = tf.transpose(rzf_precoding_matrix(h, alpha=alpha_rzf), [0, 2, 1])

    if method == "adaptive":
        alpha = alpha_selector(channel_feats)  # [B, 1, 1]
    else:
        alpha = tf.constant(0.5, dtype=tf.complex64)

    # ترکیب تطبیقی beamformers
    w_mix = alpha * w_rzf + (1.0 - alpha) * w_cbf  # [B, U, A]

    # Safe reshape → normalize → restore
    w_shape = tf.shape(w_mix)                  # [B, U, A]
    w_mix_flat = tf.reshape(w_mix, [-1, w_shape[-1]])  # [B*U, A]
    w_norm_flat = normalize_precoding_power(w_mix_flat)
    w_norm = tf.reshape(w_norm_flat, w_shape)  # [B, U, A]

    return w_norm


# ✅ Final Clean Version
def train_step(model, optimizer, x, h, channel_features, task_idx, task_name, epoch, loss_weights):
    
    alpha = tf.minimum(tf.cast(epoch, tf.float32) / 50.0, 1.0)

    with tf.GradientTape() as tape:
        w_teacher = teacher_blend(h, channel_features)

        w_pred = model(x, channel_features, task_idx, training=True)

        # ✅ α-blending
        alpha_c = tf.complex(alpha, 0.0)
        w_blend = alpha_c * w_pred + (1.0 - alpha_c) * w_teacher

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
        def complex_cosine(h, w):
            h_norm = h / tf.norm(h, axis=-1, keepdims=True)
            w_norm = w / tf.norm(w, axis=-1, keepdims=True)
            return tf.abs(tf.reduce_sum(h_norm * tf.math.conj(w_norm), axis=-1))

        cos_sim = complex_cosine(h, w_blend)
        complex_dot = tf.reduce_sum(tf.nn.l2_normalize(h, axis=-1) * tf.math.conj(tf.nn.l2_normalize(w_blend, axis=-1)), axis=-1)

        align_loss = tf.reduce_mean(1.0 - tf.square(cos_sim))



        # --- Phase Loss ---
        angle_penalty = tf.reduce_mean(tf.abs(tf.math.angle(complex_dot)))

        # --- Orthogonality ---
        W = w_blend  # [B, U, A]
        W_proj = tf.matmul(W, W, adjoint_b=True)  # [B, U, U]
        # برای پایداری عددی:
        W_proj += tf.cast(tf.eye(model.num_users, batch_shape=[tf.shape(W)[0]]), tf.complex64) * tf.complex(1e-3, 0.0)

        orth_loss = -tf.reduce_mean(tf.linalg.logdet(W_proj))


        # --- Replay & Fisher ---
        replay_loss = 0.0
        if model.use_replay:
            replay_x, replay_h = model.sample_replay(task_idx, 8)
            if replay_x is not None:
                dummy = tf.zeros_like(tf.math.real(replay_x[..., :3]))
                w_r = tf.transpose(cbf_precoding_matrix(replay_h), [0, 2, 1])
  # ✅ استفاده از cbf به‌جای پیش‌بینی مدل

                s_r = tf.matmul(w_r, tf.transpose(replay_h, [0, 2, 1]))
                desired_r = tf.linalg.diag_part(s_r)
                power_r = tf.reduce_mean(tf.abs(desired_r)**2)
                interf_r = tf.reduce_mean(tf.reduce_sum(tf.abs(s_r)**2, axis=-1)) - power_r
                sinr_r = power_r / (interf_r + 1e-3)
                replay_loss = -tf.reduce_mean(tf.math.log(1.0 + sinr_r))

        fisher_reg = model.regularization_loss()

        # Similarity-based adaptive lambda
        lambda_adaptive = tf.sigmoid(tf.reduce_mean(cos_sim))  # بین 0 و 1
        total_loss = (
            main_loss +
            align_loss * 1.0 +
            orth_loss * 0.5 +
            angle_penalty * 0.3 +
            lambda_adaptive * 3.0 * replay_loss +
            (1.0 - lambda_adaptive) * 1.0 * fisher_reg
        )

    # ✅ Gradient update
    grads = tape.gradient(total_loss, model.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    for g, v in grads_and_vars:
        tf.print("📉 Grad norm:", v.name, tf.norm(g), output_stream=sys.stdout)

    optimizer.apply_gradients(grads_and_vars)

    # ✅ Norms (fixing ? issue)
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # فقط اخطارهای مهم
    tf.config.run_functions_eagerly(True)  # 🔥 اجرای اجباری EAGER

    norm_w_scalar = tf.reduce_mean(tf.norm(w_pred, axis=-1))
    norm_h_scalar = tf.reduce_mean(tf.norm(h, axis=-1))

    # ✅ Convert to numpy only after ensuring computation
    norm_w_scalar_val = norm_w_scalar.numpy() if tf.executing_eagerly() else tf.keras.backend.eval(norm_w_scalar)
    norm_h_scalar_val = norm_h_scalar.numpy() if tf.executing_eagerly() else tf.keras.backend.eval(norm_h_scalar)

    print(f"📡 Norm(w) = {norm_w_scalar_val:.4f}")
    print(f"📡 Norm(h) = {norm_h_scalar_val:.4f}")

    # 🧠 Per-user dot diagnostics
    log_lines = []
    w0 = w_pred[0]
    h0 = h[0]

    # ➕ SINR per-user
    sinr_per_user = tf.math.log(1.0 + tf.abs(desired)**2 / (interference + 1e-3)) / tf.math.log(2.0)  # [B, U]

    for u in range(model.num_users):
        w_u = tf.nn.l2_normalize(w0[u])
        h_u = tf.nn.l2_normalize(h0[u])
        complex_dot_u = tf.reduce_sum(h_u * tf.math.conj(w_u))
        dot_real = tf.math.real(complex_dot_u)
        dot_abs = tf.abs(complex_dot_u)
        angle = tf.math.angle(complex_dot_u)

        sinr_val_raw = sinr_per_user[0, u].numpy()
        sinr_u_val = float(sinr_val_raw.reshape(-1)[0])


        if (dot_abs < 0.3) or (tf.abs(angle) > 0.8):
            line = f"User {u:02d} | dot={dot_real:.4f} | angle={angle:.4f} rad | |dot|={dot_abs:.4f} | SINR={sinr_u_val:.2f} bps/Hz"
            log_lines.append(line)


    if epoch % 20 == 0 and log_lines:
        with open("per_user_diag.log", "a") as f:
            f.write(f"[Epoch {task_idx}:{epoch}] ----\n")
            for line in log_lines:
                f.write(line + "\n")

    # 📈 KPI logging
    sinr_val, thrpt_val, lat_val = model.eval_kpi(w_pred, h)
    if epoch % 20 == 0:
        with open("summary_kpi.log", "a") as f:
            f.write(f"[Task {task_idx} - {task_name}][Epoch {epoch}] SINR={sinr_val:.2f} dB | Thrpt={thrpt_val:.4f} bps/Hz | "
                    f"Latency={lat_val:.2f} ms | Loss={total_loss.numpy():.4f} | "
                    f"Align={align_loss.numpy():.4f} | Orth={orth_loss.numpy():.4f} | "
                    f"Replay={replay_loss:.4f} | Fisher={fisher_reg:.4f}\n")

    # ✅ Return actual SINR dB
    sinr_dB = 10.0 * tf.math.log(sinr) / tf.math.log(10.0)
    return {
        "total_loss": total_loss,
        "main_loss": main_loss,
        "align_loss": align_loss,
        "orth_loss": orth_loss,
        "replay_loss": replay_loss,
        "fisher_reg": fisher_reg,
        "sinr": sinr_dB
    }
    
def training_loop(model, optimizer, tasks, num_epochs=5, batch_size=8):
    for task_idx, task in enumerate(tasks):
        print(f"\n🧠 Training Task: {task['name']} (idx={task_idx})")
        for epoch in range(num_epochs):
            x_batch, h_batch, channel_feats = model.generate_synthetic_batch(task, batch_size)
            metrics = train_step(model, optimizer, x_batch, h_batch, channel_feats, task_idx, task["name"], epoch, {
                "align": 1.0,
                "orth": 0.5,
                "phase": 0.3,
                "replay": 3.0,
                "fisher": 1.0
            })

            model.update_replay(x_batch, h_batch, task_idx)

            print(f"[{task['name']}][Epoch {epoch}] 🎯 SINR: {metrics['sinr'].numpy():.2f} dB | Loss: {metrics['total_loss'].numpy():.4f}")

        if model.use_fisher:
            print(f"📥 Updating Fisher info for task {task['name']}")
            x_fish, h_fish, feats = model.generate_synthetic_batch(task, batch_size=32)
            model.update_fisher_info(x_fish, h_fish, task_idx)


        # ✅ After task training, compute forgetting & transfer
        if model.use_fisher:
            print(f"📥 Updating Fisher info for task {task['name']}")
            x_fish, h_fish, feats = model.generate_synthetic_batch(task, batch_size=32)
            model.update_fisher_info(x_fish, h_fish, task_idx)

        # ✅ Compute Forgetting, FWT, BWT
        if task_idx > 0:
            for prev_task in range(task_idx):
                x_prev, h_prev, feats_prev = model.generate_synthetic_batch(tasks[prev_task], batch_size=16)
                w_prev = model(x_prev, feats_prev, task_idx=prev_task, training=False)
                sinr_prev = model.eval_kpi(w_prev, h_prev)[0]

                w_curr = model(x_prev, feats_prev, task_idx=task_idx, training=False)
                sinr_curr = model.eval_kpi(w_curr, h_prev)[0]

                forgetting = sinr_prev - sinr_curr
                fwt = sinr_curr  # forward transfer
                bwt = sinr_prev  # backward transfer

                with open("summary_kpi.log", "a") as f:
                    f.write(f"[TASK {task_idx}] {task['name']} | ON PREV TASK {prev_task} | "
                            f"Forget={forgetting:.2f}dB | FWT={fwt:.2f}dB | BWT={bwt:.2f}dB\n")

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
    print(f"🎲 Random Beamforming → SINR: {np.mean(results['sinr_random']):.2f} dB | "
          f"Throughput: {np.mean(results['thrpt_random']):.4f} bps/Hz | "
          f"Latency: {np.mean(results['latency_random']):.2f} ms")
    print(f"🎯 MMSE Beamforming   → SINR: {10*np.log10(np.mean(results['sinr_mmse'])):.2f} dB | "
          f"Throughput: {np.mean(results['thrpt_mmse']):.4f} bps/Hz | "
          f"Latency: {np.mean(results['latency_mmse']):.2f} ms")

    print("\n🚀 Starting Continual Learning Training...")

    model = BeamformingMetaAttentionModel(
        num_antennas=NUM_ANTENNAS,
        num_users=NUM_USERS,
        num_tasks=len(TASKS),
        use_replay=True,
        use_fisher=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # ⬇️ Initialize model once to build variables
    _ = model(tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64),
            tf.zeros((1, NUM_USERS, 3), dtype=tf.float32),
            task_idx=0, training=False)

    # ⬇️ Build optimizer with model variables
    optimizer.build(model.trainable_variables)


    # 🔥 Warmup forward passes برای پایدارسازی اولیه مدل
    for i in range(10):
        x, h, feats = model.generate_synthetic_batch(TASKS[0], batch_size=8)
        _ = model(x, feats, task_idx=0, training=True)


    # ✅ Warm-up GRUs
    for task_idx in range(len(TASKS)):
        dummy_input = tf.zeros((1, NUM_USERS, model.hidden_dim), dtype=tf.float32)
        _ = getattr(model, f"gru_task_{task_idx}")(dummy_input)
        _ = getattr(model, f"gate_task_{task_idx}")(dummy_input)

    # ✅ Build optimizer with all variables before training
    _ = model(tf.zeros((1, NUM_USERS, NUM_ANTENNAS), dtype=tf.complex64),
              tf.zeros((1, NUM_USERS, 3), dtype=tf.float32),
              task_idx=0, training=False)
    optimizer.build(model.trainable_variables)

    # ✅ Test Norms before training
    print("\n🧪 Testing Norm(w) and Norm(h) before training loop...\n")
    task = TASKS[0]
    x_batch, h_batch, feats = model.generate_synthetic_batch(task, batch_size=8)
    w = model(x_batch, feats, task_idx=0, training=True)

    norm_w = tf.reduce_mean(tf.norm(w, axis=-1))
    norm_h = tf.reduce_mean(tf.norm(h_batch, axis=-1))
    _ = tf.identity(norm_w)
    _ = tf.identity(norm_h)
    tf.print("📡 Norm(w) =", norm_w, output_stream=sys.stdout)
    tf.print("📡 Norm(h) =", norm_h, output_stream=sys.stdout)

    # 🔁 Start training loop
    kpis = training_loop(model, optimizer, TASKS, num_epochs=50, batch_size=8)