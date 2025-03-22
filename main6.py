import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import time
import gc
import os

# تنظیمات GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {physical_devices}")

# تنظیمات اولیه
NUM_ANTENNAS = 32
NUM_USERS = 5
FREQ = 28e9
POWER = 1.0
NUM_SLOTS = 200
BATCH_SIZE = 32
LAMBDA_REG = 5.0
NUM_EPOCHS = 4
CHUNK_SIZE = 50
GPU_POWER_DRAW = 400
NOISE_POWER = 0.1
REPLAY_BUFFER_SIZE = 200
NUM_RUNS = 3
INNER_STEPS = 3

# تعریف تسک‌ها
TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": 30e-9, "channel": "TDL", "model": "A"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": 50e-9, "channel": "Rayleigh"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": 100e-9, "channel": "TDL", "model": "C"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": 70e-9, "channel": "TDL", "model": "A"},
]

# مدل
class BeamformingMetaAttentionModel(Model):
    def __init__(self, num_antennas, num_users):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.gru = layers.GRU(128, return_sequences=True, dtype=tf.float32)
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=64, dtype=tf.float32)
        self.norm = layers.LayerNormalization(dtype=tf.float32)
        self.dense1 = layers.Dense(256, activation="relu", dtype=tf.float32)
        self.dense2 = layers.Dense(128, activation="relu", dtype=tf.float32)
        self.output_layer = layers.Dense(num_antennas * num_users * 2, dtype=tf.float32)
        self.generator = tf.keras.Sequential([
            layers.Dense(128, activation="relu", dtype=tf.float32),
            layers.Dense(num_antennas * num_users * 2, activation="tanh", dtype=tf.float32)
        ])
        self.generator.build((None, num_antennas * num_users * 3))
        self.memory_buffer = {"x": [], "h": []}
        self.old_params = {}

    @tf.function
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        expected_shape = [None, self.num_users, self.num_antennas]
        tf.debugging.assert_shapes([(inputs, expected_shape)])
        inputs_flat = tf.reshape(inputs, [batch_size, self.num_users * self.num_antennas])
        inputs_flat = tf.expand_dims(inputs_flat, axis=1)
        x = tf.concat([tf.math.real(inputs_flat), tf.math.imag(inputs_flat)], axis=-1)
        x = self.gru(x)
        attn_output = self.attention(x, x)
        x = self.norm(attn_output + x)
        x = tf.reduce_mean(x, axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        real = output[:, :self.num_antennas * self.num_users]
        imag = output[:, self.num_antennas * self.num_users:]
        w = tf.complex(real, imag)
        w = tf.reshape(w, [batch_size, self.num_users, self.num_antennas])
        norm_squared = tf.reduce_sum(tf.abs(w)**2, axis=-1, keepdims=True)
        norm = tf.cast(tf.sqrt(norm_squared + 1e-10), dtype=tf.complex64)
        w = w / norm * tf.complex(tf.sqrt(POWER), 0.0)
        return w

    def generate_replay(self, num_samples):
        if len(self.memory_buffer["x"]) == 0:
            return None, None
        available_samples = len(self.memory_buffer["x"])
        if available_samples < num_samples:
            idx = np.random.choice(available_samples, num_samples, replace=True)
        else:
            idx = np.random.choice(available_samples, num_samples, replace=False)
        idx = tf.convert_to_tensor(idx, dtype=tf.int32)
        x_samples = tf.gather(tf.convert_to_tensor(self.memory_buffer["x"], dtype=tf.complex64), idx)
        print(f"x_samples shape in generate_replay: {x_samples.shape}")
        x_flat = tf.reshape(x_samples, [num_samples, self.num_antennas * self.num_users])  # [16, 160]
        noise = tf.random.normal([num_samples, self.num_antennas * self.num_users], stddev=0.1, dtype=tf.float32)  # [16, 160]
        gen_input = tf.concat([tf.math.real(x_flat), tf.math.imag(x_flat), noise], axis=-1)  # [16, 480]
        gen_output = self.generator(gen_input)
        real = gen_output[:, :self.num_antennas * self.num_users]
        imag = gen_output[:, self.num_antennas * self.num_users:]
        h_gen = tf.complex(real, imag)
        h_gen = tf.reshape(h_gen, [num_samples, self.num_users, self.num_antennas])  # [16, 5, 32]
        return x_samples, h_gen

    def update_memory(self, x, h):
        x = tf.ensure_shape(x, [None, self.num_users, self.num_antennas])
        h = tf.ensure_shape(h, [None, self.num_users, self.num_antennas])
        batch_size = tf.shape(x)[0]
        idx = tf.random.uniform([], 0, batch_size, dtype=tf.int32)
        x_sample = x[idx]
        h_sample = h[idx]
        if len(self.memory_buffer["x"]) < REPLAY_BUFFER_SIZE:
            self.memory_buffer["x"].append(x_sample.numpy())
            self.memory_buffer["h"].append(h_sample.numpy())
        else:
            idx = np.random.randint(0, REPLAY_BUFFER_SIZE)
            self.memory_buffer["x"][idx] = x_sample.numpy()
            self.memory_buffer["h"][idx] = h_sample.numpy()

    def regularization_loss(self):
        penalty = 0.0
        for w in self.trainable_weights:
            if w.name in self.old_params:
                delta = w - self.old_params[w.name]
                penalty += tf.reduce_sum(tf.cast(delta, tf.float32)**2)
        return LAMBDA_REG * penalty

# تولید کانال
def generate_channel(task, num_slots):
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], NUM_USERS)
    avg_speed = np.mean(speeds) * 1000 / 3600  # m/s
    doppler_freq = avg_speed * FREQ / 3e8
    print(f"Task {task['name']}: Doppler Freq = {doppler_freq:.2f} Hz")

    if task["channel"] == "TDL":
        channel_model = sn.channel.tr38901.TDL(
            model=task["model"], delay_spread=task["delay_spread"], carrier_frequency=FREQ,
            num_tx_ant=NUM_ANTENNAS, num_rx_ant=1, dtype=tf.complex64
        )
    else:
        channel_model = sn.channel.RayleighBlockFading(
            num_rx=NUM_USERS, num_rx_ant=1, num_tx=1, num_tx_ant=NUM_ANTENNAS,
            dtype=tf.complex64
        )

    h_chunks = []
    for chunk_start in range(0, num_slots, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, num_slots)
        h_chunk = []
        for _ in range(chunk_start, chunk_end):
            if task["channel"] == "TDL":
                h_t = channel_model(BATCH_SIZE, NUM_USERS, sampling_frequency=max(500, int(2 * doppler_freq)))[0]
                print(f"TDL raw shape: {h_t.shape}")
                h_t = tf.reduce_sum(h_t, axis=-2)
                print(f"After reduce_sum paths: {h_t.shape}")
                h_t = tf.squeeze(h_t, axis=[1, 2, 3])
                print(f"After squeeze: {h_t.shape}")
                h_t = tf.reduce_mean(h_t, axis=0)
                print(f"After reduce batch: {h_t.shape}")
                h_t = tf.transpose(h_t, [1, 0])
                print(f"After transpose: {h_t.shape}")
            else:
                h_t = channel_model(BATCH_SIZE, NUM_USERS)[0]
                print(f"Rayleigh raw shape: {h_t.shape}")
                h_t = tf.squeeze(h_t, axis=[2, 3, 5])  # فقط ابعاد 1 رو squeeze می‌کنیم
                print(f"After squeeze: {h_t.shape}")
                h_t = tf.reduce_mean(h_t, axis=0)
                print(f"After reduce batch: {h_t.shape}")
                h_t = tf.transpose(h_t, [0, 2, 1])  # [5, 32, 5] -> [5, 5, 32]
                print(f"After transpose: {h_t.shape}")
                h_t = tf.reduce_mean(h_t, axis=1)  # میانگین روی بعد کاربرهای اضافی
                print(f"After reduce users: {h_t.shape}")
            h_chunk.append(h_t)
        h_chunks.append(tf.stack(h_chunk))
    h = tf.concat(h_chunks, axis=0)
    print(f"Final channel shape: {h.shape}")
    return h

# تابع Loss
def meta_loss(model, x, h):
    w = model(x)
    h_adjusted = tf.transpose(h, [0, 2, 1])
    signal_matrix = tf.matmul(w, h_adjusted)
    desired_signal = tf.linalg.diag_part(signal_matrix)
    desired_power = tf.reduce_mean(tf.abs(desired_signal)**2)
    mask = 1.0 - tf.eye(NUM_USERS, batch_shape=[tf.shape(signal_matrix)[0]])
    interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
    interference_power = tf.reduce_mean(interference)
    sinr = desired_power / (interference_power + NOISE_POWER)
    loss = -tf.math.log(1.0 + sinr) + model.regularization_loss()
    return loss

# تابع آموزش
def train_step(model, x_batch, h_batch, optimizer):
    initial_params = [tf.identity(w) for w in model.trainable_weights]
    for _ in range(INNER_STEPS):
        with tf.GradientTape() as tape:
            loss = meta_loss(model, x_batch, h_batch)
            x_replay, h_replay = model.generate_replay(BATCH_SIZE // 2)
            if x_replay is not None:
                replay_loss = meta_loss(model, x_replay, h_replay)
                loss += 0.5 * replay_loss
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    for w, init_w in zip(model.trainable_weights, initial_params):
        w.assign(init_w + 0.1 * (w - init_w))
    return loss

# حلقه اصلی
def main(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = BeamformingMetaAttentionModel(NUM_ANTENNAS, NUM_USERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting", "fwt", "bwt", "spectral_efficiency", "interference"]}
    task_performance = []
    task_initial_performance = []
    task_channels = {}

    total_start_time = time.time()
    
    for task_idx, task in enumerate(TASKS):
        print(f"Task {task['name']} ({task_idx+1}/{len(TASKS)})")
        h = generate_channel(task, NUM_SLOTS)
        task_channels[task_idx] = h
        x = h
        dataset = tf.data.Dataset.from_tensor_slices((x, h)).batch(BATCH_SIZE)

        _ = model(x[:1])
        initial_w = model(x)
        print(f"initial_w shape: {initial_w.shape}")
        h_transposed = tf.transpose(h, [0, 2, 1])
        print(f"h_transposed shape: {h_transposed.shape}")
        signal = tf.matmul(initial_w, h_transposed)
        print(f"signal shape: {signal.shape}")
        initial_signal = tf.reduce_mean(tf.abs(signal)**2).numpy()
        task_initial_performance.append(initial_signal)

        task_start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            for x_batch, h_batch in dataset:
                loss = train_step(model, x_batch, h_batch, optimizer)
                model.update_memory(x_batch, h_batch)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}")

        task_time = time.time() - task_start_time
        latency = task_time / (NUM_EPOCHS * NUM_SLOTS) * 1000
        w = model(x)
        h_adjusted = tf.transpose(h, [0, 2, 1])
        signal_matrix = tf.matmul(w, h_adjusted)
        desired_signal = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired_signal)**2).numpy()
        mask = 1.0 - tf.eye(NUM_USERS, batch_shape=[tf.shape(signal_matrix)[0]])
        interference = tf.reduce_mean(tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)).numpy()
        sinr = desired_power / (interference + NOISE_POWER)
        spectral_efficiency = np.log2(1 + sinr)
        throughput = desired_power
        energy = GPU_POWER_DRAW * task_time
        task_performance.append(throughput)

        forgetting = 0.0
        if task_idx > 0:
            past_deltas = []
            for past_idx in range(task_idx):
                past_h = task_channels[past_idx]
                past_w = model(past_h)
                past_signal = tf.reduce_mean(tf.abs(tf.matmul(past_w, tf.transpose(past_h, [0, 2, 1])))**2).numpy()
                past_deltas.append(task_initial_performance[past_idx] - past_signal)
            forgetting = np.mean(past_deltas) if past_deltas else 0.0

        fwt = 0.0
        if task_idx > 0:
            past_h = task_channels[task_idx - 1]
            past_w = model(past_h)
            past_signal = tf.reduce_mean(tf.abs(tf.matmul(past_w, tf.transpose(past_h, [0, 2, 1])))**2).numpy()
            fwt = (initial_signal - past_signal) / (past_signal + 1e-10)

        model.old_params = {w.name: w.numpy() for w in model.trainable_weights}
        results["throughput"].append(throughput)
        results["latency"].append(latency)
        results["energy"].append(energy)
        results["forgetting"].append(forgetting)
        results["fwt"].append(fwt)
        results["spectral_efficiency"].append(spectral_efficiency)
        results["interference"].append(interference)

        gc.collect()

    bwt = 0.0
    if len(task_performance) > 1:
        final_deltas = []
        for past_idx in range(len(TASKS) - 1):
            past_h = task_channels[past_idx]
            past_w = model(past_h)
            past_signal = tf.reduce_mean(tf.abs(tf.matmul(past_w, tf.transpose(past_h, [0, 2, 1])))**2).numpy()
            final_deltas.append(task_initial_performance[past_idx] - past_signal)
        bwt = np.mean(final_deltas) if final_deltas else 0.0
    results["bwt"] = [bwt] * len(TASKS)

    total_time = time.time() - total_start_time
    with open(f"results_seed_{seed}.txt", "a") as f:
        for task_idx, task in enumerate(TASKS):
            f.write(f"Task {task['name']}:\n")
            for metric in results:
                if metric != "bwt":
                    f.write(f"  {metric}: {results[metric][task_idx]:.4f}\n")
            f.write("-"*50 + "\n")
        f.write(f"BWT: {bwt:.4f}\n")
        f.write(f"Total Time: {total_time:.2f}s\n")

    return results

if __name__ == "__main__":
    all_results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting", "fwt", "bwt", "spectral_efficiency", "interference"]}
    
    for run in range(NUM_RUNS):
        print(f"Run {run+1}/{NUM_RUNS}")
        results = main(run)
        for metric in all_results:
            all_results[metric].append(results[metric])

    with open("results_summary.txt", "w") as f:
        for metric in all_results:
            values = np.array(all_results[metric])
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            f.write(f"{metric.capitalize()} (Mean ± Std):\n")
            for i, task in enumerate(TASKS):
                f.write(f"  {task['name']}: {mean[i]:.4f} ± {std[i]:.4f}\n")
            f.write("\n")

    plt.figure(figsize=(12, 8))
    for metric in all_results:
        values = np.array(all_results[metric])
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        plt.errorbar(range(len(TASKS)), mean, yerr=std, label=metric.capitalize(), marker="o")
    plt.legend()
    plt.xticks(range(len(TASKS)), [t["name"] for t in TASKS], rotation=45)
    plt.title("Performance Metrics Across Tasks")
    plt.savefig("performance_metrics.png")
    plt.close()