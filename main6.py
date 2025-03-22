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
        # Initialize buffer with fixed size
        self.memory_buffer = {
            "x": tf.Variable(tf.zeros([REPLAY_BUFFER_SIZE, num_users, num_antennas], dtype=tf.complex64), trainable=False),
            "h": tf.Variable(tf.zeros([REPLAY_BUFFER_SIZE, num_users, num_antennas], dtype=tf.complex64), trainable=False)
        }
        self.buffer_size = tf.constant(REPLAY_BUFFER_SIZE, dtype=tf.int32)
        self.buffer_count = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.old_params = {}

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        expected_shape = [None, self.num_users, self.num_antennas]
        tf.debugging.assert_shapes([(inputs, expected_shape)])
        inputs_flat = tf.reshape(inputs, [batch_size, self.num_users * self.num_antennas])
        inputs_flat = tf.expand_dims(inputs_flat, axis=1)
        # Split real and imaginary parts but don’t cast yet
        real_part = tf.math.real(inputs_flat)  # Keep as float32 implicitly
        imag_part = tf.math.imag(inputs_flat)
        x = tf.concat([real_part, imag_part], axis=-1)  # Shape: [batch, 1, num_users * num_antennas * 2]
        x = self.gru(x)
        attn_output = self.attention(x, x)
        x = self.norm(attn_output + x)
        x = tf.reduce_mean(x, axis=1)  # Shape: [batch, 128]
        x = self.dense1(x)  # Shape: [batch, 256]
        x = self.dense2(x)  # Shape: [batch, 128]
        output = self.output_layer(x)  # Shape: [batch, num_antennas * num_users * 2]
        real = output[:, :self.num_antennas * self.num_users]
        imag = output[:, self.num_antennas * self.num_users:]
        w = tf.complex(real, imag)
        w = tf.reshape(w, [batch_size, self.num_users, self.num_antennas])
        norm_squared = tf.reduce_sum(tf.abs(w)**2, axis=-1, keepdims=True)
        norm = tf.sqrt(norm_squared + 1e-10)  # Keep as float32 for stability
        w = w / tf.cast(norm, tf.complex64) * tf.complex(tf.sqrt(POWER), 0.0)
        return w

    @tf.function(reduce_retracing=True)
    def generate_replay(self, num_samples):
        buffer_len = self.buffer_count
        tf.print("Buffer length in generate_replay:", buffer_len)
        
        def empty_case():
            x_dummy = tf.zeros([num_samples, self.num_users, self.num_antennas], dtype=tf.complex64)
            h_dummy = tf.zeros([num_samples, self.num_users, self.num_antennas], dtype=tf.complex64)
            return x_dummy, h_dummy

        def populated_case():
            available_samples = tf.minimum(buffer_len, self.buffer_size)
            idx = tf.cond(
                available_samples < num_samples,
                lambda: tf.random.uniform([num_samples], 0, available_samples, dtype=tf.int32),
                lambda: tf.random.shuffle(tf.range(available_samples, dtype=tf.int32))[:num_samples]
            )
            x_buffer = self.memory_buffer["x"][:available_samples]
            x_samples = tf.gather(x_buffer, idx)
            tf.print("x_samples shape in generate_replay:", tf.shape(x_samples))
            x_flat = tf.reshape(x_samples, [num_samples, self.num_antennas * self.num_users])
            noise = tf.random.normal([num_samples, self.num_antennas * self.num_users], stddev=0.1, dtype=tf.float32)
            real_part = tf.cast(tf.math.real(x_flat), tf.float32)
            imag_part = tf.cast(tf.math.imag(x_flat), tf.float32)
            gen_input = tf.concat([real_part, imag_part, noise], axis=-1)
            gen_output = self.generator(gen_input)
            real = gen_output[:, :self.num_antennas * self.num_users]
            imag = gen_output[:, self.num_antennas * self.num_users:]
            h_gen = tf.complex(real, imag)
            h_gen = tf.reshape(h_gen, [num_samples, self.num_users, self.num_antennas])
            return x_samples, h_gen

        x_replay, h_replay = tf.cond(
            buffer_len == 0,
            empty_case,
            populated_case
        )
        return x_replay, h_replay

    @tf.function(reduce_retracing=True)
    def train_generator(self, x_batch, h_batch, optimizer):
        batch_size = tf.shape(x_batch)[0]
        num_samples = batch_size // 2
        with tf.GradientTape() as tape:
            tape.watch(self.generator.trainable_variables)
            x_replay, h_replay = self.generate_replay(num_samples)
            is_dummy = tf.reduce_all(tf.equal(x_replay, 0.0))
            gen_loss = tf.cond(
                is_dummy,
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: self._compute_gen_loss(x_batch[:num_samples], h_replay)
            )
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        if grads is None or all(g is None for g in grads):
            tf.print("Warning: No gradients computed for generator!")
        else:
            optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return gen_loss

    @tf.function(reduce_retracing=True)
    def _compute_gen_loss(self, x_batch, h_replay):
        w = self(x_batch)
        h_adjusted = tf.transpose(h_replay, [0, 2, 1])
        signal_matrix = tf.matmul(w, h_adjusted)
        desired_signal = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired_signal)**2)
        eye = tf.eye(self.num_users, dtype=tf.float32)
        mask = 1.0 - eye
        num_samples = tf.shape(x_batch)[0]
        mask = tf.tile(mask[tf.newaxis, :, :], [num_samples, 1, 1])
        interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
        interference_power = tf.reduce_mean(interference)
        sinr = desired_power / (interference_power + NOISE_POWER)
        return -tf.math.log(1.0 + sinr)

    def update_memory(self, x, h):
        x = tf.ensure_shape(x, [None, self.num_users, self.num_antennas])
        h = tf.ensure_shape(h, [None, self.num_users, self.num_antennas])
        batch_size = tf.shape(x)[0]
        idx = tf.random.uniform([], 0, batch_size, dtype=tf.int32)
        x_sample = x[idx]
        h_sample = h[idx]
        current_count = self.buffer_count.read_value()
        # Use tf.cond to handle buffer update within graph
        def append_to_buffer():
            new_idx = current_count
            self.memory_buffer["x"].scatter_nd_update([[new_idx]], tf.expand_dims(x_sample, 0))
            self.memory_buffer["h"].scatter_nd_update([[new_idx]], tf.expand_dims(h_sample, 0))
            self.buffer_count.assign_add(1)
        
        def replace_in_buffer():
            replace_idx = tf.random.uniform([], 0, self.buffer_size, dtype=tf.int32)
            self.memory_buffer["x"].scatter_nd_update([[replace_idx]], tf.expand_dims(x_sample, 0))
            self.memory_buffer["h"].scatter_nd_update([[replace_idx]], tf.expand_dims(h_sample, 0))
        
        tf.cond(
            current_count < self.buffer_size,
            append_to_buffer,
            replace_in_buffer
        )

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
                h_t = tf.squeeze(h_t, axis=[2, 3, 5])
                print(f"After squeeze: {h_t.shape}")
                h_t = tf.reduce_mean(h_t, axis=0)
                print(f"After reduce batch: {h_t.shape}")
                h_t = tf.transpose(h_t, [0, 2, 1])
                print(f"After transpose: {h_t.shape}")
                h_t = tf.reduce_mean(h_t, axis=1)
                print(f"After reduce users: {h_t.shape}")
            h_chunk.append(h_t)
        h_chunks.append(tf.stack(h_chunk))
    h = tf.concat(h_chunks, axis=0)
    print(f"Final channel shape: {h.shape}")
    return h

# تابع Loss
@tf.function(reduce_retracing=True)
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
@tf.function(reduce_retracing=True)
def train_step(model, x_batch, h_batch, optimizer, gen_optimizer):
    batch_size = tf.shape(x_batch)[0]
    with tf.GradientTape() as tape:
        w = model(x_batch)
        h_adjusted = tf.transpose(h_batch, [0, 2, 1])
        signal_matrix = tf.matmul(w, h_adjusted)
        desired_signal = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired_signal)**2)
        eye = tf.eye(model.num_users, dtype=tf.float32)
        mask = 1.0 - eye
        mask = tf.tile(mask[tf.newaxis, :, :], [batch_size, 1, 1])
        interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
        interference_power = tf.reduce_mean(interference)
        sinr = desired_power / (interference_power + NOISE_POWER)
        tf.print("Desired Power:", desired_power, "Interference Power:", interference_power, "SINR:", sinr)
        loss = -tf.math.log(1.0 + sinr)
        for _ in range(INNER_STEPS):
            x_replay, h_replay = model.generate_replay(batch_size // 2)
            if not tf.reduce_all(tf.equal(x_replay, 0.0)):
                w_replay = model(x_replay)
                h_replay_adjusted = tf.transpose(h_replay, [0, 2, 1])
                signal_matrix_replay = tf.matmul(w_replay, h_replay_adjusted)
                desired_signal_replay = tf.linalg.diag_part(signal_matrix_replay)
                desired_power_replay = tf.reduce_mean(tf.abs(desired_signal_replay)**2)
                interference_replay = tf.reduce_sum(tf.abs(signal_matrix_replay)**2 * mask[:batch_size // 2], axis=-1)
                interference_power_replay = tf.reduce_mean(interference_replay)
                sinr_replay = desired_power_replay / (interference_power_replay + NOISE_POWER)
                loss += -tf.math.log(1.0 + sinr_replay)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    gen_loss = model.train_generator(x_batch, h_batch, gen_optimizer)
    return loss, gen_loss

def main(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = BeamformingMetaAttentionModel(NUM_ANTENNAS, NUM_USERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Build optimizers before entering tf.function
    # Generate real and imaginary parts separately for complex64
    real_part = tf.random.normal([BATCH_SIZE, NUM_USERS, NUM_ANTENNAS], dtype=tf.float32)
    imag_part = tf.random.normal([BATCH_SIZE, NUM_USERS, NUM_ANTENNAS], dtype=tf.float32)
    dummy_x = tf.complex(real_part, imag_part)  # Combine into complex64 tensor
    _ = model(dummy_x)  # First call to build model variables
    optimizer.build(model.trainable_variables)
    gen_optimizer.build(model.generator.trainable_variables)

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
                loss, gen_loss = train_step(model, x_batch, h_batch, optimizer, gen_optimizer)
                model.update_memory(x_batch, h_batch)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}, Gen Loss: {gen_loss:.4f}")

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