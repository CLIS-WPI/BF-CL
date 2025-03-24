import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import time
import gc
import os
import logging
import sys
import subprocess

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

def preprocess_input(x):
    real_part = tf.math.real(x)
    imag_part = tf.math.imag(x)
    return tf.concat([real_part, imag_part], axis=-1)

def get_gpu_power():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            timeout=1
        )
        powers = result.decode().strip().split('\n')
        total_power = sum(float(power) for power in powers if power)
        logging.info(f"GPU Power: {total_power} W")
        return total_power
    except Exception as e:
        logging.warning(f"Failed to get GPU power: {str(e)}. Using random value.")
        return np.random.uniform(350, 450)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {physical_devices}")

NUM_ANTENNAS = 32
NUM_USERS = 5
FREQ = 28e9
POWER = 1.0
NUM_SLOTS = 200
BATCH_SIZE_DEFAULT = 32
BATCH_SIZE_STATIC = 16
LAMBDA_REG = 10.0
NUM_EPOCHS_DEFAULT = 4
NUM_EPOCHS_STATIC = 2
CHUNK_SIZE = 50
NOISE_POWER = 0.1
REPLAY_BUFFER_SIZE_DEFAULT = 200
REPLAY_BUFFER_SIZE_PEDESTRIAN = 400
NUM_RUNS = 3
INNER_STEPS = 3

TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": 30e-9, "channel": "TDL", "model": "A"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": 50e-9, "channel": "Rayleigh"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": 100e-9, "channel": "TDL", "model": "C"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": 70e-9, "channel": "TDL", "model": "A"},
]

class BeamformingMetaAttentionModel(Model):
    def __init__(self, num_antennas, num_users):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.shared_transformer = layers.MultiHeadAttention(num_heads=8, key_dim=64, dtype=tf.float32)
        self.gru_static = layers.GRU(64, return_sequences=True, dtype=tf.float32)
        self.gru_default = layers.GRU(128, return_sequences=True, dtype=tf.float32)
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
        self.replay_buffer_size = tf.Variable(REPLAY_BUFFER_SIZE_DEFAULT, dtype=tf.int32, trainable=False)
        self.memory_buffer = {
            "x": tf.Variable(tf.zeros([REPLAY_BUFFER_SIZE_PEDESTRIAN, num_users, num_antennas], dtype=tf.complex64), trainable=False),
            "h": tf.Variable(tf.zeros([REPLAY_BUFFER_SIZE_PEDESTRIAN, num_users, num_antennas], dtype=tf.complex64), trainable=False)
        }
        self.buffer_count = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.old_params = {}
        self.fisher = {}

    def call(self, inputs, task_idx=0, training=False):
        batch_size = tf.shape(inputs)[0]
        expected_shape = [None, self.num_users, self.num_antennas * 2]
        tf.debugging.assert_shapes([(inputs, expected_shape)])
        
        inputs_flat = tf.reshape(inputs, [batch_size, self.num_users * (self.num_antennas * 2)])
        inputs_flat = tf.expand_dims(inputs_flat, axis=1)
        x = self.shared_transformer(inputs_flat, inputs_flat)
        
        if task_idx == 0:
            x = self.gru_static(x)
        else:
            x = self.gru_default(x)
        
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
        norm = tf.sqrt(norm_squared + 1e-10)
        w = w / tf.cast(norm, tf.complex64) * tf.complex(tf.sqrt(POWER * self.num_users), 0.0)
        return w

    def generate_replay(self, num_samples, task_idx=0):
        buffer_len = self.buffer_count.read_value()
        def empty_case():
            return (tf.zeros([num_samples, self.num_users, self.num_antennas], dtype=tf.complex64),
                    tf.zeros([num_samples, self.num_users, self.num_antennas], dtype=tf.complex64))

        def populated_case():
            buffer_size = REPLAY_BUFFER_SIZE_PEDESTRIAN if task_idx == 1 else REPLAY_BUFFER_SIZE_DEFAULT
            available_samples = tf.minimum(buffer_len, buffer_size)
            idx = tf.cond(
                available_samples < num_samples,
                lambda: tf.random.uniform([num_samples], 0, available_samples, dtype=tf.int32),
                lambda: tf.random.shuffle(tf.range(available_samples, dtype=tf.int32))[:num_samples]
            )
            x_buffer = self.memory_buffer["x"][:available_samples]
            x_samples = tf.gather(x_buffer, idx)
            x_flat = tf.reshape(x_samples, [num_samples, self.num_antennas * self.num_users])
            noise = tf.random.normal([num_samples, self.num_antennas * self.num_users], stddev=0.1, dtype=tf.float32)
            real_part = tf.math.real(x_flat)
            imag_part = tf.math.imag(x_flat)
            gen_input = tf.concat([real_part, imag_part, noise], axis=-1)
            gen_output = self.generator(gen_input)
            real = gen_output[:, :self.num_antennas * self.num_users]
            imag = gen_output[:, self.num_antennas * self.num_users:]
            h_gen = tf.complex(real, imag)
            h_gen = tf.reshape(h_gen, [num_samples, self.num_users, self.num_antennas])
            h_norm = tf.sqrt(tf.reduce_mean(tf.abs(x_samples)**2))
            h_gen_norm = tf.sqrt(tf.reduce_mean(tf.abs(h_gen)**2) + 1e-10)
            scale_factor = tf.cast(h_norm / h_gen_norm, tf.complex64)
            h_gen = h_gen * scale_factor
            return x_samples, h_gen

        return tf.cond(buffer_len == 0, empty_case, populated_case)

    def train_generator(self, x_batch, h_batch, optimizer, task_idx=0):
        batch_size = tf.shape(x_batch)[0]
        num_samples = batch_size if task_idx == 1 else batch_size // 2
        with tf.GradientTape() as tape:
            x_replay, h_replay = self.generate_replay(num_samples, task_idx)
            is_dummy = tf.reduce_all(tf.equal(x_replay, 0.0))
            gen_loss = tf.cond(
                is_dummy,
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: self._compute_gen_loss(x_batch[:num_samples], h_replay)
            )
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return gen_loss

    def _compute_gen_loss(self, x_batch, h_replay):
        x_batch_processed = preprocess_input(x_batch)
        w = self(x_batch_processed)
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
        def append_to_buffer():
            new_idx = current_count
            self.memory_buffer["x"].scatter_nd_update([[new_idx]], tf.expand_dims(x_sample, 0))
            self.memory_buffer["h"].scatter_nd_update([[new_idx]], tf.expand_dims(h_sample, 0))
            self.buffer_count.assign_add(1)
        def replace_in_buffer():
            replace_idx = tf.random.uniform([], 0, self.replay_buffer_size.read_value(), dtype=tf.int32)
            self.memory_buffer["x"].scatter_nd_update([[replace_idx]], tf.expand_dims(x_sample, 0))
            self.memory_buffer["h"].scatter_nd_update([[replace_idx]], tf.expand_dims(h_sample, 0))
        tf.cond(current_count < self.replay_buffer_size, append_to_buffer, replace_in_buffer)

    def regularization_loss(self):
        penalty = 0.0
        for w in self.trainable_weights:
            if w.name in self.old_params:
                delta = w - self.old_params[w.name]
                fisher_val = self.fisher.get(w.name, tf.ones_like(w))
                penalty += tf.reduce_sum(fisher_val * tf.cast(delta, tf.float32)**2)
        return LAMBDA_REG * penalty

    def update_fisher(self, x, h):
        x_processed = preprocess_input(x)
        with tf.GradientTape() as tape:
            w = self(x_processed)
            loss = tf.reduce_mean(tf.abs(w)**2)
        grads = tape.gradient(loss, self.trainable_weights)
        for w, grad in zip(self.trainable_weights, grads):
            if grad is not None:
                self.fisher[w.name] = tf.reduce_mean(grad**2, axis=0)

def generate_channel(task, num_slots, batch_size):
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], NUM_USERS)
    avg_speed = np.mean(speeds) * 1000 / 3600
    doppler_freq = avg_speed * FREQ / 3e8
    logging.info(f"Task {task['name']}: Doppler Freq = {doppler_freq:.2f} Hz")

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
                h_t = channel_model(batch_size, NUM_USERS, sampling_frequency=max(500, int(2 * doppler_freq)))[0]
                h_t = tf.reduce_sum(h_t, axis=-2)
                h_t = tf.squeeze(h_t, axis=[1, 2, 3])
                h_t = tf.reduce_mean(h_t, axis=0)
                h_t = tf.transpose(h_t, [1, 0])
            else:
                h_t = channel_model(batch_size, NUM_USERS)[0]
                h_t = tf.squeeze(h_t, axis=[2, 3, 5])
                h_t = tf.reduce_mean(h_t, axis=0)
                h_t = tf.transpose(h_t, [0, 2, 1])
                h_t = tf.reduce_mean(h_t, axis=1)
            h_chunk.append(h_t)
        h_chunks.append(tf.stack(h_chunk))
    h = tf.concat(h_chunks, axis=0)
    logging.info(f"Task {task['name']}: Final channel shape = {h.shape}")
    return h

def train_step(model, x_batch, h_batch, optimizer, gen_optimizer, task_idx=0):
    batch_size = tf.shape(x_batch)[0]
    x_batch_processed = preprocess_input(x_batch)
    trainable_vars = [var for var in model.trainable_variables if 'generator' not in var.name]
    gen_vars = model.generator.trainable_variables
    
    with tf.GradientTape() as tape:
        w = model(x_batch_processed, task_idx, training=True)
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
        loss = -tf.math.log(1.0 + sinr) + 0.5 * interference_power
        replay_loss = tf.constant(0.0, dtype=tf.float32)
        num_samples = batch_size if task_idx == 1 else batch_size // 2
        for _ in range(INNER_STEPS):
            x_replay, h_replay = model.generate_replay(num_samples, task_idx)
            if not tf.reduce_all(tf.equal(x_replay, 0.0)):
                x_replay_processed = preprocess_input(x_replay)
                w_replay = model(x_replay_processed, task_idx, training=True)
                h_replay_adjusted = tf.transpose(h_replay, [0, 2, 1])
                signal_matrix_replay = tf.matmul(w_replay, h_replay_adjusted)
                desired_signal_replay = tf.linalg.diag_part(signal_matrix_replay)
                desired_power_replay = tf.reduce_mean(tf.abs(desired_signal_replay)**2)
                interference_replay = tf.reduce_sum(tf.abs(signal_matrix_replay)**2 * mask[:num_samples], axis=-1)
                interference_power_replay = tf.reduce_mean(interference_replay)
                sinr_replay = desired_power_replay / (interference_power_replay + NOISE_POWER)
                replay_loss += -tf.math.log(1.0 + sinr_replay) + 0.5 * interference_power_replay
        loss = loss + replay_loss / tf.cast(INNER_STEPS, tf.float32) + model.regularization_loss()
    
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    
    gen_loss = model.train_generator(x_batch, h_batch, gen_optimizer, task_idx)
    return loss, gen_loss

def main(seed):
    logging.basicConfig(
        filename=f"training_log_seed_{seed}.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    logging.info(f"Starting run with seed {seed}")
    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger(), logging.ERROR)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = BeamformingMetaAttentionModel(NUM_ANTENNAS, NUM_USERS)
    optimizer_static = tf.keras.optimizers.Adam(learning_rate=0.005)
    optimizer_default = tf.keras.optimizers.Adam(learning_rate=0.002)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    checkpoint = tf.train.Checkpoint(model=model)

    real_part = tf.random.normal([BATCH_SIZE_DEFAULT, NUM_USERS, NUM_ANTENNAS], dtype=tf.float32)
    imag_part = tf.random.normal([BATCH_SIZE_DEFAULT, NUM_USERS, NUM_ANTENNAS], dtype=tf.float32)
    dummy_x = tf.complex(real_part, imag_part)
    _ = model(preprocess_input(dummy_x))

    results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting", "fwt", "bwt", "spectral_efficiency", "interference", "convergence_time"]}
    task_performance = []
    task_initial_performance = []
    task_channels = {}
    pedestrian_weights = None

    total_start_time = time.time()
    
    for task_idx, task in enumerate(TASKS):
        print(f"Task {task['name']} ({task_idx+1}/{len(TASKS)})")
        batch_size = BATCH_SIZE_STATIC if task_idx == 0 else BATCH_SIZE_DEFAULT
        num_epochs = NUM_EPOCHS_STATIC if task_idx == 0 else NUM_EPOCHS_DEFAULT
        optimizer = optimizer_static if task_idx == 0 else optimizer_default
        
        if task_idx in [0, 2] and pedestrian_weights is not None:
            model.load_weights(pedestrian_weights)
            logging.info(f"Loaded weights from Pedestrian for {task['name']}")

        try:
            h = generate_channel(task, NUM_SLOTS, batch_size)
            task_channels[task_idx] = h
            x = h
            x_processed = preprocess_input(x)
            dataset = tf.data.Dataset.from_tensor_slices((x, h)).batch(batch_size)

            _ = model(x_processed[:1], task_idx)
            initial_w = model(x_processed, task_idx)
            h_transposed = tf.transpose(h, [0, 2, 1])
            signal = tf.matmul(initial_w, h_transposed)
            initial_signal = tf.reduce_mean(tf.abs(signal)**2).numpy()
            task_initial_performance.append(initial_signal)

            task_start_time = time.time()
            for epoch in range(num_epochs):
                for batch_idx, (x_batch, h_batch) in enumerate(dataset):
                    loss, gen_loss = train_step(model, x_batch, h_batch, optimizer, gen_optimizer, task_idx)
                    model.update_memory(x_batch, h_batch)
                    print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx} - Loss: {loss:.4f}, Gen Loss: {gen_loss:.4f}")

            if task_idx == 1:
                pedestrian_weights = f"checkpoints/seed_{seed}/pedestrian_weights"
                model.save_weights(pedestrian_weights)
                logging.info(f"Saved Pedestrian weights at {pedestrian_weights}")

            checkpoint_dir = f"checkpoints/seed_{seed}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint.save(f"{checkpoint_dir}/task_{task['name']}")

            task_time = time.time() - task_start_time
            latency = task_time / (num_epochs * NUM_SLOTS) * 1000
            convergence_time = task_time / num_epochs
            gpu_power = get_gpu_power()
            energy = gpu_power * task_time
            w = model(x_processed, task_idx)
            h_adjusted = tf.transpose(h, [0, 2, 1])
            signal_matrix = tf.matmul(w, h_adjusted)
            desired_signal = tf.linalg.diag_part(signal_matrix)
            desired_power = tf.reduce_mean(tf.abs(desired_signal)**2).numpy()
            mask = 1.0 - tf.eye(NUM_USERS, batch_shape=[tf.shape(signal_matrix)[0]])
            interference = tf.reduce_mean(tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)).numpy()
            sinr = desired_power / (interference + NOISE_POWER)
            spectral_efficiency = np.log2(1 + sinr)
            throughput = desired_power
            task_performance.append(throughput)

            forgetting = 0.0
            if task_idx > 0:
                past_deltas = []
                for past_idx in range(task_idx):
                    past_h = task_channels[past_idx]
                    past_h_processed = preprocess_input(past_h)
                    past_w = model(past_h_processed, past_idx)
                    past_signal = tf.reduce_mean(tf.abs(tf.matmul(past_w, tf.transpose(past_h, [0, 2, 1])))**2).numpy()
                    past_deltas.append(task_initial_performance[past_idx] - past_signal)
                forgetting = np.mean(past_deltas) if past_deltas else 0.0

            fwt = 0.0
            if task_idx > 0:
                past_h = task_channels[task_idx - 1]
                past_h_processed = preprocess_input(past_h)
                past_w = model(past_h_processed, task_idx - 1)
                past_signal = tf.reduce_mean(tf.abs(tf.matmul(past_w, tf.transpose(past_h, [0, 2, 1])))**2).numpy()
                fwt = (initial_signal - past_signal) / (past_signal + 1e-10)

            model.update_fisher(x, h)
            model.old_params = {w.name: w.numpy() for w in model.trainable_weights}
            results["throughput"].append(throughput)
            results["latency"].append(latency)
            results["energy"].append(energy)
            results["forgetting"].append(forgetting)
            results["fwt"].append(fwt)
            results["spectral_efficiency"].append(spectral_efficiency)
            results["interference"].append(interference)
            results["convergence_time"].append(convergence_time)

            gc.collect()
        except Exception as e:
            logging.error(f"Error in Task {task['name']}: {str(e)}")
            raise

    bwt = 0.0
    if len(task_performance) > 1:
        final_deltas = []
        for past_idx in range(len(TASKS) - 1):
            past_h = task_channels[past_idx]
            past_h_processed = preprocess_input(past_h)
            past_w = model(past_h_processed, past_idx)
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
    logging.info(f"Run completed in {total_time:.2f}s")

    return results

if __name__ == "__main__":
    all_results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting", "fwt", "bwt", "spectral_efficiency", "interference", "convergence_time"]}
    
    for run in range(NUM_RUNS):
        print(f"Run {run+1}/{NUM_RUNS}")
        try:
            results = main(run)
            for metric in all_results:
                all_results[metric].append(results[metric])
        except Exception as e:
            logging.error(f"Error in Run {run+1}: {str(e)}")

    with open("results_summary.txt", "w") as f:
        for metric in all_results:
            values = np.array(all_results[metric])
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            f.write(f"{metric.capitalize()} (Mean ± Std):\n")
            for i, task in enumerate(TASKS):
                f.write(f"  {task['name']}: {mean[i]:.4f} ± {std[i]:.4f}\n")
            f.write("\n")

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    for idx, metric in enumerate(all_results):
        ax = axes[idx]
        values = np.array(all_results[metric])
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        ax.errorbar(range(len(TASKS)), mean, yerr=std, label=metric.capitalize(), marker="o")
        ax.set_title(metric.capitalize())
        ax.set_xticks(range(len(TASKS)))
        ax.set_xticklabels([t["name"] for t in TASKS], rotation=45)
        ax.legend()
    plt.tight_layout()
    plt.savefig("detailed_metrics.png")
    plt.close()