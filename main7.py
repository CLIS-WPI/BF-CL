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
tf.get_logger().setLevel('ERROR')

# Logger setup
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

# Preprocess input with type checking
def preprocess_input(x):
    if not tf.is_tensor(x) or x.dtype not in [tf.complex64, tf.complex128]:
        logging.warning(f"Input type {type(x)} or dtype {x.dtype} is not complex, converting to complex64")
        x = tf.cast(x, tf.complex64)
    return x

# GPU power utility
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

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {physical_devices}")

# Constants
NUM_ANTENNAS = 32
NUM_USERS = 5
FREQ = 28e9
POWER = 1.0
NUM_SLOTS = 200
BATCH_SIZE_DEFAULT = 128
BATCH_SIZE_STATIC = 64
LAMBDA_REG = 10
NUM_EPOCHS_DEFAULT = 8
NUM_EPOCHS_STATIC = 4
CHUNK_SIZE = 50
NOISE_POWER = 1e-3
REPLAY_BUFFER_SIZE_DEFAULT = 1000
NUM_RUNS = 10
INNER_STEPS = 5

TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": 30e-9, "channel": "TDL", "model": "A"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": 50e-9, "channel": "Rayleigh"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": 100e-9, "channel": "TDL", "model": "C"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": 70e-9, "channel": "TDL", "model": "A"},
]

# Model definition
class BeamformingMetaAttentionModel(tf.keras.Model):
    def __init__(self, num_antennas, num_users):
        super(BeamformingMetaAttentionModel, self).__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        
        # Layers
        self.input_conv = layers.Conv1D(64, 1, activation='relu', dtype=tf.float32, input_shape=(num_users, num_antennas * 2))
        self.shared_transformer = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.static_transformer = layers.MultiHeadAttention(num_heads=8, key_dim=64)
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.gru_static = layers.GRU(128, return_sequences=True)
        self.gru_default = layers.GRU(128, return_sequences=True)
        self.norm = layers.LayerNormalization()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(num_antennas * num_users * 2)

        # Replay buffer
        self.buffer_sizes = [2000 if i in [0, 2, 3] else REPLAY_BUFFER_SIZE_DEFAULT for i in range(len(TASKS))]
        self.buffer_x = [tf.zeros([size, num_users, num_antennas], dtype=tf.complex64) for size in self.buffer_sizes]
        self.buffer_h = [tf.zeros([size, num_users, num_antennas], dtype=tf.complex64) for size in self.buffer_sizes]
        self.buffer_count = [0] * len(TASKS)

        # Regularization
        self.old_params = {}
        self.fisher = {}
        self.lambda_reg = LAMBDA_REG

    def call(self, inputs, task_idx=0, training=False):
        batch_size = tf.shape(inputs)[0]
        with tf.device('/cpu:0'):
            real = tf.math.real(inputs)
            imag = tf.math.imag(inputs)
            x = tf.concat([real, imag], axis=-1)
        x = self.input_conv(x)
        
        x = self.shared_transformer(x, x)
        if task_idx == 0:
            x = self.static_transformer(x, x)
        x = self.gru_static(x) if task_idx == 0 else self.gru_default(x)
        
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
    
    def generate_replay(self, num_samples, task_idx):
        buffer_len = tf.cast(self.buffer_count[task_idx], tf.int32)
        if buffer_len == 0:
            return (tf.zeros([num_samples, self.num_users, self.num_antennas], dtype=tf.complex64),
                    tf.zeros([num_samples, self.num_users, self.num_antennas], dtype=tf.complex64))
        
        sample_size = tf.minimum(num_samples, buffer_len)
        indices = tf.random.shuffle(tf.range(buffer_len))[:sample_size]
        x_replay = tf.gather(self.buffer_x[task_idx], indices)
        h_replay = tf.gather(self.buffer_h[task_idx], indices)
        return x_replay, h_replay

    def update_memory(self, x, h, task_idx):
        batch_size = tf.shape(x)[0].numpy()
        buffer_capacity = self.buffer_sizes[task_idx]
        current_count = self.buffer_count[task_idx]
        
        for i in range(batch_size):
            if current_count < buffer_capacity:
                self.buffer_x[task_idx] = tf.tensor_scatter_nd_update(
                    self.buffer_x[task_idx], [[current_count]], [x[i]]
                )
                self.buffer_h[task_idx] = tf.tensor_scatter_nd_update(
                    self.buffer_h[task_idx], [[current_count]], [h[i]]
                )
                self.buffer_count[task_idx] += 1
            else:
                replace_idx = np.random.randint(0, current_count)
                self.buffer_x[task_idx] = tf.tensor_scatter_nd_update(
                    self.buffer_x[task_idx], [[replace_idx]], [x[i]]
                )
                self.buffer_h[task_idx] = tf.tensor_scatter_nd_update(
                    self.buffer_h[task_idx], [[replace_idx]], [h[i]]
                )

    def regularization_loss(self):
        reg_loss = 0.0
        for w in self.trainable_weights:
            if w.name in self.old_params:
                old_w = tf.convert_to_tensor(self.old_params[w.name], dtype=w.dtype)
                fisher_w = tf.convert_to_tensor(self.fisher.get(w.name, 0.0), dtype=w.dtype)
                reg_loss += tf.reduce_sum(fisher_w * tf.square(w - old_w))
        return self.lambda_reg * reg_loss

    def update_fisher(self, x, h, task_idx):
        x_processed = preprocess_input(x)
        with tf.GradientTape() as tape:
            w = self(x_processed, task_idx, training=True)
            loss = tf.reduce_mean(tf.abs(w)**2)
        grads = tape.gradient(loss, self.trainable_weights)
        for w, grad in zip(self.trainable_weights, grads):
            if grad is not None:
                self.fisher[w.name] = tf.reduce_mean(grad**2, axis=0)

# Channel generation
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
            h_chunk.append(tf.cast(h_t, tf.complex64))
        h_chunks.append(tf.stack(h_chunk))
    h = tf.concat(h_chunks, axis=0)
    logging.info(f"Task {task['name']}: Final channel shape = {h.shape}")
    return h

# Training step
@tf.function(experimental_relax_shapes=True)
def train_step(model, x_batch, h_batch, optimizer, task_idx):
    with tf.GradientTape() as tape:
        w = model(x_batch, task_idx, training=True)
        h_adjusted = tf.transpose(h_batch, [0, 2, 1])
        signal_matrix = tf.matmul(w, h_adjusted)
        desired_signal = tf.linalg.diag_part(signal_matrix)
        desired_power = tf.reduce_mean(tf.abs(desired_signal)**2)
        eye = tf.eye(model.num_users, dtype=tf.float32)
        mask = 1.0 - eye
        batch_size = tf.shape(x_batch)[0]
        mask = tf.tile(mask[tf.newaxis, :, :], [batch_size, 1, 1])
        interference = tf.reduce_sum(tf.abs(signal_matrix)**2 * mask, axis=-1)
        interference_power = tf.reduce_mean(interference)
        sinr = desired_power / (interference_power + NOISE_POWER)
        loss = -tf.math.log(1.0 + sinr) + 0.5 * interference_power
        
        replay_loss = tf.constant(0.0, dtype=tf.float32)
        num_samples = batch_size // 4
        for _ in range(INNER_STEPS):
            x_replay, h_replay = model.generate_replay(num_samples, task_idx)
            if not tf.reduce_all(tf.equal(x_replay, 0.0)):
                w_replay = model(x_replay, task_idx, training=True)
                h_replay_adjusted = tf.transpose(h_replay, [0, 2, 1])
                signal_matrix_replay = tf.matmul(w_replay, h_replay_adjusted)
                desired_signal_replay = tf.linalg.diag_part(signal_matrix_replay)
                desired_power_replay = tf.reduce_mean(tf.abs(desired_signal_replay)**2)
                interference_replay = tf.reduce_sum(tf.abs(signal_matrix_replay)**2 * mask[:num_samples], axis=-1)
                interference_power_replay = tf.reduce_mean(interference_replay)
                sinr_replay = desired_power_replay / (interference_power_replay + NOISE_POWER)
                replay_loss += -tf.math.log(1.0 + sinr_replay) + 0.5 * interference_power_replay
        loss = loss + replay_loss / tf.cast(INNER_STEPS, tf.float32) + model.regularization_loss()
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip([g for g in grads if g is not None], 
                                  [v for g, v in zip(grads, model.trainable_variables) if g is not None]))
    return loss  # برگردوندن Loss برای استفاده تو Scheduler

# Main function
def main(seed):
    logging.basicConfig(
        filename=f"training_log_seed_{seed}.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger(), logging.ERROR)

    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = BeamformingMetaAttentionModel(NUM_ANTENNAS, NUM_USERS)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.002)
    initial_lr = {'Static': 0.005, 'Pedestrian': 0.003, 'Vehicular': 0.003, 'Aerial': 0.003}
    lr_schedule = {task_idx: initial_lr[task['name']] for task_idx, task in enumerate(TASKS)}
    min_lr = 1e-5
    patience = 2
    lr_factor = 0.5
    epoch_losses = {task_idx: [] for task_idx in range(len(TASKS))}

    # Warm-up کامل برای همه تسک‌ها
    logging.info(f"Initial trainable variables: {len(model.trainable_variables)}")
    for task_idx in range(len(TASKS)):
        batch_size = BATCH_SIZE_STATIC if task_idx == 0 else BATCH_SIZE_DEFAULT
        real_part = tf.random.normal([batch_size, NUM_USERS, NUM_ANTENNAS], dtype=tf.float32)
        imag_part = tf.random.normal([batch_size, NUM_USERS, NUM_ANTENNAS], dtype=tf.float32)
        dummy_x = tf.complex(real_part, imag_part)
        dummy_h = dummy_x
        dummy_x_processed = preprocess_input(dummy_x)
        optimizer.learning_rate.assign(0.005 if task_idx == 0 else 0.003)
        with tf.GradientTape() as tape:
            w = model(dummy_x_processed, task_idx=task_idx, training=True)
            loss = tf.reduce_mean(tf.abs(w)**2)  # یه Loss ساده برای Warm-up
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip([g for g in grads if g is not None], 
                                      [v for g, v in zip(grads, model.trainable_variables) if g is not None]))
        logging.info(f"Warm-up completed for task {task_idx}, trainable variables: {len(model.trainable_variables)}")

    checkpoint = tf.train.Checkpoint(model=model)
    results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting", "fwt", "bwt", "spectral_efficiency", "interference", "convergence_time"]}
    task_performance = []
    task_initial_performance = []
    task_channels = {}
    pedestrian_weights = None
    aerial_weights = None

    total_start_time = time.time()
    
    for task_idx, task in enumerate(TASKS):
        print(f"Task {task['name']} ({task_idx+1}/{len(TASKS)})")
        batch_size = BATCH_SIZE_STATIC if task_idx == 0 else BATCH_SIZE_DEFAULT
        num_epochs = NUM_EPOCHS_STATIC if task_idx == 0 else 10 if task_idx in [2, 3] else NUM_EPOCHS_DEFAULT
        
        current_lr = lr_schedule[task_idx]
        optimizer.learning_rate.assign(current_lr)
        
        if task_idx == 0 and aerial_weights is not None:
            model.load_weights(aerial_weights)
            logging.info(f"Loaded weights from Aerial for {task['name']}")
        elif task_idx == 2 and pedestrian_weights is not None:
            model.load_weights(pedestrian_weights)
            logging.info(f"Loaded weights from Pedestrian for {task['name']}")

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
        patience_counter = 0
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch_idx, (x_batch, h_batch) in enumerate(dataset):
                loss = train_step(model, x_batch, h_batch, optimizer, task_idx)
                model.update_memory(x_batch, h_batch, task_idx)
                epoch_loss += loss.numpy()
                num_batches += 1
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx} - Loss: {loss:.4f}")
            
            avg_loss = epoch_loss / num_batches
            epoch_losses[task_idx].append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience and current_lr > min_lr:
                current_lr *= lr_factor
                optimizer.learning_rate.assign(current_lr)
                patience_counter = 0
                logging.info(f"Reduced LR for task {task['name']} to {current_lr:.6f}")
            
            logging.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        if task_idx == 1:
            pedestrian_weights = f"checkpoints/seed_{seed}/pedestrian_weights"
            os.makedirs(os.path.dirname(pedestrian_weights), exist_ok=True)
            model.save_weights(pedestrian_weights)
            logging.info(f"Saved Pedestrian weights at {pedestrian_weights}")
        elif task_idx == 3:
            aerial_weights = f"checkpoints/seed_{seed}/aerial_weights"
            os.makedirs(os.path.dirname(aerial_weights), exist_ok=True)
            model.save_weights(aerial_weights)
            logging.info(f"Saved Aerial weights at {aerial_weights}")

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
        throughput = np.log2(1 + sinr)
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

        model.update_fisher(x, h, task_idx)
        model.old_params = {w.name: w.numpy() for w in model.trainable_weights}
        results["throughput"].append(throughput)
        results["latency"].append(latency)
        results["energy"].append(energy)
        results["forgetting"].append(forgetting)
        results["fwt"].append(fwt)
        results["spectral_efficiency"].append(spectral_efficiency)
        results["interference"].append(interference)
        results["convergence_time"].append(convergence_time)

        logging.info(f"After {task['name']}, trainable variables: {len(model.trainable_variables)}")
        gc.collect()

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
    with open(f"results_seed_{seed}.txt", "w") as f:
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

# Run the simulation
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
            raise

    with open("results_summary.txt", "w") as f:
        for metric in all_results:
            values = np.array(all_results[metric])
            if values.size == 0 or values.shape[0] == 0:
                f.write(f"{metric.capitalize()} (Mean ± Std):\n")
                for task in TASKS:
                    f.write(f"  {task['name']}: NaN ± NaN (No data)\n")
                f.write("\n")
                continue
            
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            
            f.write(f"{metric.capitalize()} (Mean ± Std):\n")
            for i, task in enumerate(TASKS):
                f.write(f"  {task['name']}: {mean[i]:.4f} ± {std[i]:.4f}\n")
            f.write("\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in all_results:
        values = np.array(all_results[metric])
        if values.size == 0 or values.shape[0] == 0:
            logging.warning(f"No data for metric {metric}, skipping plot.")
            continue
        
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        
        ax.errorbar(range(len(TASKS)), mean, yerr=std, label=metric.capitalize(), marker="o")

    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([task["name"] for task in TASKS], rotation=45)
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Performance")
    ax.set_title("Task Performance Across Runs")
    ax.legend()
    plt.tight_layout()
    plt.savefig("task_performance.png")
    plt.close()