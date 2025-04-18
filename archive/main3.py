import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import time
import gc
import os

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Initial Settings
NUM_ANTENNAS = 32
NUM_USERS = 5
FREQ = 28e9
POWER = 1.0
NUM_SLOTS = 200
BATCH_SIZE = 18
LAMBDA_SI = 50.0  # کاهش از 100.0 به 50.0
NUM_EPOCHS = 4
CHUNK_SIZE = 50
GPU_POWER_DRAW = 400
NOISE_POWER = 0.1
REPLAY_BUFFER_SIZE = 200
NUM_RUNS = 10

# Define tasks with their properties
TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": 30e-9, "channel": "TDL", "model": "A"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": 70e-9, "channel": "TDL", "model": "A"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": 100e-9, "channel": "TDL", "model": "C"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": 50e-9, "channel": "Rayleigh"},
    {"name": "Random", "speed_range": [10, 100], "delay_spread": 90e-9, "channel": "Rayleigh"},
]

# Beamforming Model with Synaptic Intelligence (SI)
class BeamformingModelWithSI(Model):
    def __init__(self, num_antennas, num_users):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.dense1_real = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.001))  # افزایش از 64 به 128
        self.dense1_imag = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.001))  # افزایش از 64 به 128
        self.dense2_real = layers.Dense(64, activation="relu")  # افزایش از 32 به 64
        self.dense2_imag = layers.Dense(64, activation="relu")  # افزایش از 32 به 64
        self.output_real = layers.Dense(num_antennas * num_users)
        self.output_imag = layers.Dense(num_antennas * num_users)
        self.omega = {}
        self.old_params = {}
        self.task_params = {}

    @tf.function
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        rank = tf.rank(inputs)

        if rank == 4:
            batch_size = input_shape[0]
            slots_per_batch = input_shape[1]
        else:
            batch_size = input_shape[0]
            slots_per_batch = 1
            inputs = tf.expand_dims(inputs, axis=1)

        inputs_flat = tf.reshape(inputs, [batch_size * slots_per_batch, self.num_antennas * self.num_users])
        real_inputs = tf.cast(tf.math.real(inputs_flat), tf.float32)
        imag_inputs = tf.cast(tf.math.imag(inputs_flat), tf.float32)
        real_x = self.dense1_real(real_inputs)
        imag_x = self.dense1_imag(imag_inputs)
        real_x = self.dense2_real(real_x)
        imag_x = self.dense2_imag(imag_x)
        real_output = self.output_real(real_x)
        imag_output = self.output_imag(imag_x)
        w = tf.complex(real_output, imag_output)
        w = tf.reshape(w, [batch_size, slots_per_batch, self.num_users, self.num_antennas])
        norm_squared = tf.reduce_sum(tf.abs(w)**2, axis=-1, keepdims=True)
        norm = tf.cast(tf.sqrt(norm_squared + 1e-10), dtype=tf.complex64)  # Add epsilon to prevent division by zero
        power = tf.complex(tf.sqrt(POWER), 0.0)
        w = w / norm * power
        return w

    def track_omega(self, gradients):
        for w, g in zip(self.trainable_weights, gradients):
            if w.name not in self.omega:
                self.omega[w.name] = tf.zeros_like(w)
            if g is not None:
                self.omega[w.name] += tf.reduce_mean(g**2)

    def si_loss(self, lambda_si=2000.0):
        penalty = 0.0
        for w in self.trainable_weights:
            if w.name in self.old_params:
                delta = w - self.old_params[w.name]
                penalty += tf.reduce_sum(self.omega[w.name] * delta**2)
        return lambda_si * penalty

    def save_task_params(self, task_idx):
        self.task_params[task_idx] = {w.name: w.numpy() for w in self.trainable_weights}

    def _trackable_children(self, save_type="checkpoint", **kwargs):
        children = super()._trackable_children(save_type, **kwargs)
        if save_type == "checkpoint":
            children.pop("omega", None)
            children.pop("old_params", None)
            children.pop("task_params", None)
        return children

# Variational Autoencoder (VAE) for Generative Replay
class VAE(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32)  # Latent space dimension
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(input_dim, activation="linear"),
            layers.Reshape([NUM_ANTENNAS, NUM_USERS])
        ])

    def call(self, inputs):
        z = self.encoder(inputs)
        return self.decoder(z)

    def train_vae(self, data, epochs=10):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                reconstructed = self(data)
                loss = tf.reduce_mean(tf.square(reconstructed - data))
            grads = tape.gradient(loss, self.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.trainable_weights))
            print(f"VAE Epoch {epoch+1}/{epochs}, Loss: {loss.numpy():.4f}")

# SI Loss Function
def si_loss(model, x, h, lambda_si=2000.0):
    w = tf.cast(model(x), tf.complex64)
    w_real = tf.math.real(w)
    w_imag = tf.math.imag(w)
    if tf.reduce_any(tf.math.is_nan(w_real)) or tf.reduce_any(tf.math.is_nan(w_imag)):
        print("Warning: NaN detected in w")
    
    rank = tf.rank(h)
    if rank == 3:
        h = tf.expand_dims(h, axis=1)
    
    h_adjusted = tf.cast(tf.transpose(h, [0, 1, 3, 2]), tf.complex64)
    h_real = tf.math.real(h_adjusted)
    h_imag = tf.math.imag(h_adjusted)
    if tf.reduce_any(tf.math.is_nan(h_real)) or tf.reduce_any(tf.math.is_nan(h_imag)):
        print("Warning: NaN detected in h_adjusted")
    
    product = w * h_adjusted
    product_real = tf.math.real(product)
    product_imag = tf.math.imag(product)
    if tf.reduce_any(tf.math.is_nan(product_real)) or tf.reduce_any(tf.math.is_nan(product_imag)):
        print("Warning: NaN detected in product")
    
    sum_result = tf.reduce_sum(product, axis=-1)
    sum_result_real = tf.math.real(sum_result)
    sum_result_imag = tf.math.imag(sum_result)
    if tf.reduce_any(tf.math.is_nan(sum_result_real)) or tf.reduce_any(tf.math.is_nan(sum_result_imag)):
        print("Warning: NaN detected in sum_result")
    
    abs_sum = tf.abs(sum_result)
    if tf.reduce_any(tf.math.is_nan(abs_sum)):
        print("Warning: NaN detected in abs_sum")
    
    signal_power = tf.reduce_mean(abs_sum**2)
    if tf.reduce_any(tf.math.is_nan(signal_power)):
        print("Warning: NaN detected in signal_power")
    
    loss = -signal_power
    si_penalty = model.si_loss(lambda_si)
    si_penalty_value = si_penalty.numpy() if isinstance(si_penalty, tf.Tensor) else si_penalty
    print(f"SI Penalty: {si_penalty_value:.6f}")
    if tf.reduce_any(tf.math.is_nan(si_penalty)):
        print("Warning: NaN detected in si_penalty")
    
    loss += si_penalty
    if tf.reduce_any(tf.math.is_nan(loss)):
        print("Warning: NaN detected in loss")
    
    return loss

# Debug Channel Properties
def debug_channel_properties(task_name, h_sample):
    h_sample_np = h_sample.numpy() if isinstance(h_sample, tf.Tensor) else h_sample
    print(f"Debug for {task_name} channel:")
    print(f"Shape: {h_sample_np.shape}")
    print(f"Mean magnitude: {np.mean(np.abs(h_sample_np))}")
    print(f"Std magnitude: {np.std(np.abs(h_sample_np))}")
    print(f"Min/Max magnitude: {np.min(np.abs(h_sample_np))}, {np.max(np.abs(h_sample_np))}")
    condition_numbers = []
    for i in range(min(5, h_sample_np.shape[0])):
        h_mat = h_sample_np[i, 0]
        if h_mat.shape[0] == h_mat.shape[1]:
            condition_numbers.append(np.linalg.cond(h_mat))
    print(f"Condition number: {np.mean(condition_numbers) if condition_numbers else 'N/A'}")
    print("-" * 30)

# Generate Channel Data
def generate_channel(task, num_slots, task_idx=0):
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], NUM_USERS)
    avg_speed = np.mean(speeds) * 1000 / 3600  # Convert km/h to m/s
    doppler_freq = avg_speed * FREQ / 3e8  # Doppler frequency in Hz
    
    if task["channel"] == "TDL":
        channel_model = sn.channel.tr38901.TDL(
            model=task["model"],
            delay_spread=task["delay_spread"],
            carrier_frequency=FREQ,
            num_tx_ant=NUM_ANTENNAS,
            num_rx_ant=NUM_USERS,
            dtype=tf.complex64
        )
    else:
        channel_model = sn.channel.RayleighBlockFading(
            num_rx=NUM_USERS,
            num_rx_ant=NUM_USERS,
            num_tx=NUM_ANTENNAS,
            num_tx_ant=NUM_ANTENNAS,
            dtype=tf.complex64
        )

    h_chunks = []
    start_time = time.time()
    print(f"Generating channel for {task['name']}...")
    
    for chunk_start in range(0, num_slots, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, num_slots)
        h_chunk = []
        for t in range(chunk_start, chunk_end):
            if task["channel"] == "TDL":
                channel_response = channel_model(
                    batch_size=BATCH_SIZE,
                    num_time_steps=1,
                    sampling_frequency=max(500, int(2 * doppler_freq))
                )
                h_t = channel_response[0]
                h_t = tf.reduce_sum(h_t, axis=5)
                h_t = tf.squeeze(h_t, axis=[3, 5])
                h_t = tf.squeeze(h_t, axis=1)
                h_t = tf.transpose(h_t, [0, 2, 1])
            else:
                h_t_tuple = channel_model(
                    batch_size=BATCH_SIZE,
                    num_time_steps=1
                )
                h_t = h_t_tuple[0]
                h_t = h_t[..., 0, :]
                h_t = tf.squeeze(h_t, axis=-1)
                h_t = tf.reduce_mean(h_t, axis=[2, 4])
                h_t = tf.transpose(h_t, [0, 2, 1])
                h_t_magnitude = tf.abs(h_t)
                target_mean_magnitude = 0.89
                current_mean_magnitude = tf.reduce_mean(h_t_magnitude)
                scaling_factor = target_mean_magnitude / (current_mean_magnitude + 1e-10)
                scaling_factor = tf.cast(scaling_factor, tf.complex64)
                h_t = h_t * scaling_factor
            
            expected_shape = (BATCH_SIZE, NUM_ANTENNAS, NUM_USERS)
            if h_t.shape != expected_shape:
                raise ValueError(
                    f"Unexpected shape for h_t in task {task['name']}: "
                    f"got {h_t.shape}, expected {expected_shape}"
                )
            
            h_chunk.append(h_t)
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        h_chunks.append(tf.stack(h_chunk))
    
    h = tf.concat(h_chunks, axis=0)
    print(f"Channel for {task['name']} generated in {time.time() - start_time:.2f}s")
    debug_channel_properties(task["name"], h[:5])
    return h

# Training Step with SI and Generative Replay
def train_step(model, x_batch, h_batch, optimizer, task_idx, replay_buffer=None, lambda_si=2000.0):
    with tf.GradientTape() as tape:
        loss = si_loss(model, x_batch, h_batch, lambda_si)
        if replay_buffer and len(replay_buffer[0]) > 0:
            replay_idx = tf.random.uniform(shape=[min(BATCH_SIZE, len(replay_buffer[0]))], minval=0, maxval=len(replay_buffer[0]), dtype=tf.int32)
            replay_x = tf.cast(tf.gather(replay_buffer[0], replay_idx), tf.complex64)
            replay_h = tf.cast(tf.gather(replay_buffer[1], replay_idx), tf.complex64)
            replay_loss = si_loss(model, replay_x, replay_h, lambda_si)
            loss += replay_loss
    grads = tape.gradient(loss, model.trainable_weights)
    grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    model.track_omega(grads)
    return loss

# Main Training Function
def main(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting"]}
    task_performance = []
    task_initial_performance = []
    task_channels = {}
    replay_buffer = [[], []]
    task_losses = {task["name"]: [] for task in TASKS}
    fwt_values = []
    vae_models = {}
    
    strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
    with strategy.scope():
        model = BeamformingModelWithSI(NUM_ANTENNAS, NUM_USERS)
        dummy_input = tf.zeros((BATCH_SIZE, 1, NUM_ANTENNAS, NUM_USERS), dtype=tf.complex64)
        model(dummy_input)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    results_file = f"results_seed_{seed}.txt"
    
    total_start_time = time.time()
    with open(results_file, 'w') as f:
        f.write("Beamforming Continual Learning Results\n")
        f.write(f"Tasks: {len(TASKS)}, Epochs: {NUM_EPOCHS}, Slots: {NUM_SLOTS}, GPU Power: {GPU_POWER_DRAW}W, Noise Power: {NOISE_POWER}\n")
        f.write("="*50 + "\n")
    
    random_model = BeamformingModelWithSI(NUM_ANTENNAS, NUM_USERS)
    random_model(dummy_input)
    random_model.set_weights([tf.random.normal(w.shape) for w in model.get_weights()])
    
    for task_idx, task in enumerate(TASKS):
        print(f"Starting Task {task['name']} ({task_idx+1}/{len(TASKS)})")
        h = generate_channel(task, NUM_SLOTS, task_idx)
        task_channels[task_idx] = h
        x = h
        dataset = tf.data.Dataset.from_tensor_slices((x, h)).batch(BATCH_SIZE)
        
        vae = VAE(NUM_ANTENNAS * NUM_USERS)
        vae_input = tf.reshape(tf.cast(x, tf.float32), [-1, NUM_ANTENNAS, NUM_USERS])
        vae.train_vae(vae_input, epochs=20)
        vae_models[task_idx] = vae
        
        if task_idx > 0:
            replay_x, replay_h = [], []
            for past_idx in range(task_idx):
                past_h = task_channels[past_idx]
                indices = np.random.choice(past_h.shape[0], size=24, replace=False)  # کاهش از 32 به 24
                sampled_h = tf.gather(past_h, indices)
                replay_x.append(sampled_h)
                replay_h.append(sampled_h)
            replay_buffer[0] = tf.concat(replay_x, axis=0).numpy()
            replay_buffer[1] = tf.concat(replay_h, axis=0).numpy()
        
        replay_buffer[0] = np.array(replay_buffer[0], dtype=np.complex64)
        replay_buffer[1] = np.array(replay_buffer[1], dtype=np.complex64)
        
        initial_w = model(x)
        initial_throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(initial_w * tf.transpose(h, [0, 1, 3, 2]), axis=-1))**2).numpy()
        random_w = random_model(x)
        random_throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(random_w * tf.transpose(h, [0, 1, 3, 2]), axis=-1))**2).numpy()
        fwt = (initial_throughput - random_throughput) / (random_throughput + 1e-10) if task_idx > 0 else 0.0
        fwt_values.append(fwt)
        task_initial_performance.append(initial_throughput)
        
        with strategy.scope():
            dist_dataset = strategy.experimental_distribute_dataset(dataset)
            num_batches = (x.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Training {task['name']} - {num_batches} batches per epoch...")
            
            task_start_time = time.time()
            for epoch in range(NUM_EPOCHS):
                epoch_loss = 0
                batches = 0
                for x_batch, h_batch in dist_dataset:
                    loss = strategy.run(train_step, args=(model, x_batch, h_batch, optimizer,
                                                        task_idx, replay_buffer, LAMBDA_SI))
                    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
                    epoch_loss += loss
                    batches += 1
                avg_epoch_loss = epoch_loss / batches
                task_losses[task["name"]].append(avg_epoch_loss.numpy())
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} for {task['name']}: Avg Loss = {avg_epoch_loss:.6f}")
            
        task_time = time.time() - task_start_time
        latency = task_time / (NUM_EPOCHS * NUM_SLOTS) * 1000
        w = model(x)
        throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(w * tf.transpose(h, [0, 1, 3, 2]), axis=-1))**2).numpy()
        capacity = np.log2(1 + throughput / NOISE_POWER)
        task_performance.append(throughput)
        
        forgetting = 0.0
        if task_idx > 0:
            past_performances = []
            for past_idx in range(task_idx):
                past_h = task_channels[past_idx]
                past_w = model(past_h)
                past_throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(past_w * tf.transpose(past_h, [0, 1, 3, 2]), axis=-1))**2).numpy()
                forgetting_delta = task_performance[past_idx] - past_throughput
                past_performances.append(forgetting_delta)
            forgetting = np.mean(past_performances) if past_performances else 0.0
        
        model.old_params = {w.name: w.numpy() for w in model.trainable_weights}
        model.save_task_params(task_idx)
        
        energy = GPU_POWER_DRAW * task_time
        results["throughput"].append(capacity)
        results["latency"].append(latency)
        results["energy"].append(energy)
        results["forgetting"].append(forgetting)

        with open(results_file, 'a') as f:
            f.write(f"Task {task['name']}:\n")
            f.write(f"  Capacity: {capacity:.4f} bits/s/Hz\n")
            f.write(f"  Latency: {latency:.2f} ms/slot\n")
            f.write(f"  Energy: {energy:.2f} J\n")
            f.write(f"  Forgetting: {forgetting:.4f}\n")
            f.write(f"  FWT: {fwt:.4f}\n")
            f.write(f"  Losses: {[float(loss) for loss in task_losses[task['name']]]}\n")
            f.write(f"  Total Task Time: {task_time:.2f}s\n")
            f.write("-"*50 + "\n")
        
        weights_dir = os.path.join(save_dir, f"model_task_{task['name']}_seed_{seed}")
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        model.save_weights(os.path.join(weights_dir, "weights"))
        print(f"Task {task['name']} completed")
    
    bwt = 0.0
    if len(task_performance) > 1:
        final_performances = []
        for past_idx in range(len(TASKS)-1):
            past_h = task_channels[past_idx]
            past_w = model(past_h)
            past_throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(past_w * tf.transpose(past_h, [0, 1, 3, 2]), axis=-1))**2).numpy()
            final_performances.append(past_throughput - task_initial_performance[past_idx])
        bwt = np.mean(final_performances) if final_performances else 0.0

    total_training_time = time.time() - total_start_time
    with open(results_file, 'a') as f:
        f.write("Summary:\n")
        f.write(f"  Total Training Time: {total_training_time:.2f}s\n")
        f.write(f"  Avg Capacity: {np.mean(results['throughput']):.4f} bits/s/Hz\n")
        f.write(f"  Avg Latency: {np.mean(results['latency']):.2f} ms/slot\n")
        f.write(f"  Avg Energy: {np.mean(results['energy']):.2f} J\n")
        f.write(f"  Avg Forgetting: {np.mean(results['forgetting']):.4f}\n")
        f.write(f"  Avg FWT: {np.mean(fwt_values[1:]):.4f}\n")
        f.write(f"  Avg BWT: {bwt:.4f}\n")
        f.write("="*50 + "\n")
    
    return results, task_initial_performance, task_performance, fwt_values

# Main Execution
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    all_results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting"]}
    all_initial_performances = []
    all_task_performances = []
    all_fwt_values = []
    
    for run in range(NUM_RUNS):
        print(f"Run {run+1}/{NUM_RUNS}")
        results, initial_performance, task_performance, fwt_values = main(seed=run)
        for metric in all_results:
            all_results[metric].append(results[metric])
        all_initial_performances.append(initial_performance)
        all_task_performances.append(task_performance)
        all_fwt_values.append(fwt_values)
    
    # Summary of Results
    summary_file = "results_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Final Summary Across Runs\n")
        f.write("="*50 + "\n")
        for metric in all_results:
            metric_values = np.array(all_results[metric])
            mean_values = np.mean(metric_values, axis=0)
            std_values = np.std(metric_values, axis=0)
            f.write(f"{metric.capitalize()} (Mean ± Std):\n")
            for task_idx, task in enumerate(TASKS):
                f.write(f"  {task['name']}: {mean_values[task_idx]:.4f} ± {std_values[task_idx]:.4f}\n")
            f.write(f"  Overall Avg: {np.mean(mean_values):.4f} ± {np.mean(std_values):.4f}\n")
            f.write("-"*50 + "\n")
        mean_fwt = np.mean(np.array(all_fwt_values), axis=0)
        mean_bwt = np.mean([r for r in np.mean(np.array(all_task_performances), axis=0) - np.mean(np.array(all_initial_performances), axis=0) if r is not None])
        f.write("CL Metrics (Mean):\n")
        f.write(f"  Forgetting: {np.mean(np.array(all_results['forgetting']), axis=0)[-1]:.4f}\n")
        f.write(f"  FWT: {np.mean(mean_fwt[1:]):.4f}\n")
        f.write(f"  BWT: {mean_bwt:.4f}\n")
        f.write("-"*50 + "\n")
    
        # Graph 1: Throughput, Latency, Energy
        plt.figure(figsize=(12, 6))
        for metric in ["throughput", "latency", "energy"]:
            metric_values = np.array(all_results[metric])
            mean_values = np.mean(metric_values, axis=0)
            std_values = np.std(metric_values, axis=0)
            plt.errorbar(range(len(TASKS)), mean_values, yerr=std_values, label=metric.capitalize(), marker='o', capsize=5)
        plt.xlabel("Task", fontsize=10)
        plt.ylabel("Metric Value", fontsize=10)
        plt.title("Performance Across Mobility Tasks (Mean ± Std) - Throughput, Latency, Energy", fontsize=12)
        plt.grid(True)
        plt.yscale('log')
        plt.xticks(range(len(TASKS)), [task["name"] for task in TASKS], rotation=45, fontsize=8)
        plt.legend(fontsize=8)
        plt.subplots_adjust(bottom=0.35, top=0.85)
        plt.tight_layout()
        plt.savefig("performance_metrics_1.png")
        plt.close()

        # Graph 2: Forgetting
        plt.figure(figsize=(12, 6))
        forgetting_values = np.array(all_results["forgetting"])
        mean_forgetting = np.mean(forgetting_values, axis=0)
        std_forgetting = np.std(forgetting_values, axis=0)
        plt.errorbar(range(len(TASKS)), mean_forgetting, yerr=std_forgetting, label="Forgetting", marker='o', capsize=5, color='red')
        plt.xlabel("Task", fontsize=10)
        plt.ylabel("Forgetting Value", fontsize=10)
        plt.title("Performance Across Mobility Tasks (Mean ± Std) - Forgetting", fontsize=12)
        plt.grid(True)
        plt.ylim(-2.5, 0)
        plt.xticks(range(len(TASKS)), [task["name"] for task in TASKS], rotation=45, fontsize=8)
        plt.legend(fontsize=8)
        plt.subplots_adjust(bottom=0.35, top=0.85)
        plt.tight_layout()
        plt.savefig("performance_metrics_2.png")
        plt.close()

        # Graph 3: Continual Learning Metrics (Forgetting, FWT, BWT)
        plt.figure(figsize=(12, 6))
        cl_metrics = {
            "Forgetting": np.mean(np.array(all_results["forgetting"]), axis=0)[-1],
            "FWT": np.mean(np.array(all_fwt_values), axis=0)[1:].mean(),
            "BWT": np.mean([r for r in np.mean(np.array(all_task_performances), axis=0) - np.mean(np.array(all_initial_performances), axis=0) if r is not None])
        }
        metrics_values = list(cl_metrics.values())
        plt.bar(range(len(cl_metrics)), metrics_values, color=['red', 'blue', 'green'])
        plt.ylim(-3, 2)
        for i, v in enumerate(metrics_values):
            plt.text(i, v + 0.1, str(round(v, 4)), ha='center', fontsize=8)
        plt.xlabel("Metric", fontsize=10)
        plt.ylabel("Value", fontsize=10)
        plt.title("Continual Learning Metrics (Mean Across Runs)", fontsize=12)
        plt.xticks(range(len(cl_metrics)), list(cl_metrics.keys()), rotation=45, fontsize=8)
        plt.grid(True)
        plt.text(0.5, -2.7, "Continual Learning Metrics: Forgetting (red) shows accuracy loss on previous tasks, "
            "FWT (blue) indicates forward transfer to new tasks, and BWT (green) reflects backward transfer to previous tasks.",
            ha='center', transform=plt.gca().get_xaxis_transform(), fontsize=8)
        plt.subplots_adjust(bottom=0.35, top=0.85)
        plt.tight_layout()
        plt.savefig("cl_metrics.png")
        plt.close()