import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import time
import gc
import os

tf.get_logger().setLevel('ERROR')

# Initial Settings
NUM_ANTENNAS = 32
NUM_USERS = 5
FREQ = 28e9
POWER = 1.0
NUM_SLOTS = 200
BATCH_SIZE = 16
LAMBDA_EWC = 1000.0
NUM_EPOCHS = 5
CHUNK_SIZE = 50
GPU_POWER_DRAW = 400
NOISE_POWER = 0.1
REPLAY_BUFFER_SIZE = 100
NUM_RUNS = 5

TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": 30e-9, "channel": "TDL", "model": "A"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": 70e-9, "channel": "TDL", "model": "A"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": 100e-9, "channel": "TDL", "model": "C"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": 50e-9, "channel": "Rayleigh"},
    {"name": "Random", "speed_range": [10, 100], "delay_spread": 90e-9, "channel": "Rayleigh"},
]

class BeamformingModel(Model):
    def __init__(self, num_antennas, num_users):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.dense1_real = layers.Dense(64, activation="relu")
        self.dense1_imag = layers.Dense(64, activation="relu")
        self.dense2_real = layers.Dense(32, activation="relu")
        self.dense2_imag = layers.Dense(32, activation="relu")
        self.output_real = layers.Dense(num_antennas * num_users)
        self.output_imag = layers.Dense(num_antennas * num_users)

    @tf.function
    def call(self, inputs):
        batch_size, slots_per_batch = tf.shape(inputs)[0], tf.shape(inputs)[1]
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
        norm = tf.cast(tf.sqrt(norm_squared), dtype=tf.complex64)
        power = tf.complex(tf.sqrt(POWER), 0.0)
        w = w / norm * power
        return w

def compute_fisher(model, data, num_samples=50):
    fisher = {w.name: tf.zeros_like(w) for w in model.trainable_weights}
    data_size = tf.shape(data)[0]
    for _ in range(num_samples):
        idx = tf.random.uniform(shape=[BATCH_SIZE], minval=0, maxval=data_size, dtype=tf.int32)
        x_batch = tf.gather(data, idx)
        with tf.GradientTape() as tape:
            logits = model(x_batch)
            log_likelihood = tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.abs(logits)**2, axis=-1)))
        grads = tape.gradient(log_likelihood, model.trainable_weights)
        for w, g in zip(model.trainable_weights, grads):
            if g is not None:
                fisher[w.name] += g**2 / num_samples
    return fisher

def ewc_loss(model, x, h, fisher, old_params, lambda_ewc):
    w = tf.cast(model(x), tf.complex64)
    h_adjusted = tf.cast(tf.transpose(h, [0, 1, 3, 2]), tf.complex64)
    product = w * h_adjusted
    sum_result = tf.reduce_sum(product, axis=-1)
    abs_sum = tf.abs(sum_result)
    signal_power = tf.reduce_mean(abs_sum**2)
    loss = -signal_power
    if fisher:
        ewc_penalty = 0.0
        for w_var in model.trainable_weights:
            w_name = w_var.name
            if w_name in fisher:
                ewc_penalty += tf.reduce_sum(fisher[w_name] * (w_var - old_params[w_name])**2)
        loss += lambda_ewc * ewc_penalty
    return loss

def debug_channel_properties(task_name, h_sample):
    """Print statistics of generated channel matrices"""
    # Convert to numpy for analysis
    h_sample_np = h_sample.numpy() if isinstance(h_sample, tf.Tensor) else h_sample
    print(f"Debug for {task_name} channel:")
    print(f"Shape: {h_sample_np.shape}")
    print(f"Mean magnitude: {np.mean(np.abs(h_sample_np))}")
    print(f"Std magnitude: {np.std(np.abs(h_sample_np))}")
    print(f"Min/Max magnitude: {np.min(np.abs(h_sample_np))}, {np.max(np.abs(h_sample_np))}")
    # Calculate condition number for the first few matrices
    condition_numbers = []
    for i in range(min(5, h_sample_np.shape[0])):
        h_mat = h_sample_np[i, 0]  # Take the first slot
        if h_mat.shape[0] == h_mat.shape[1]:  # Ensure square matrix for condition number
            condition_numbers.append(np.linalg.cond(h_mat))
    print(f"Condition number: {np.mean(condition_numbers) if condition_numbers else 'N/A'}")
    print("-" * 30)

def generate_channel(task, num_slots, task_idx=0):
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], NUM_USERS)
    avg_speed = np.mean(speeds) * 1000 / 3600
    doppler_freq = avg_speed * FREQ / 3e8
    
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
            tf.keras.backend.clear_session()
            gc.collect()
            if task["channel"] == "TDL":
                channel_response = channel_model(batch_size=BATCH_SIZE, num_time_steps=1,
                                                sampling_frequency=max(500, int(2 * doppler_freq)))
                h_t = channel_response[0]
                h_t = tf.reduce_sum(h_t, axis=5)
                h_t = tf.squeeze(h_t, axis=[3, 5])
                h_t = tf.squeeze(h_t, axis=1)
                h_t = tf.transpose(h_t, [0, 2, 1])
            else:
                h_t_tuple = channel_model(batch_size=BATCH_SIZE, num_time_steps=1)
                h_t = h_t_tuple[0]
                h_t = h_t[..., 0, :]  # حذف time step
                h_t = tf.squeeze(h_t, axis=-1)  # حذف محور اضافی
                h_t = tf.reduce_mean(h_t, axis=[2, 4])  # میانگین‌گیری روی محورهای اضافی
                h_t = tf.transpose(h_t, [0, 2, 1])
                # نرمال‌سازی مقادیر Rayleigh
                h_t_magnitude = tf.abs(h_t)
                target_mean_magnitude = 0.89  # میانگین TDL
                current_mean_magnitude = tf.reduce_mean(h_t_magnitude)
                scaling_factor = target_mean_magnitude / (current_mean_magnitude + 1e-10)  # برای جلوگیری از تقسیم بر صفر
                # تبدیل scaling_factor به complex64
                scaling_factor = tf.cast(scaling_factor, tf.complex64)
                h_t = h_t * scaling_factor
            h_chunk.append(h_t)
        h_chunks.append(tf.stack(h_chunk))
    h = tf.concat(h_chunks, axis=0)
    print(f"Channel for {task['name']} generated in {time.time() - start_time:.2f}s")
    debug_channel_properties(task["name"], h[:5])
    return h

def train_step(model, x_batch, h_batch, optimizer, fisher_dict, old_params, task_idx, replay_buffer=None):
    with tf.GradientTape() as tape:
        loss = ewc_loss(model, x_batch, h_batch, fisher_dict if task_idx > 0 else None,
                       old_params, LAMBDA_EWC)
        if replay_buffer and len(replay_buffer[0]) > 0:
            replay_idx = tf.random.uniform(shape=[min(BATCH_SIZE, len(replay_buffer[0]))], minval=0, maxval=len(replay_buffer[0]), dtype=tf.int32)
            replay_x = tf.cast(tf.gather(replay_buffer[0], replay_idx), tf.complex64)
            replay_h = tf.cast(tf.gather(replay_buffer[1], replay_idx), tf.complex64)
            replay_loss = ewc_loss(model, replay_x, replay_h, fisher_dict if task_idx > 0 else None,
                                   old_params, LAMBDA_EWC)
            loss += replay_loss
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

def main(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting"]}
    task_performance = []
    task_channels = {}
    old_params = {}
    fisher_dict = {}
    replay_buffer = [[], []]
    task_losses = {task["name"]: [] for task in TASKS}
    
    strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
    model = BeamformingModel(NUM_ANTENNAS, NUM_USERS)
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
    
    for task_idx, task in enumerate(TASKS):
        print(f"Starting Task {task['name']} ({task_idx+1}/{len(TASKS)})")
        h = generate_channel(task, NUM_SLOTS, task_idx)
        task_channels[task_idx] = h
        x = h
        dataset = tf.data.Dataset.from_tensor_slices((x, h)).batch(BATCH_SIZE)
        
        if task_idx == 0:
            replay_buffer[0] = x[:min(BATCH_SIZE, REPLAY_BUFFER_SIZE)].numpy()
            replay_buffer[1] = h[:min(BATCH_SIZE, REPLAY_BUFFER_SIZE)].numpy()
        else:
            if len(replay_buffer[0]) < REPLAY_BUFFER_SIZE:
                new_x = x[:min(BATCH_SIZE, REPLAY_BUFFER_SIZE - len(replay_buffer[0]))].numpy()
                new_h = h[:min(BATCH_SIZE, REPLAY_BUFFER_SIZE - len(replay_buffer[1]))].numpy()
                replay_buffer[0] = np.concatenate([replay_buffer[0], new_x], axis=0) if replay_buffer[0].size else new_x
                replay_buffer[1] = np.concatenate([replay_buffer[1], new_h], axis=0) if replay_buffer[1].size else new_h
            else:
                idx = task_idx % REPLAY_BUFFER_SIZE
                replay_buffer[0][idx:idx+1] = x[:1].numpy()
                replay_buffer[1][idx:idx+1] = h[:1].numpy()
        
        replay_buffer[0] = np.array(replay_buffer[0], dtype=np.complex64)
        replay_buffer[1] = np.array(replay_buffer[1], dtype=np.complex64)
        
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
                                                        fisher_dict, old_params, task_idx, replay_buffer))
                    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
                    epoch_loss += loss
                    batches += 1
                avg_epoch_loss = epoch_loss / batches
                task_losses[task['name']].append(avg_epoch_loss.numpy())
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
        
        old_params = {w.name: w.numpy() for w in model.trainable_weights}
        fisher_dict = compute_fisher(model, x)
        
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
            f.write(f"  Losses: {[float(loss) for loss in task_losses[task['name']]]}\n")
            f.write(f"  Total Task Time: {task_time:.2f}s\n")
            f.write("-"*50 + "\n")
        model.save(os.path.join(save_dir, f"model_task_{task['name']}_seed_{seed}"))
        print(f"Task {task['name']} completed")
    
    total_training_time = time.time() - total_start_time
    with open(results_file, 'a') as f:
        f.write("Summary:\n")
        f.write(f"  Total Training Time: {total_training_time:.2f}s\n")
        f.write(f"  Avg Capacity: {np.mean(results['throughput']):.4f} bits/s/Hz\n")
        f.write(f"  Avg Latency: {np.mean(results['latency']):.2f} ms/slot\n")
        f.write(f"  Avg Energy: {np.mean(results['energy']):.2f} J\n")
        f.write(f"  Avg Forgetting: {np.mean(results['forgetting']):.4f}\n")
        f.write("="*50 + "\n")
    
    plt.figure(figsize=(12, 8))
    for metric in results:
        plt.plot(results[metric], label=metric, marker='o')
    plt.legend()
    plt.xlabel("Task")
    plt.ylabel("Metric Value")
    plt.title("Performance Across Mobility Tasks")
    plt.grid(True)
    plt.yscale('log')
    plt.xticks(range(len(TASKS)), [task["name"] for task in TASKS], rotation=45)
    plt.tight_layout()
    plt.savefig(f"mobility_tasks_results_seed_{seed}.png")
    plt.close()
    print(f"Training completed in {total_training_time:.2f}s")
    
    return results

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    all_results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting"]}
    
    for run in range(NUM_RUNS):
        print(f"Run {run+1}/{NUM_RUNS}")
        results = main(seed=run)
        for metric in all_results:
            all_results[metric].append(results[metric])
    
    # محاسبه میانگین و انحراف معیار
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
    
    # گراف اول: Throughput، Latency، Energy
    plt.figure(figsize=(12, 6))
    for metric in ["throughput", "latency", "energy"]:
        metric_values = np.array(all_results[metric])
        mean_values = np.mean(metric_values, axis=0)
        std_values = np.std(metric_values, axis=0)
        plt.errorbar(range(len(TASKS)), mean_values, yerr=std_values, label=metric.capitalize(), marker='o', capsize=5)

    plt.xlabel("Task")
    plt.ylabel("Metric Value")
    plt.title("Performance Across Mobility Tasks (Mean ± Std) - Throughput, Latency, Energy")
    plt.grid(True)
    plt.yscale('log')  # مقیاس لگاریتمی برای معیارهای بزرگ
    plt.xticks(range(len(TASKS)), [task["name"] for task in TASKS], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("performance_metrics_1.png")
    plt.close()

    # گراف دوم: Forgetting
    plt.figure(figsize=(12, 6))
    forgetting_values = np.array(all_results["forgetting"])
    mean_forgetting = np.mean(forgetting_values, axis=0)
    std_forgetting = np.std(forgetting_values, axis=0)
    plt.errorbar(range(len(TASKS)), mean_forgetting, yerr=std_forgetting, label="Forgetting", marker='o', capsize=5, color='red')

    plt.xlabel("Task")
    plt.ylabel("Forgetting Value")
    plt.title("Performance Across Mobility Tasks (Mean ± Std) - Forgetting")
    plt.grid(True)
    plt.ylim(-2.5, 0)  # محدوده مناسب برای Forgetting
    plt.xticks(range(len(TASKS)), [task["name"] for task in TASKS], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("performance_metrics_2.png")
    plt.close()

    print("Final summary and plots generated.")