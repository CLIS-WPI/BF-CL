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
LAMBDA_EWC = 10000.0
NUM_EPOCHS = 5
CHUNK_SIZE = 50

TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": 30e-9, "channel": "TDL", "model": "A"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": 50e-9, "channel": "Rayleigh"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": 100e-9, "channel": "TDL", "model": "C"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": 70e-9, "channel": "TDL", "model": "A"},
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
        self.output_real = layers.Dense(num_antennas * num_users)  # Output for all users
        self.output_imag = layers.Dense(num_antennas * num_users)

    @tf.function
    def call(self, inputs):
        batch_size, slots_per_batch = tf.shape(inputs)[0], tf.shape(inputs)[1]
        # inputs: (batch, slots_per_batch, num_tx, num_rx) = (16, 16, 32, 5)
        inputs_flat = tf.reshape(inputs, [batch_size * slots_per_batch, self.num_antennas * self.num_users])  # (256, 160)
        real_inputs = tf.cast(tf.math.real(inputs_flat), tf.float32)
        imag_inputs = tf.cast(tf.math.imag(inputs_flat), tf.float32)
        real_x = self.dense1_real(real_inputs)
        imag_x = self.dense1_imag(imag_inputs)
        real_x = self.dense2_real(real_x)
        imag_x = self.dense2_imag(imag_x)
        real_output = self.output_real(real_x)  # (256, 160)
        imag_output = self.output_imag(imag_x)
        w = tf.complex(real_output, imag_output)
        w = tf.reshape(w, [batch_size, slots_per_batch, self.num_users, self.num_antennas])  # (16, 16, 5, 32)
        norm_squared = tf.reduce_sum(tf.abs(w)**2, axis=-1, keepdims=True)
        norm = tf.cast(tf.sqrt(norm_squared), dtype=tf.complex64)
        power = tf.complex(tf.sqrt(POWER), 0.0)
        w = w / norm * power
        print(f"Model output w shape: {w.shape}")
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
    w = model(x)
    h_adjusted = tf.transpose(h, [0, 1, 3, 2])  # (16, 16, 5, 32)
    print(f"ewc_loss: w shape: {w.shape}, h_adjusted shape: {h_adjusted.shape}")
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
        channel_info = f"TDL-{task['model']}"
        sampling_frequency = max(500, int(2 * doppler_freq))
    else:
        print(f"Initializing RayleighBlockFading with num_rx={NUM_USERS}, num_rx_ant={NUM_USERS}, "
              f"num_tx={NUM_ANTENNAS}, num_tx_ant={NUM_ANTENNAS}")
        channel_model = sn.channel.RayleighBlockFading(
            num_rx=NUM_USERS,
            num_rx_ant=NUM_USERS,
            num_tx=NUM_ANTENNAS,
            num_tx_ant=NUM_ANTENNAS,
            dtype=tf.complex64
        )
        channel_info = "Rayleigh"
        sampling_frequency = 500

    print(f"Task {task['name']}: Speed range: {speeds.min():.2f} to {speeds.max():.2f} km/h, "
          f"Delay spread: {task['delay_spread']:.2e} s, Channel: {channel_info}, "
          f"Doppler freq: {doppler_freq:.2f} Hz")
    
    h_chunks = []
    start_time = time.time()
    
    for chunk_start in range(0, num_slots, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, num_slots)
        print(f"Processing chunk {chunk_start}-{chunk_end}/{num_slots}")
        h_chunk = []
        for t in range(chunk_start, chunk_end):
            if t % 50 == 0 or t == chunk_start:
                print(f"Generating channel slot {t}/{num_slots}")
            tf.keras.backend.clear_session()
            gc.collect()
            if task["channel"] == "TDL":
                channel_response = channel_model(batch_size=BATCH_SIZE, num_time_steps=1,
                                                sampling_frequency=sampling_frequency)
                h_t = channel_response[0]
                print(f"TDL raw h_t shape: {h_t.shape}")
                h_t = tf.reduce_sum(h_t, axis=5)
                print(f"TDL after sum h_t shape: {h_t.shape}")
                h_t = tf.squeeze(h_t, axis=[3, 5])
                print(f"TDL after squeeze h_t shape: {h_t.shape}")
                h_t = tf.squeeze(h_t, axis=1)
                h_t = tf.transpose(h_t, [0, 2, 1])
                print(f"TDL final h_t shape: {h_t.shape}")
            else:
                h_t_tuple = channel_model(batch_size=BATCH_SIZE, num_time_steps=1)
                h_t = h_t_tuple[0]
                h_t = h_t[..., 0, :]
                h_t = tf.squeeze(h_t, axis=-1)
                h_t = tf.reduce_mean(h_t, axis=[2, 4])
                h_t = tf.transpose(h_t, [0, 2, 1])
            h_chunk.append(h_t)
        h_chunks.append(tf.stack(h_chunk))
    h = tf.concat(h_chunks, axis=0)
    print(f"Channel generation time: {time.time() - start_time:.2f} s")
    print(f"Final channel shape: {h.shape}, dtype: {h.dtype}")
    return h

def train_step(model, x_batch, h_batch, optimizer, fisher_dict, old_params, task_idx):
    with tf.GradientTape() as tape:
        loss = ewc_loss(model, x_batch, h_batch, fisher_dict if task_idx > 0 else None,
                       old_params, LAMBDA_EWC)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

def main():
    print("Starting Beamforming Continual Learning Training")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Available GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(f"SIONNA version: {sn.__version__}")
    print("\nTraining Configuration:")
    print(f"Number of Antennas: {NUM_ANTENNAS}")
    print(f"Number of Users: {NUM_USERS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Number of Slots: {NUM_SLOTS}")
    print(f"EWC Lambda: {LAMBDA_EWC}")
    
    results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting"]}
    task_performance = []
    task_channels = {}
    old_params = {}
    fisher_dict = {}
    
    strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    model = BeamformingModel(NUM_ANTENNAS, NUM_USERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    total_start_time = time.time()
    for task_idx, task in enumerate(TASKS):
        print(f"\n{'='*80}")
        print(f"Training on Task {task_idx+1}/{len(TASKS)}: {task['name']}")
        print(f"{'='*80}")
        
        h = generate_channel(task, NUM_SLOTS, task_idx)
        task_channels[task_idx] = h
        x = h
        dataset = tf.data.Dataset.from_tensor_slices((x, h)).batch(BATCH_SIZE)
        
        with strategy.scope():
            dist_dataset = strategy.experimental_distribute_dataset(dataset)
            num_samples = x.shape[0]
            num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Number of batches per epoch: {num_batches}")
            
            task_start_time = time.time()
            for epoch in range(NUM_EPOCHS):
                print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
                print("-" * 40)
                epoch_start_time = time.time()
                epoch_loss = 0
                batches = 0
                progress_interval = max(1, num_batches // 20)
                for x_batch, h_batch in dist_dataset:
                    batch_start_time = time.time()
                    loss = strategy.run(train_step, args=(model, x_batch, h_batch, optimizer,
                                                        fisher_dict, old_params, task_idx))
                    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
                    epoch_loss += loss
                    batches += 1
                    if batches % progress_interval == 0:
                        progress = (batches / num_batches) * 100
                        batch_time = time.time() - batch_start_time
                        print(f"Progress: {progress:3.1f}% [{batches}/{num_batches}] "
                              f"| Batch Loss: {loss:.6f} | Batch Time: {batch_time:.3f}s")
                avg_epoch_loss = epoch_loss / batches
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1} Summary: Avg Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s")
        
        task_time = time.time() - task_start_time
        latency = task_time / (NUM_EPOCHS * NUM_SLOTS) * 1000
        w = model(x)
        throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(w * tf.transpose(h, [0, 1, 3, 2]), axis=-1))**2).numpy()
        task_performance.append(throughput)
        
        forgetting = 0.0
        if task_idx > 0:
            past_performances = []
            print("\nEvaluating forgetting on previous tasks:")
            for past_idx in range(task_idx):
                past_h = task_channels[past_idx]
                past_w = model(past_h)
                past_throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(past_w * tf.transpose(past_h, [0, 1, 3, 2]), axis=-1))**2).numpy()
                forgetting_delta = task_performance[past_idx] - past_throughput
                past_performances.append(forgetting_delta)
                print(f"Task {TASKS[past_idx]['name']} forgetting: {forgetting_delta:.4f}")
            forgetting = np.mean(past_performances) if past_performances else 0.0
        
        print("\nComputing Fisher information...")
        old_params = {w.name: w.numpy() for w in model.trainable_weights}
        fisher_dict = compute_fisher(model, x)
        
        energy = task_time * 100
        results["throughput"].append(throughput)
        results["latency"].append(latency)
        results["energy"].append(energy)
        results["forgetting"].append(forgetting)
        
        print(f"\nTask {task['name']} Final Results:")
        print("-" * 40)
        print(f"Throughput: {throughput:.2f}")
        print(f"Latency: {latency:.2f} ms/slot")
        print(f"Energy: {energy:.2f} W")
        print(f"Forgetting: {forgetting:.4f}")
        print(f"Total Task Time: {task_time:.2f}s")
        model.save(os.path.join(save_dir, f"model_task_{task['name']}"))
        print(f"Model saved for task {task['name']}")
    
    total_training_time = time.time() - total_start_time
    print(f"\nTotal Training Time: {total_training_time:.2f}s")
    print("\nGenerating results plot...")
    plt.figure(figsize=(12, 8))
    for metric in results:
        plt.plot(results[metric], label=metric, marker='o')
    plt.legend()
    plt.xlabel("Task")
    plt.ylabel("Metric Value")
    plt.title("Performance Across Mobility Tasks")
    plt.grid(True)
    plt.xticks(range(len(TASKS)), [task["name"] for task in TASKS], rotation=45)
    plt.tight_layout()
    plt.savefig("mobility_tasks_results_final.png")
    plt.show()
    print("\nTraining complete!")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()