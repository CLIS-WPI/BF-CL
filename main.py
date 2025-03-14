import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import time

# Silence casting warnings
tf.get_logger().setLevel('ERROR')

# Initial Settings
NUM_ANTENNAS = 64  # Number of BS antennas
NUM_USERS = 10     # Number of users
FREQ = 28e9        # Frequency: 28 GHz (mmWave)
POWER = 1.0        # Normalized power
NUM_SLOTS = 1000   # Number of time slots per task
BATCH_SIZE = 128   # Batch size
LAMBDA_EWC = 10.0  # Strong regularization for EWC
NUM_EPOCHS = 10    # Number of epochs

# Define Tasks (speeds in km/h)
TASKS = [
    {"name": "Static", "speed_range": [0, 5]},
    {"name": "Pedestrian", "speed_range": [5, 10]},
    {"name": "Vehicular", "speed_range": [60, 120]},
    {"name": "Aerial", "speed_range": [20, 50]},
    {"name": "Random", "speed_range": [10, 100]},
]

# 1. Define Beamforming Model
class BeamformingModel(Model):
    def __init__(self, num_antennas, num_users):
        super(BeamformingModel, self).__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.dense1_real = layers.Dense(128, activation="relu")
        self.dense1_imag = layers.Dense(128, activation="relu")
        self.dense2_real = layers.Dense(64, activation="relu")
        self.dense2_imag = layers.Dense(64, activation="relu")
        self.output_real = layers.Dense(num_antennas)
        self.output_imag = layers.Dense(num_antennas)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        real_inputs = tf.cast(tf.math.real(inputs), tf.float32)
        imag_inputs = tf.cast(tf.math.imag(inputs), tf.float32)
        real_x = self.dense1_real(real_inputs)
        imag_x = self.dense1_imag(imag_inputs)
        real_x = self.dense2_real(real_x)
        imag_x = self.dense2_imag(imag_x)
        real_output = self.output_real(real_x)
        imag_output = self.output_imag(imag_x)
        w = tf.complex(real_output, imag_output)
        norm_squared = tf.reduce_sum(tf.abs(w)**2, axis=1, keepdims=True)
        norm = tf.cast(tf.sqrt(norm_squared), dtype=tf.complex64)
        power = tf.complex(tf.sqrt(POWER), 0.0)
        w = w / norm * power
        return w

# 2. Compute Fisher Information for EWC
def compute_fisher(model, data, num_samples=100):
    fisher = {w.name: tf.zeros_like(w) for w in model.trainable_weights}
    data_size = tf.shape(data)[0]
    for _ in range(num_samples):
        idx = tf.random.uniform(shape=[BATCH_SIZE], minval=0, maxval=data_size, dtype=tf.int32)
        x_batch = tf.gather(data, idx)
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            log_likelihood = tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.abs(logits)**2, axis=1)))
        grads = tape.gradient(log_likelihood, model.trainable_weights)
        for w, g in zip(model.trainable_weights, grads):
            if g is not None:
                fisher[w.name] += g**2 / num_samples
    return fisher

# 3. Loss Function with EWC
def ewc_loss(model, x, h, fisher, old_params, lambda_ewc):
    w = model(x)
    product = w * h
    sum_result = tf.reduce_sum(product, axis=1)
    abs_sum = tf.abs(sum_result)
    signal_power = tf.reduce_mean(abs_sum**2)
    loss = -signal_power  # Maximize signal power
    if fisher:
        ewc_penalty = 0.0
        for w in model.trainable_weights:
            w_name = w.name
            if w_name in fisher:
                ewc_penalty += tf.reduce_sum(fisher[w_name] * (w - old_params[w_name])**2)
        loss += lambda_ewc * ewc_penalty
    return loss

# 4. Channel Generation with SIONNA (Using TDL Model)
def generate_channel(task, num_slots, task_idx=0):
    # Calculate average speed for Doppler
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], NUM_USERS)
    avg_speed = np.mean(speeds) * 1000 / 3600  # Convert km/h to m/s

    # Setup antenna arrays (using PanelArray as per TR38.901)
    tx_array = sn.channel.tr38901.PanelArray(
        num_rows_per_panel=8,
        num_cols_per_panel=8,
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=FREQ
    )
    
    rx_array = sn.channel.tr38901.PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=FREQ
    )

    # Use TDL model with mobility
    channel_model = sn.channel.tr38901.TDL(
        model="A",              # TDL-A model (urban micro-like)
        delay_spread=100e-9,    # Delay spread in seconds
        carrier_frequency=FREQ,
        num_tx_ant=NUM_ANTENNAS,
        num_rx_ant=NUM_USERS,
        ut_velocity=avg_speed   # User terminal velocity in m/s
    )

    print(f"Task {task['name']}: User speeds range: {speeds.min():.2f} to {speeds.max():.2f} km/h, "
          f"Avg speed: {avg_speed:.2f} m/s")

    # Generate channels
    h = []
    start_time = time.time()
    for t in range(num_slots):
        if t % 200 == 0:
            print(f"Generating channel for slot {t}/{num_slots}")
        channel_response = channel_model(batch_size=BATCH_SIZE)
        h.append(channel_response)
    
    channel_gen_time = time.time() - start_time
    print(f"Channel generation time: {channel_gen_time:.2f} seconds")
    
    result = tf.stack(h)
    print(f"Channel data shape: {result.shape}")
    return result

# 5. Main Training and Evaluation Loop
def main():
    print("Starting Beamforming Continual Learning Training")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Available GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    model = BeamformingModel(NUM_ANTENNAS, NUM_USERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting"]}
    old_params = {}
    fisher_dict = {}
    task_performance = []
    task_channels = {}

    for task_idx, task in enumerate(TASKS):
        print(f"\n----------------------------------------------")
        print(f"Training on Task {task_idx+1}/{len(TASKS)}: {task['name']}")
        print(f"----------------------------------------------")
        
        # Generate and store channel data
        h = generate_channel(task, NUM_SLOTS, task_idx)
        task_channels[task_idx] = h
        x = h
        
        # Train model with timing
        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0
            batches = 0
            for t in range(0, NUM_SLOTS, BATCH_SIZE):
                batches += 1
                if t % (5 * BATCH_SIZE) == 0:
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, batch starting at t={t}")
                x_batch = x[t:t+BATCH_SIZE]
                h_batch = h[t:t+BATCH_SIZE]
                with tf.GradientTape() as tape:
                    loss = ewc_loss(model, x_batch, h_batch, 
                                  fisher_dict if task_idx > 0 else None, 
                                  old_params, LAMBDA_EWC)
                epoch_loss += loss.numpy()
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            avg_epoch_loss = epoch_loss / batches
            print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.6f}")
        
        # Measure latency
        training_time = time.time() - start_time
        latency = training_time / (NUM_EPOCHS * NUM_SLOTS) * 1000  # ms per slot
        
        # Initial evaluation for current task
        w = model(x)
        throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(w * h, axis=1))**2).numpy()
        task_performance.append(throughput)
        
        # Compute forgetting by re-evaluating past tasks
        forgetting = 0.0
        if task_idx > 0:
            past_performances = []
            for past_idx in range(task_idx):
                past_h = task_channels[past_idx]
                past_w = model(past_h)
                past_throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(past_w * past_h, axis=1))**2).numpy()
                forgetting_delta = task_performance[past_idx] - past_throughput
                past_performances.append(forgetting_delta)
            forgetting = np.mean(past_performances) if past_performances else 0.0
        
        # Save parameters and Fisher for EWC
        old_params = {w.name: w.numpy() for w in model.trainable_weights}
        fisher_dict = compute_fisher(model, x)
        
        # Energy estimation
        energy = training_time * 100  # Arbitrary watts
        
        # Store results
        results["throughput"].append(throughput)
        results["latency"].append(latency)
        results["energy"].append(energy)
        results["forgetting"].append(forgetting)
        
        print(f"Task {task['name']}: Throughput={throughput:.2f}, Latency={latency:.2f} ms/slot, "
              f"Energy={energy:.2f} W, Forgetting={forgetting:.2f}")
    
    # 6. Plot Results
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
    plt.savefig("mobility_tasks_results_updated.png")
    print("Saved plot to mobility_tasks_results_updated.png")
    plt.show()
    
    print("Training complete!")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()