import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

# Silence casting warnings
tf.get_logger().setLevel('ERROR')

# Initial Settings
NUM_ANTENNAS = 64  # Number of BS antennas
NUM_USERS = 10     # Number of users
FREQ = 28e9        # Frequency: 28 GHz (mmWave)
POWER = 1.0        # Normalized power
NUM_SLOTS = 1000   # Number of time slots per task
BATCH_SIZE = 128   # Batch size
LAMBDA_EWC = 0.1   # EWC regularization coefficient

# Define Tasks (speeds in km/h)
TASKS = [
    {"name": "Static", "speed_range": [0, 5]},
    {"name": "Pedestrian", "speed_range": [5, 10]},
    {"name": "Vehicular", "speed_range": [60, 120]},
    {"name": "Aerial", "speed_range": [20, 50]},
    {"name": "Random", "speed_range": [10, 100]},
]

# 1. Define Beamforming Model - FIXED VERSION
class BeamformingModel(Model):
    def __init__(self, num_antennas, num_users):
        super(BeamformingModel, self).__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        
        # Define separate layers for real and imaginary parts
        self.dense1_real = layers.Dense(128, activation="relu")
        self.dense1_imag = layers.Dense(128, activation="relu")
        
        self.dense2_real = layers.Dense(64, activation="relu")
        self.dense2_imag = layers.Dense(64, activation="relu")
        
        self.output_real = layers.Dense(num_antennas)
        self.output_imag = layers.Dense(num_antennas)

    def call(self, inputs):
        # Get batch size
        batch_size = tf.shape(inputs)[0]
        
        # Extract real and imaginary parts and explicitly cast to float32
        real_inputs = tf.cast(tf.math.real(inputs), tf.float32)
        imag_inputs = tf.cast(tf.math.imag(inputs), tf.float32)
        
        # Process real and imaginary parts separately
        real_x = self.dense1_real(real_inputs)
        imag_x = self.dense1_imag(imag_inputs)
        
        real_x = self.dense2_real(real_x)
        imag_x = self.dense2_imag(imag_x)
        
        real_output = self.output_real(real_x)
        imag_output = self.output_imag(imag_x)
        
        # Combine into complex output
        w = tf.complex(real_output, imag_output)
        
        # FIXED: Normalize beamforming weights with proper complex casting
        norm_squared = tf.reduce_sum(tf.abs(w)**2, axis=1, keepdims=True)
        norm = tf.cast(tf.sqrt(norm_squared), dtype=tf.complex64)
        
        power = tf.complex(tf.sqrt(POWER), 0.0)
        w = w / norm * power
        
        return w

# 2. Compute Fisher Information for EWC - FIXED VERSION
def compute_fisher(model, data, num_samples=100):
    # Initialize Fisher information dictionary
    fisher = {w.name: tf.zeros_like(w) for w in model.trainable_weights}
    
    # Get data shape information
    data_size = tf.shape(data)[0]
    
    for _ in range(num_samples):
        # Generate random indices using TensorFlow operations 
        # instead of numpy to maintain compatibility
        idx = tf.random.uniform(
            shape=[BATCH_SIZE], 
            minval=0, 
            maxval=data_size, 
            dtype=tf.int32
        )
        
        # Gather data using TensorFlow operations
        x_batch = tf.gather(data, idx)
        
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            log_likelihood = tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.abs(logits)**2, axis=1)))
        
        grads = tape.gradient(log_likelihood, model.trainable_weights)
        
        for w, g in zip(model.trainable_weights, grads):
            if g is not None:
                fisher[w.name] += g**2 / num_samples
                
    return fisher

# 3. Loss Function with EWC - FIXED to avoid casting warnings
def ewc_loss(model, x, h, fisher, old_params, lambda_ewc):
    w = model(x)
    
    # First multiply the beamforming weights with the channel
    product = w * h
    
    # Sum the product along the appropriate axis
    sum_result = tf.reduce_sum(product, axis=1)
    
    # Get the absolute value (magnitude) of complex numbers
    # This is where the complex-to-float conversion happens,
    # but we acknowledge this is intentional behavior
    abs_sum = tf.abs(sum_result)
    
    # Square and calculate the mean
    signal_power = tf.reduce_mean(abs_sum**2)
    
    # Maximize signal power
    loss = -signal_power
    
    # Apply EWC regularization if provided
    if fisher:
        ewc_penalty = 0.0
        for w in model.trainable_weights:
            w_name = w.name
            if w_name in fisher:
                ewc_penalty += tf.reduce_sum(fisher[w_name] * (w - old_params[w_name])**2)
        loss += lambda_ewc * ewc_penalty
    
    return loss

# 4. Channel Generation with SIONNA - FIXED VERSION WITH UNIQUE SCENE OBJECTS
def generate_channel(task, num_slots, task_idx=0):
    # Create scene with a unique name for each task to avoid conflicts
    scene = sn.rt.Scene()
    task_suffix = f"_{task_idx}"  # Add task index to make names unique

    # Define transmitter and receivers with unique names
    tx = sn.rt.Transmitter(name=f"tx{task_suffix}", position=[0, 0, 10])
    rx_positions = np.random.uniform(-100, 100, (NUM_USERS, 3))
    rx_positions[:, 2] = 1.5  # User height
    rxs = [sn.rt.Receiver(name=f"rx-{i}{task_suffix}", position=rx_positions[i]) for i in range(NUM_USERS)]

    # Configure antenna arrays with spacing and pattern
    wavelength = 3e8 / FREQ  # Wavelength = c/f
    spacing = wavelength / 2  # Half-wavelength spacing
    scene.tx_array = sn.rt.PlanarArray(
        num_rows=8,
        num_cols=8,
        vertical_spacing=spacing,
        horizontal_spacing=spacing,
        pattern="iso",  # Isotropic pattern
        polarization="V"
    )
    scene.rx_array = sn.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=spacing,
        horizontal_spacing=spacing,
        pattern="iso",
        polarization="V"
    )

    # Add devices to scene
    scene.add(tx)
    for rx in rxs:
        scene.add(rx)

    # Use FlatFadingChannel with correct parameters
    channel_model = sn.channel.FlatFadingChannel(
        num_tx_ant=NUM_ANTENNAS,  # Number of transmit antennas
        num_rx_ant=NUM_USERS,     # Number of users (receivers)
        add_awgn=True,            # Add noise
        return_channel=True       # Return channel coefficients
    )

    # Generate channel with mobility
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], NUM_USERS)
    print(f"User speeds range: {speeds.min():.2f} to {speeds.max():.2f} km/h")
    
    h = []
    for t in range(num_slots):
        if t % 200 == 0:
            print(f"Generating channel for slot {t}/{num_slots}")
            
        # Update receiver positions to simulate mobility
        for i, rx in enumerate(rxs):
            rx.position += [speeds[i] * 0.001 * np.cos(t * 0.1), speeds[i] * 0.001 * np.sin(t * 0.1), 0]
            
        # Generate channel
        x = tf.ones([BATCH_SIZE, NUM_ANTENNAS], dtype=tf.complex64)
        no = tf.ones([BATCH_SIZE, NUM_USERS], dtype=tf.complex64)
        
        # FIXED: FlatFadingChannel returns a tuple, extract the channel matrix (second element)
        channel_output = channel_model([x, no])
        channel_matrix = channel_output[1]  # Get the channel matrix from the tuple
        
        h.append(channel_matrix)
    
    result = tf.stack(h)
    print(f"Channel data shape: {result.shape}")
    return result

# 5. Main Training and Evaluation Loop
def main():
    print("Starting Beamforming Continual Learning training")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Available GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    model = BeamformingModel(NUM_ANTENNAS, NUM_USERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting"]}
    old_params = {}
    fisher_dict = {}
    task_performance = []

    for task_idx, task in enumerate(TASKS):
        print(f"\n----------------------------------------------")
        print(f"Training on Task {task_idx+1}/{len(TASKS)}: {task['name']}")
        print(f"----------------------------------------------")
        
        # Generate channel data - now passing the task_idx to make scene objects unique
        h = generate_channel(task, NUM_SLOTS, task_idx)
        x = h  # Use channel data as input
        
        # Train model
        for epoch in range(10):
            epoch_loss = 0
            batches = 0
            
            for t in range(0, NUM_SLOTS, BATCH_SIZE):
                batches += 1
                if t % (5 * BATCH_SIZE) == 0:
                    print(f"Epoch {epoch+1}/10, batch starting at t={t}")
                    
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
        
        # Save parameters and Fisher for EWC
        old_params = {w.name: w.numpy() for w in model.trainable_weights}
        fisher_dict = compute_fisher(model, x)
        
        # Evaluation
        w = model(x)
        throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(w * h, axis=1))**2).numpy()
        latency = 0.01  # Placeholder (replace with real timing)
        energy = 0.5 * NUM_SLOTS * NUM_ANTENNAS  # Placeholder
        task_performance.append(throughput)
        
        # Compute Forgetting
        if task_idx > 0:
            forgetting = np.mean([task_performance[i] - task_performance[-1] for i in range(task_idx)])
            results["forgetting"].append(forgetting)
        else:
            results["forgetting"].append(0.0)
            
        results["throughput"].append(throughput)
        results["latency"].append(latency)
        results["energy"].append(energy)
        
        print(f"Task {task['name']}: Throughput={throughput:.2f}, Forgetting={results['forgetting'][-1]:.2f}")
    
    # 6. Plot Results
    plt.figure(figsize=(10, 6))
    for metric in results:
        plt.plot(results[metric], label=metric)
    plt.legend()
    plt.xlabel("Task")
    plt.ylabel("Metric Value")
    plt.title("Performance Across Mobility Tasks")
    plt.grid(True)
    plt.xticks(range(len(TASKS)), [task["name"] for task in TASKS], rotation=45)
    plt.tight_layout()
    plt.savefig("mobility_tasks_results.png")
    print("Saved plot to mobility_tasks_results.png")
    plt.show()
    
    print("Training complete!")

if __name__ == "__main__":
    # Disable eager execution warnings
    tf.get_logger().setLevel('ERROR')
    main()