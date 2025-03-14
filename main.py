import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

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

# 1. Define Beamforming Model
class BeamformingModel(Model):
    def __init__(self, num_antennas, num_users):
        super(BeamformingModel, self).__init__()
        self.dense1 = layers.Dense(128, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.output_layer = layers.Dense(num_antennas * 2)  # Real + Imag

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.output_layer(x)
        # Convert to complex vector
        real = x[:, :NUM_ANTENNAS]
        imag = x[:, NUM_ANTENNAS:]
        w = tf.complex(real, imag)
        w = w / tf.sqrt(tf.reduce_sum(tf.abs(w)**2, axis=1, keepdims=True)) * tf.sqrt(POWER)
        return w

# 2. Compute Fisher Information for EWC
def compute_fisher(model, data, num_samples=100):
    fisher = {w.name: tf.zeros_like(w) for w in model.trainable_weights}
    for _ in range(num_samples):
        idx = np.random.randint(0, len(data), BATCH_SIZE)
        x_batch = data[idx]
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
    signal_power = tf.reduce_mean(tf.abs(tf.reduce_sum(w * h, axis=1))**2)
    loss = -signal_power  # Maximize signal power
    if fisher:
        ewc_penalty = 0.0
        for w in model.trainable_weights:
            w_name = w.name
            if w_name in fisher:
                ewc_penalty += tf.reduce_sum(fisher[w_name] * (w - old_params[w_name])**2)
        loss += lambda_ewc * ewc_penalty
    return loss

# 4. Channel Generation with SIONNA (Fixed)
def generate_channel(task, num_slots):
    # Create scene
    scene = sn.rt.Scene()

    # Define transmitter and receivers
    tx = sn.rt.Transmitter(name="tx", position=[0, 0, 10])
    rx_positions = np.random.uniform(-100, 100, (NUM_USERS, 3))
    rx_positions[:, 2] = 1.5  # User height
    rxs = [sn.rt.Receiver(name=f"rx-{i}", position=rx_positions[i]) for i in range(NUM_USERS)]

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
    # With this corrected version:
    channel_model = sn.channel.FlatFadingChannel(
        num_tx_ant=NUM_ANTENNAS,  # Number of transmit antennas
        num_rx_ant=NUM_USERS,     # Number of users (receivers)
        add_awgn=True,            # Add noise
        return_channel=True       # Return channel coefficients
    )

    # Generate channel with mobility
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], NUM_USERS)
    h = []
    for t in range(num_slots):
        # Update receiver positions to simulate mobility
        for i, rx in enumerate(rxs):
            rx.position += [speeds[i] * 0.001 * np.cos(t * 0.1), speeds[i] * 0.001 * np.sin(t * 0.1), 0]
        # Generate channel
        x = tf.ones([BATCH_SIZE, NUM_ANTENNAS], dtype=tf.complex64)
        no = tf.ones([BATCH_SIZE, NUM_USERS], dtype=tf.complex64)
        h_t = channel_model([x, no])
        h.append(h_t)  # [batch_size, num_antennas, num_users]
    return tf.stack(h)

# 5. Main Training and Evaluation Loop
model = BeamformingModel(NUM_ANTENNAS, NUM_USERS)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting"]}
old_params = {}
fisher_dict = {}
task_performance = []

for task_idx, task in enumerate(TASKS):
    print(f"Training on Task: {task['name']}")
    
    # Generate channel data
    h = generate_channel(task, NUM_SLOTS)
    x = tf.abs(h)  # Input to model (magnitude of CSI)

    # Train model
    for epoch in range(10):
        for t in range(0, NUM_SLOTS, BATCH_SIZE):
            x_batch = x[t:t+BATCH_SIZE]
            h_batch = h[t:t+BATCH_SIZE]
            with tf.GradientTape() as tape:
                loss = ewc_loss(model, x_batch, h_batch, fisher_dict if task_idx > 0 else None, old_params, LAMBDA_EWC)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

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
for metric in results:
    plt.plot(results[metric], label=metric)
plt.legend()
plt.xlabel("Task")
plt.ylabel("Metric Value")
plt.title("Performance Across Mobility Tasks")
plt.show()