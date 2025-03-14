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

# 1. Define Beamforming Model with extensive debug prints
class BeamformingModel(Model):
    def __init__(self, num_antennas, num_users):
        super(BeamformingModel, self).__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        
        # Define separate layers for real and imaginary parts
        # Define all layers with tf.keras.layers.Dense to have full control
        self.dense1_real = tf.keras.layers.Dense(128, activation="relu", name="dense1_real")
        self.dense1_imag = tf.keras.layers.Dense(128, activation="relu", name="dense1_imag")
        
        self.dense2_real = tf.keras.layers.Dense(64, activation="relu", name="dense2_real")
        self.dense2_imag = tf.keras.layers.Dense(64, activation="relu", name="dense2_imag")
        
        self.output_real = tf.keras.layers.Dense(num_antennas, name="output_real")
        self.output_imag = tf.keras.layers.Dense(num_antennas, name="output_imag")
        
        print("DEBUG: Model initialized with layers:")
        print(f"  dense1_real: {self.dense1_real}")
        print(f"  dense1_imag: {self.dense1_imag}")
        
    def call(self, inputs):
        # Debug input shape and dtype
        print("\nDEBUG CALL START ===========================")
        print(f"Input shape: {inputs.shape}, dtype: {inputs.dtype}")
        
        # Get batch size
        batch_size = tf.shape(inputs)[0]
        print(f"Batch size: {batch_size}")
        
        # Extract real and imaginary parts and explicitly cast to float32
        # Adding debug prints
        print(f"Before extraction - inputs dtype: {inputs.dtype}")
        
        # Use tf.debugging.check_numerics to verify inputs
        try:
            tf.debugging.check_numerics(inputs, "Input tensor contains invalid values")
            print("Input tensor passed numeric check")
        except Exception as e:
            print(f"Input tensor check failed: {e}")
        
        real_inputs = tf.cast(tf.math.real(inputs), tf.float32)
        imag_inputs = tf.cast(tf.math.imag(inputs), tf.float32)
        
        print(f"After extraction - real_inputs: {real_inputs.dtype}, imag_inputs: {imag_inputs.dtype}")
        
        # Check layer dtypes
        print(f"dense1_real input dtype policy: {self.dense1_real.dtype_policy}")
        print(f"dense1_imag input dtype policy: {self.dense1_imag.dtype_policy}")
        
        # Process real and imaginary parts separately
        # Adding debug before each layer call
        print("Before dense1_real - real_inputs shape:", real_inputs.shape)
        real_x = self.dense1_real(real_inputs)
        print(f"After dense1_real - real_x: {real_x.dtype}, shape: {real_x.shape}")
        
        print("Before dense1_imag - imag_inputs shape:", imag_inputs.shape)
        imag_x = self.dense1_imag(imag_inputs)
        print(f"After dense1_imag - imag_x: {imag_x.dtype}, shape: {imag_x.shape}")
        
        real_x = self.dense2_real(real_x)
        print(f"After dense2_real - real_x: {real_x.dtype}, shape: {real_x.shape}")
        
        imag_x = self.dense2_imag(imag_x)
        print(f"After dense2_imag - imag_x: {imag_x.dtype}, shape: {imag_x.shape}")
        
        real_output = self.output_real(real_x)
        print(f"After output_real - real_output: {real_output.dtype}, shape: {real_output.shape}")
        
        imag_output = self.output_imag(imag_x)
        print(f"After output_imag - imag_output: {imag_output.dtype}, shape: {imag_output.shape}")
        
        # Combine into complex output
        print("Before combining to complex - checking output types:")
        print(f"  real_output: {real_output.dtype}")
        print(f"  imag_output: {imag_output.dtype}")
        
        w = tf.complex(real_output, imag_output)
        print(f"After combining to complex - w: {w.dtype}, shape: {w.shape}")
        
        # Normalize beamforming weights
        norm = tf.sqrt(tf.reduce_sum(tf.abs(w)**2, axis=1, keepdims=True))
        print(f"Norm computed: {norm.dtype}, shape: {norm.shape}")
        
        power = tf.complex(tf.sqrt(POWER), 0.0)
        print(f"Power: {power.dtype}")
        
        w = w / norm * power
        print(f"Final output - w: {w.dtype}, shape: {w.shape}")
        print("DEBUG CALL END ===========================\n")
        
        return w

# 2. Compute Fisher Information for EWC with debug info
def compute_fisher(model, data, num_samples=100):
    print(f"\nDEBUG: compute_fisher - data shape: {data.shape}, dtype: {data.dtype}")
    print(f"DEBUG: compute_fisher - num_samples: {num_samples}")
    
    fisher = {w.name: tf.zeros_like(w) for w in model.trainable_weights}
    print(f"DEBUG: Initialized fisher dict with {len(fisher)} keys")
    
    for s in range(num_samples):
        if s == 0:  # Only print for first sample to avoid spam
            print(f"DEBUG: compute_fisher - processing sample {s+1}/{num_samples}")
            
        idx = np.random.randint(0, len(data), BATCH_SIZE)
        x_batch = data[idx]
        
        if s == 0:
            print(f"DEBUG: x_batch shape: {x_batch.shape}, dtype: {x_batch.dtype}")
        
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            log_likelihood = tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.abs(logits)**2, axis=1)))
            
            if s == 0:
                print(f"DEBUG: logits shape: {logits.shape}, dtype: {logits.dtype}")
                print(f"DEBUG: log_likelihood: {log_likelihood.numpy()}")
        
        grads = tape.gradient(log_likelihood, model.trainable_weights)
        
        if s == 0:
            print(f"DEBUG: Got {len(grads)} gradients")
            
        for w, g in zip(model.trainable_weights, grads):
            if g is not None:
                fisher[w.name] += g**2 / num_samples
                
    print(f"DEBUG: Finished compute_fisher with {len(fisher)} parameters\n")
    return fisher

# 3. Loss Function with EWC with debug
def ewc_loss(model, x, h, fisher, old_params, lambda_ewc):
    print(f"\nDEBUG: ewc_loss - x shape: {x.shape}, dtype: {x.dtype}")
    print(f"DEBUG: ewc_loss - h shape: {h.shape}, dtype: {h.dtype}")
    print(f"DEBUG: ewc_loss - Using fisher: {fisher is not None}")
    
    w = model(x)
    print(f"DEBUG: ewc_loss - model output w shape: {w.shape}, dtype: {w.dtype}")
    
    mult_result = w * h
    print(f"DEBUG: ewc_loss - w*h result shape: {mult_result.shape}, dtype: {mult_result.dtype}")
    
    signal_power = tf.reduce_mean(tf.abs(tf.reduce_sum(mult_result, axis=1))**2)
    print(f"DEBUG: ewc_loss - signal_power: {signal_power.numpy()}")
    
    loss = -signal_power  # Maximize signal power
    
    if fisher:
        ewc_penalty = 0.0
        print(f"DEBUG: ewc_loss - calculating EWC penalty with {len(fisher)} parameters")
        
        for w in model.trainable_weights:
            w_name = w.name
            if w_name in fisher:
                ewc_penalty += tf.reduce_sum(fisher[w_name] * (w - old_params[w_name])**2)
                
        print(f"DEBUG: ewc_loss - ewc_penalty: {ewc_penalty.numpy()}")
        loss += lambda_ewc * ewc_penalty
        
    print(f"DEBUG: ewc_loss - final loss: {loss.numpy()}\n")
    return loss

# Fixed generate_channel function
def generate_channel(task, num_slots):
    print(f"\nDEBUG: generate_channel - task: {task['name']}, num_slots: {num_slots}")
    
    # Create scene
    scene = sn.rt.Scene()
    
    # Define transmitter and receivers
    tx = sn.rt.Transmitter(name="tx", position=[0, 0, 10])
    rx_positions = np.random.uniform(-100, 100, (NUM_USERS, 3))
    rx_positions[:, 2] = 1.5  # User height
    rxs = [sn.rt.Receiver(name=f"rx-{i}", position=rx_positions[i]) for i in range(NUM_USERS)]
    
    print(f"DEBUG: Created scene with 1 transmitter and {NUM_USERS} receivers")
    
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
    
    print(f"DEBUG: Configured antenna arrays")
    
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
    
    print(f"DEBUG: Created channel model")
    
    # Generate channel with mobility
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], NUM_USERS)
    print(f"DEBUG: User speeds range: {speeds.min():.2f} to {speeds.max():.2f} km/h")
    
    h = []
    for t in range(num_slots):
        if t % 100 == 0:
            print(f"DEBUG: Generating channel for slot {t}/{num_slots}")
            
        # Update receiver positions to simulate mobility
        for i, rx in enumerate(rxs):
            rx.position += [speeds[i] * 0.001 * np.cos(t * 0.1), speeds[i] * 0.001 * np.sin(t * 0.1), 0]
            
        # Generate channel
        x = tf.ones([BATCH_SIZE, NUM_ANTENNAS], dtype=tf.complex64)
        no = tf.ones([BATCH_SIZE, NUM_USERS], dtype=tf.complex64)
        
        # FlatFadingChannel returns a tuple, need to extract the channel part
        channel_output = channel_model([x, no])
        
        # Debug the output tuple
        if t == 0:
            print(f"DEBUG: channel_output is a tuple of length: {len(channel_output)}")
            for i, item in enumerate(channel_output):
                if hasattr(item, 'shape'):
                    print(f"DEBUG: channel_output[{i}] shape: {item.shape}, dtype: {item.dtype}")
                else:
                    print(f"DEBUG: channel_output[{i}] type: {type(item)}")
        
        # Extract the channel matrix (typically the first or second element)
        # Let's try to find which element is the channel matrix
        channel_matrix = None
        
        if hasattr(channel_output, 'shape'):  # If it's a single tensor
            channel_matrix = channel_output
        else:  # If it's a tuple
            # Try to find the channel matrix by looking for complex tensor with right shape
            for item in channel_output:
                if hasattr(item, 'shape') and len(item.shape) >= 2:
                    if item.shape[-2:] == (NUM_ANTENNAS, NUM_USERS) or item.shape[-2:] == (NUM_USERS, NUM_ANTENNAS):
                        channel_matrix = item
                        break
            
            # If we couldn't find it, take the first complex tensor
            if channel_matrix is None:
                for item in channel_output:
                    if hasattr(item, 'dtype') and 'complex' in str(item.dtype):
                        channel_matrix = item
                        break
            
            # If we still couldn't find it, take the first tensor
            if channel_matrix is None and len(channel_output) > 0:
                for item in channel_output:
                    if hasattr(item, 'shape'):
                        channel_matrix = item
                        break
        
        if t == 0:
            if channel_matrix is not None:
                print(f"DEBUG: Selected channel_matrix shape: {channel_matrix.shape}, dtype: {channel_matrix.dtype}")
            else:
                print("DEBUG: CRITICAL ERROR - Could not find channel matrix in output")
                # Placeholder for error case - just use zeros
                channel_matrix = tf.zeros([BATCH_SIZE, NUM_ANTENNAS, NUM_USERS], dtype=tf.complex64)
        
        h.append(channel_matrix)
    
    result = tf.stack(h)
    print(f"DEBUG: Final channel data shape: {result.shape}, dtype: {result.dtype}\n")
    return result

# 5. Main Training and Evaluation Loop with added debug info
def main():
    print("\nDEBUG: Starting main training function")
    print(f"DEBUG: TensorFlow version: {tf.__version__}")
    print(f"DEBUG: Available GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    model = BeamformingModel(NUM_ANTENNAS, NUM_USERS)
    print("DEBUG: Created model")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    print("DEBUG: Created optimizer")
    
    results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting"]}
    old_params = {}
    fisher_dict = {}
    task_performance = []
    
    for task_idx, task in enumerate(TASKS):
        print(f"\n----------------------------------------------")
        print(f"DEBUG: Starting Task {task_idx+1}/{len(TASKS)}: {task['name']}")
        print(f"----------------------------------------------")
        
        # Generate channel data
        h = generate_channel(task, NUM_SLOTS)
        print(f"DEBUG: Generated channel data for task: {task['name']}")
        
        x = h  # Keep complex channel state information
        print(f"DEBUG: Using channel data as input, shape: {x.shape}, dtype: {x.dtype}")
        
        # Train model
        for epoch in range(10):
            print(f"\nDEBUG: Starting epoch {epoch+1}/10")
            
            epoch_loss = 0
            batches = 0
            
            for t in range(0, NUM_SLOTS, BATCH_SIZE):
                batches += 1
                if t % (5 * BATCH_SIZE) == 0:
                    print(f"DEBUG: Training on batch starting at t={t}")
                    
                x_batch = x[t:t+BATCH_SIZE]
                h_batch = h[t:t+BATCH_SIZE]
                
                with tf.GradientTape() as tape:
                    loss = ewc_loss(model, x_batch, h_batch, 
                                   fisher_dict if task_idx > 0 else None, 
                                   old_params, LAMBDA_EWC)
                    
                epoch_loss += loss.numpy()
                
                grads = tape.gradient(loss, model.trainable_weights)
                
                if t == 0 and epoch == 0:
                    print(f"DEBUG: Number of gradients: {len(grads)}")
                    print(f"DEBUG: Checking for None gradients: {[g is None for g in grads]}")
                    
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                
            avg_epoch_loss = epoch_loss / batches
            print(f"DEBUG: Epoch {epoch+1} average loss: {avg_epoch_loss:.6f}")
        
        # Save parameters and Fisher for EWC
        print(f"DEBUG: Saving parameters and computing Fisher information")
        old_params = {w.name: w.numpy() for w in model.trainable_weights}
        fisher_dict = compute_fisher(model, x)
        
        # Evaluation
        print(f"DEBUG: Evaluating model performance")
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
    print("\nDEBUG: Plotting results")
    for metric in results:
        plt.plot(results[metric], label=metric)
    plt.legend()
    plt.xlabel("Task")
    plt.ylabel("Metric Value")
    plt.title("Performance Across Mobility Tasks")
    plt.savefig("mobility_tasks_results.png")
    print("DEBUG: Saved plot to mobility_tasks_results.png")
    plt.show()
    
    print("DEBUG: Training complete!")

if __name__ == "__main__":
    print("DEBUG: Script started")
    main()