import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import time
import gc
import os
# Silence casting warnings
tf.get_logger().setLevel('ERROR')

# Initial Settings (Reduced for memory efficiency)
NUM_ANTENNAS = 32  # Reduced from 64
NUM_USERS = 5      # Reduced from 10
FREQ = 28e9        # Frequency: 28 GHz (mmWave)
POWER = 1.0        # Normalized power
NUM_SLOTS = 200    # Reduced from 500
BATCH_SIZE = 16    # Reduced from 32
LAMBDA_EWC = 10.0  # Strong regularization for EWC
NUM_EPOCHS = 5     # Reduced from 10

# Define Tasks (speeds in km/h)
TASKS = [
    {"name": "Static", "speed_range": [0, 5]},
    {"name": "Pedestrian", "speed_range": [5, 10]},
    {"name": "Vehicular", "speed_range": [60, 120]},
    {"name": "Aerial", "speed_range": [20, 50]},
    {"name": "Random", "speed_range": [10, 100]},
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
        self.output_real = layers.Dense(num_antennas)
        self.output_imag = layers.Dense(num_antennas)

    def call(self, inputs):
        print(f"\nDebug BeamformingModel call:")
        print(f"Input shape: {inputs.shape}")
        print(f"Input total elements: {tf.size(inputs)}")
        
        # Store original shape
        original_shape = tf.shape(inputs)
        print(f"Original shape: {original_shape}")
        
        # For input shape (16, 16, 1, 5, 1, 32, 23, 3)
        # We need to reshape it to have the antenna dimension (32) as the last dimension
        if len(inputs.shape) > 2:
            # Combine all dimensions except the antenna dimension (index 5)
            # First, move antenna dimension to the end
            perm = [0, 1, 2, 3, 4, 6, 7, 5]  # Move dim 5 (32) to the end
            inputs = tf.transpose(inputs, perm)
            
            # Now reshape to 2D tensor with last dimension as num_antennas
            new_shape = [-1, self.num_antennas]
            inputs = tf.reshape(inputs, new_shape)
        
        print(f"Reshaped input shape: {inputs.shape}")
        
        # Process real and imaginary parts
        real_inputs = tf.cast(tf.math.real(inputs), tf.float32)
        imag_inputs = tf.cast(tf.math.imag(inputs), tf.float32)
        
        # Pass through dense layers
        real_x = self.dense1_real(real_inputs)
        imag_x = self.dense1_imag(imag_inputs)
        real_x = self.dense2_real(real_x)
        imag_x = self.dense2_imag(imag_x)
        real_output = self.output_real(real_x)
        imag_output = self.output_imag(imag_x)
        
        # Combine real and imaginary parts
        w = tf.complex(real_output, imag_output)
        
        # Normalize before reshaping back
        norm_squared = tf.reduce_sum(tf.abs(w)**2, axis=-1, keepdims=True)
        norm = tf.cast(tf.sqrt(norm_squared), dtype=tf.complex64)
        power = tf.complex(tf.sqrt(POWER), 0.0)
        w = w / norm * power
        
        # If input was multi-dimensional, reshape back to original dimensions
        if len(original_shape) > 2:
            # Calculate new shape
            new_shape = tf.concat([original_shape[:5], 
                                original_shape[6:8], 
                                [self.num_antennas]], axis=0)
            w = tf.reshape(w, new_shape)
            
            # Transpose back to original dimension order
            inv_perm = [0, 1, 2, 3, 4, 7, 5, 6]  # Move antenna dim back to position 5
            w = tf.transpose(w, inv_perm)
        
        print(f"Output shape: {w.shape}")
        
        return w
    
def compute_fisher(model, data, num_samples=50):  # Reduced samples
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

def ewc_loss(model, x, h, fisher, old_params, lambda_ewc):
    # Print shapes for debugging
    print(f"Input shapes - x: {x.shape}, h: {h.shape}")
    
    # Ensure tensors have compatible shapes
    w = model(x)
    print(f"Model output shape - w: {w.shape}")
    
    # Reshape tensors if needed
    if len(w.shape) > 5 or len(h.shape) > 5:
        w = tf.reshape(w, [-1, NUM_ANTENNAS])
        h = tf.reshape(h, [-1, NUM_ANTENNAS])
    
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

def generate_channel(task, num_slots, task_idx=0):
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], NUM_USERS)
    
    channel_model = sn.channel.tr38901.TDL(
        model="A",
        delay_spread=tf.cast(100e-9, dtype=tf.float32),
        carrier_frequency=tf.cast(FREQ, dtype=tf.float32),
        num_tx_ant=NUM_ANTENNAS,
        num_rx_ant=NUM_USERS,
        min_speed=tf.cast(speeds.min(), dtype=tf.float32),  # Use min_speed instead of ut_velocity
        max_speed=tf.cast(speeds.max(), dtype=tf.float32),  # Add max_speed parameter
        dtype=tf.complex64
    )

    h = []
    sampling_frequency = 500
    num_time_steps = 3
    chunk_size = 10
    
    for chunk_start in range(0, num_slots, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_slots)
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        for t in range(chunk_start, chunk_end):
            channel_response = channel_model(
                batch_size=BATCH_SIZE,
                num_time_steps=num_time_steps,
                sampling_frequency=sampling_frequency
            )
            h.append(channel_response[0])
            del channel_response
            
    return tf.stack(h)

def train_step(model, x_batch, h_batch, optimizer, fisher_dict, old_params, task_idx):
    with tf.GradientTape() as tape:
        # Ensure input tensors are properly shaped
        x_batch = tf.reshape(x_batch, [-1, NUM_ANTENNAS])
        h_batch = tf.reshape(h_batch, [-1, NUM_ANTENNAS])
        
        loss = ewc_loss(model, x_batch, h_batch,
                       fisher_dict if task_idx > 0 else None,
                       old_params, LAMBDA_EWC)
    
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

def main():
    print("Starting Beamforming Continual Learning Training")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Number of Antennas: {NUM_ANTENNAS}")
    print(f"Number of Users: {NUM_USERS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Number of Slots: {NUM_SLOTS}")
    print(f"EWC Lambda: {LAMBDA_EWC}")
    
    # Initialize results dictionary and tracking variables
    results = {metric: [] for metric in ["throughput", "latency", "energy", "forgetting"]}
    task_performance = []
    task_channels = {}
    old_params = {}
    fisher_dict = {}
    
    # Setup distribution strategy
    strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
    # Create model and optimizer outside the strategy scope
    model = BeamformingModel(NUM_ANTENNAS, NUM_USERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Create a directory to save models if it doesn't exist
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    total_start_time = time.time()
    
    # Moving the strategy.scope() to only wrap the training loop
    for task_idx, task in enumerate(TASKS):
        print(f"\n{'='*80}")
        print(f"Training on Task {task_idx+1}/{len(TASKS)}: {task['name']}")
        print(f"Speed Range: {task['speed_range'][0]}-{task['speed_range'][1]} km/h")
        print(f"{'='*80}")
        
        print("\nGenerating channel data...")
        h = generate_channel(task, NUM_SLOTS, task_idx)
        task_channels[task_idx] = h
        x = h
        
        print("\nPreparing dataset...")
        dataset = tf.data.Dataset.from_tensor_slices((x, h))
        dataset = dataset.batch(BATCH_SIZE)
        
        # Training within strategy scope
        with strategy.scope():
            dist_dataset = strategy.experimental_distribute_dataset(dataset)
            
            # Calculate number of batches without using cardinality
            num_samples = x.shape[0]
            num_batches = num_samples // BATCH_SIZE
            if num_samples % BATCH_SIZE != 0:
                num_batches += 1
            print(f"Number of batches per epoch: {num_batches}")
            
            task_start_time = time.time()
            for epoch in range(NUM_EPOCHS):
                print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
                print("-" * 40)
                epoch_start_time = time.time()
                epoch_loss = 0
                batches = 0
                
                # Progress bar variables
                progress_interval = max(1, num_batches // 20)  # Update progress ~20 times per epoch
                
                for x_batch, h_batch in dist_dataset:
                    batch_start_time = time.time()
                    loss = strategy.run(train_step, args=(model, x_batch, h_batch,
                                                        optimizer, fisher_dict,
                                                        old_params, task_idx))
                    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
                    epoch_loss += loss
                    batches += 1
                    
                    # Print progress every few batches
                    if batches % progress_interval == 0:
                        progress = (batches / num_batches) * 100
                        batch_time = time.time() - batch_start_time
                        print(f"Progress: {progress:3.1f}% [{batches}/{num_batches}] "
                              f"| Batch Loss: {loss:.6f} | Batch Time: {batch_time:.3f}s")
                
                avg_epoch_loss = epoch_loss / batches
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"Average Loss: {avg_epoch_loss:.6f}")
                print(f"Epoch Time: {epoch_time:.2f}s")
        
        # Evaluation and metrics outside the strategy scope
        task_time = time.time() - task_start_time
        training_time = task_time
        latency = training_time / (NUM_EPOCHS * NUM_SLOTS) * 1000
        
        print(f"\nTask {task['name']} Completion Metrics:")
        print("-" * 40)
        
        w = model(x)
        throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(w * h, axis=1))**2).numpy()
        task_performance.append(throughput)
        
        # Calculate forgetting
        forgetting = 0.0
        if task_idx > 0:
            past_performances = []
            print("\nEvaluating forgetting on previous tasks:")
            for past_idx in range(task_idx):
                past_h = task_channels[past_idx]
                past_w = model(past_h)
                past_throughput = tf.reduce_mean(tf.abs(tf.reduce_sum(past_w * past_h, axis=1))**2).numpy()
                forgetting_delta = task_performance[past_idx] - past_throughput
                past_performances.append(forgetting_delta)
                print(f"Task {past_idx+1} forgetting: {forgetting_delta:.4f}")
            forgetting = np.mean(past_performances) if past_performances else 0.0
        
        # Update EWC parameters
        print("\nComputing Fisher information...")
        old_params = {w.name: w.numpy() for w in model.trainable_weights}
        fisher_dict = compute_fisher(model, x)
        
        energy = training_time * 100
        
        # Store results
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
        
        # Add model saving here
        save_dir = "saved_models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save(os.path.join(save_dir, f"model_task_{task['name']}"))
        print(f"\nModel saved for task {task['name']}")
    
    total_training_time = time.time() - total_start_time
    print(f"\nTotal Training Time: {total_training_time:.2f}s")
    
    # Plot results
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
    plt.savefig("mobility_tasks_results_updated.png")
    plt.show()
    
    print("\nTraining complete!")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()