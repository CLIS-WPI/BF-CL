import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
import traceback
import sys
import os

# Enable TensorFlow debug logging for casting operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs
tf.debugging.set_log_device_placement(True)

# Redirect TensorFlow warnings to capture them
import logging
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.DEBUG)

# Custom debug logger for complex operations
class ComplexCastingDebugger:
    def __init__(self):
        self.active = True
        self.watched_ops = ['cast', 'real', 'imag', 'complex']
        
    def watch_tensor(self, name, tensor):
        if not self.active:
            return tensor
            
        if hasattr(tensor, 'dtype'):
            if 'complex' in str(tensor.dtype):
                print(f"DEBUG-COMPLEX: {name} is complex {tensor.dtype}")
                # Print stack trace to find where this complex tensor is being used
                traceback.print_stack(limit=10)
                
        return tensor
        
    def trace_op(self, op_name, *args, **kwargs):
        if op_name in self.watched_ops:
            print(f"DEBUG-OP: {op_name} called with types: {[type(a) for a in args]}")
            if len(args) > 0 and hasattr(args[0], 'dtype'):
                print(f"DEBUG-OP: {op_name} input dtype: {args[0].dtype}")
            traceback.print_stack(limit=5)
            
debugger = ComplexCastingDebugger()

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

# Override tf.math operations to add debugging
original_real = tf.math.real
original_imag = tf.math.imag
original_complex = tf.complex

def debug_real(input):
    print("DEBUG-OVERRIDE: tf.math.real called")
    print(f"DEBUG-OVERRIDE: Input type: {type(input)}, dtype: {input.dtype if hasattr(input, 'dtype') else 'unknown'}")
    traceback.print_stack(limit=10)
    return original_real(input)

def debug_imag(input):
    print("DEBUG-OVERRIDE: tf.math.imag called")
    print(f"DEBUG-OVERRIDE: Input type: {type(input)}, dtype: {input.dtype if hasattr(input, 'dtype') else 'unknown'}")
    traceback.print_stack(limit=10)
    return original_imag(input)

def debug_complex(real, imag):
    print("DEBUG-OVERRIDE: tf.complex called")
    print(f"DEBUG-OVERRIDE: Real type: {type(real)}, dtype: {real.dtype if hasattr(real, 'dtype') else 'unknown'}")
    print(f"DEBUG-OVERRIDE: Imag type: {type(imag)}, dtype: {imag.dtype if hasattr(imag, 'dtype') else 'unknown'}")
    traceback.print_stack(limit=10)
    return original_complex(real, imag)

# Uncomment to add extreme debugging
#tf.math.real = debug_real
#tf.math.imag = debug_imag
#tf.complex = debug_complex

# 1. Define Beamforming Model with super detailed debugging
class BeamformingModel(Model):
    def __init__(self, num_antennas, num_users):
        super(BeamformingModel, self).__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        
        # Define separate layers for real and imaginary parts explicitly with float32 dtype
        self.dense1_real = layers.Dense(128, activation="relu", dtype=tf.float32, name="dense1_real")
        self.dense1_imag = layers.Dense(128, activation="relu", dtype=tf.float32, name="dense1_imag")
        
        self.dense2_real = layers.Dense(64, activation="relu", dtype=tf.float32, name="dense2_real")
        self.dense2_imag = layers.Dense(64, activation="relu", dtype=tf.float32, name="dense2_imag")
        
        self.output_real = layers.Dense(num_antennas, dtype=tf.float32, name="output_real")
        self.output_imag = layers.Dense(num_antennas, dtype=tf.float32, name="output_imag")
        
        print("DEBUG: Model initialized - layer dtypes explicitly set to float32")

    def call(self, inputs):
        print(f"\nDEBUG-CALL: Input shape: {inputs.shape}, dtype: {inputs.dtype}")
        
        # Special debug for first call to identify specific shape and ordering
        if not hasattr(self, '_first_call_done'):
            self._first_call_done = True
            if len(inputs.shape) >= 3:
                first_dims = [i for i in range(len(inputs.shape))]
                print(f"DEBUG-FIRST-CALL: Input has {len(inputs.shape)} dimensions")
                print(f"DEBUG-FIRST-CALL: Interpreting as: batch_dim={first_dims[0]}, dim1={first_dims[1]}, dim2={first_dims[2]}")
                print(f"DEBUG-FIRST-CALL: Input tensor has shape {inputs.shape}")
        
        # Get batch size
        batch_size = tf.shape(inputs)[0]
        
        # Extract real and imaginary parts and EXPLICITLY cast to float32
        print("DEBUG-CALL: Extracting real part...")
        real_inputs = tf.cast(tf.math.real(inputs), tf.float32)
        print(f"DEBUG-CALL: real_inputs dtype: {real_inputs.dtype}")
        
        print("DEBUG-CALL: Extracting imaginary part...")
        imag_inputs = tf.cast(tf.math.imag(inputs), tf.float32)
        print(f"DEBUG-CALL: imag_inputs dtype: {imag_inputs.dtype}")
        
        # Process real and imaginary parts separately
        print("DEBUG-CALL: Passing through dense1_real...")
        real_x = self.dense1_real(real_inputs)
        
        print("DEBUG-CALL: Passing through dense1_imag...")
        imag_x = self.dense1_imag(imag_inputs)
        
        real_x = self.dense2_real(real_x)
        imag_x = self.dense2_imag(imag_x)
        
        real_output = self.output_real(real_x)
        imag_output = self.output_imag(imag_x)
        
        # Combine into complex output
        print("DEBUG-CALL: Creating complex output...")
        w = tf.complex(real_output, imag_output)
        print(f"DEBUG-CALL: Complex output w dtype: {w.dtype}")
        
        # Normalize beamforming weights with proper casting
        print("DEBUG-CALL: Computing normalization...")
        norm_squared = tf.reduce_sum(tf.abs(w)**2, axis=1, keepdims=True)
        norm = tf.cast(tf.sqrt(norm_squared), dtype=tf.complex64)
        print(f"DEBUG-CALL: norm dtype: {norm.dtype}")
        
        power = tf.complex(tf.sqrt(POWER), 0.0)
        print(f"DEBUG-CALL: power dtype: {power.dtype}")
        
        w = w / norm * power
        print(f"DEBUG-CALL: Normalized w dtype: {w.dtype}")
        print("DEBUG-CALL: Returning output")
        
        return w

# Loss Function with debugging
def ewc_loss(model, x, h, fisher, old_params, lambda_ewc):
    print(f"\nDEBUG-LOSS: x shape: {x.shape}, dtype: {x.dtype}")
    print(f"DEBUG-LOSS: h shape: {h.shape}, dtype: {h.dtype}")
    
    w = model(x)
    print(f"DEBUG-LOSS: model output w shape: {w.shape}, dtype: {w.dtype}")
    
    # This is where another complex-to-float conversion might be happening
    print(f"DEBUG-LOSS: Computing signal power...")
    
    # Add detailed inspection of intermediate values
    mult_result = w * h
    print(f"DEBUG-LOSS: w*h result shape: {mult_result.shape}, dtype: {mult_result.dtype}")
    
    sum_result = tf.reduce_sum(mult_result, axis=1)
    print(f"DEBUG-LOSS: sum_result shape: {sum_result.shape}, dtype: {sum_result.dtype}")
    
    abs_result = tf.abs(sum_result)
    print(f"DEBUG-LOSS: abs_result shape: {abs_result.shape}, dtype: {abs_result.dtype}")
    
    squared = abs_result**2
    print(f"DEBUG-LOSS: squared shape: {squared.shape}, dtype: {squared.dtype}")
    
    signal_power = tf.reduce_mean(squared)
    print(f"DEBUG-LOSS: signal_power: {signal_power.numpy()}, dtype: {signal_power.dtype}")
    
    loss = -signal_power  # Maximize signal power
    
    if fisher:
        ewc_penalty = 0.0
        print(f"DEBUG-LOSS: Calculating EWC penalty with {len(fisher)} parameters")
        
        for w in model.trainable_weights:
            w_name = w.name
            if w_name in fisher:
                ewc_penalty += tf.reduce_sum(fisher[w_name] * (w - old_params[w_name])**2)
                
        print(f"DEBUG-LOSS: ewc_penalty: {ewc_penalty.numpy() if hasattr(ewc_penalty, 'numpy') else ewc_penalty}")
        loss += lambda_ewc * ewc_penalty
        
    print(f"DEBUG-LOSS: final loss: {loss.numpy() if hasattr(loss, 'numpy') else loss}")
    return loss

# Fixed generate_channel function with EXTENSIVE debugging
def generate_channel(task, num_slots):
    print(f"\nDEBUG-CHANNEL: generate_channel for task: {task['name']}")
    
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
    
    print("DEBUG-CHANNEL: Created scene and added devices")
    
    # Use FlatFadingChannel with correct parameters
    channel_model = sn.channel.FlatFadingChannel(
        num_tx_ant=NUM_ANTENNAS,  # Number of transmit antennas
        num_rx_ant=NUM_USERS,     # Number of users (receivers)
        add_awgn=True,            # Add noise
        return_channel=True       # Return channel coefficients
    )
    
    print("DEBUG-CHANNEL: Created channel model")
    
    # Generate channel with mobility
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], NUM_USERS)
    print(f"DEBUG-CHANNEL: User speeds range: {speeds.min():.2f} to {speeds.max():.2f} km/h")
    
    h = []
    for t in range(num_slots):
        if t % 100 == 0:
            print(f"DEBUG-CHANNEL: Generating channel for slot {t}/{num_slots}")
            
        # Update receiver positions to simulate mobility
        for i, rx in enumerate(rxs):
            rx.position += [speeds[i] * 0.001 * np.cos(t * 0.1), speeds[i] * 0.001 * np.sin(t * 0.1), 0]
            
        # Generate channel
        print(f"DEBUG-CHANNEL: Creating input tensors for channel model") if t == 0 else None
        x = tf.ones([BATCH_SIZE, NUM_ANTENNAS], dtype=tf.complex64)
        no = tf.ones([BATCH_SIZE, NUM_USERS], dtype=tf.complex64)
        
        if t == 0:
            print(f"DEBUG-CHANNEL: x shape: {x.shape}, dtype: {x.dtype}")
            print(f"DEBUG-CHANNEL: no shape: {no.shape}, dtype: {no.dtype}")
        
        # FlatFadingChannel returns a tuple, extract the channel part
        print(f"DEBUG-CHANNEL: Calling channel_model") if t == 0 else None
        channel_output = channel_model([x, no])
        
        # Debug the output tuple
        if t == 0:
            print(f"DEBUG-CHANNEL: channel_output is a tuple of length: {len(channel_output)}")
            for i, item in enumerate(channel_output):
                if hasattr(item, 'shape'):
                    print(f"DEBUG-CHANNEL: channel_output[{i}] shape: {item.shape}, dtype: {item.dtype}")
                    
                    # Debug the first few values to see if they're actually complex
                    if 'complex' in str(item.dtype):
                        sample = item.numpy().flatten()[:5]
                        print(f"DEBUG-CHANNEL: Sample values: {sample}")
                        print(f"DEBUG-CHANNEL: Real parts: {np.real(sample)}")
                        print(f"DEBUG-CHANNEL: Imag parts: {np.imag(sample)}")
                else:
                    print(f"DEBUG-CHANNEL: channel_output[{i}] type: {type(item)}")
        
        # Extract the channel matrix (element 1 based on your logs)
        channel_matrix = channel_output[1]
        
        if t == 0:
            print(f"DEBUG-CHANNEL: Selected channel_matrix shape: {channel_matrix.shape}, dtype: {channel_matrix.dtype}")
            
            # Check if the channel matrix is what we expect
            if channel_matrix.shape[-2:] != (NUM_USERS, NUM_ANTENNAS) and channel_matrix.shape[-2:] != (NUM_ANTENNAS, NUM_USERS):
                print(f"DEBUG-CHANNEL: WARNING - Channel matrix shape doesn't match expected dimensions")
        
        h.append(channel_matrix)
    
    result = tf.stack(h)
    print(f"DEBUG-CHANNEL: Final stacked channel data shape: {result.shape}, dtype: {result.dtype}")
    
    # Add extra check for the channel structure
    print(f"DEBUG-CHANNEL: Channel dimensions interpretation:")
    if len(result.shape) == 4:
        print(f"  dimension 0: {result.shape[0]} time slots")
        print(f"  dimension 1: {result.shape[1]} batch size")
        print(f"  dimension 2: {result.shape[2]} appears to be {result.shape[2]}")
        print(f"  dimension 3: {result.shape[3]} appears to be {result.shape[3]}")
        print(f"  This structure means each element is of size: {result.shape[2]}x{result.shape[3]}")
    
    return result

# Test function that only runs the first task with minimal iterations
def test_first_task():
    print("DEBUG-TEST: Running debug test on first task only")
    
    model = BeamformingModel(NUM_ANTENNAS, NUM_USERS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Generate channel data for the first task only
    task = TASKS[0]
    print(f"DEBUG-TEST: Generating channel for task: {task['name']}")
    
    # Use a smaller number of slots for faster testing
    test_slots = 50
    h = generate_channel(task, test_slots)
    x = h  # Use channel data as input
    
    print(f"DEBUG-TEST: Channel data shape: {h.shape}, dtype: {h.dtype}")
    
    # Try passing the data through the model
    print("DEBUG-TEST: Testing model with sample data...")
    
    # Get a single batch
    x_batch = x[0:BATCH_SIZE]
    h_batch = h[0:BATCH_SIZE]
    
    print(f"DEBUG-TEST: x_batch shape: {x_batch.shape}, dtype: {x_batch.dtype}")
    print(f"DEBUG-TEST: h_batch shape: {h_batch.shape}, dtype: {h_batch.dtype}")
    
    # Call the model directly
    print("DEBUG-TEST: Calling model directly...")
    output = model(x_batch)
    print(f"DEBUG-TEST: Model output shape: {output.shape}, dtype: {output.dtype}")
    
    # Call the loss function
    print("DEBUG-TEST: Computing loss...")
    loss = ewc_loss(model, x_batch, h_batch, None, {}, LAMBDA_EWC)
    print(f"DEBUG-TEST: Loss: {loss}")
    
    print("DEBUG-TEST: Test complete!")

# Run the test
if __name__ == "__main__":
    print("DEBUG: Starting detailed debug test")
    # Run only the first task with minimal iterations for debugging
    test_first_task()