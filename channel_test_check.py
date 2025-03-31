# channel_test_check.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading

# Configs
NUM_ANTENNAS = 64
BATCH_SIZE = 4
NUM_USERS = 8
FREQ = 28e9  # 28 GHz
NUM_SLOTS = 10

def apply_beamspace(h):
    h = tf.reshape(h, [tf.shape(h)[0], -1, NUM_ANTENNAS])  # Always [B, U, A]
    dft_matrix = tf.signal.fft(tf.eye(NUM_ANTENNAS, dtype=tf.complex64))  # [A, A]
    return tf.einsum("bua,ac->buc", h, dft_matrix)

def generate_channel(task_name):
    print(f"\n== Testing {task_name} ==")

    if task_name == "TDL":
        delay_spread = 100e-9
        doppler = 500
        model = "A"
        speed_range = [20, 40]
        sampling_freq = int(min(1 / delay_spread, 2 * doppler))

        user_channels = []
        for _ in range(NUM_USERS):
            tdl = TDL(
                model=model,
                delay_spread=delay_spread,
                carrier_frequency=FREQ,
                num_tx_ant=NUM_ANTENNAS,
                num_rx_ant=1,
                min_speed=speed_range[0],
                max_speed=speed_range[1]
            )
            h_t, _ = tdl(batch_size=BATCH_SIZE, num_time_steps=NUM_SLOTS, sampling_frequency=sampling_freq)
            h_t = tf.reduce_mean(h_t, axis=[-1, -2])  # [B, A]
            h_t = tf.expand_dims(h_t, axis=1)        # [B, 1, A]
            h_beam = apply_beamspace(h_t)
            user_channels.append(tf.squeeze(h_beam, axis=1))  # [B, A]

        h = tf.stack(user_channels, axis=1)  # [B, U, A]

    elif task_name == "Rayleigh":
        channel_model = RayleighBlockFading(
            num_rx=NUM_USERS,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=NUM_ANTENNAS
        )
        h, _ = channel_model(batch_size=BATCH_SIZE, num_time_steps=1)  # [B, U, A]
        h = apply_beamspace(h)

    elif task_name == "Mixed":
        model = np.random.choice(["A", "C"]) if np.random.rand() < 0.5 else "Rayleigh"
        print(f"→ Mixed mode picked model: {model}")
        if model != "Rayleigh":
            delay_spread = 200e-9
            doppler = 1000
            speed_range = [10, 60]
            sampling_freq = int(min(1 / delay_spread, 2 * doppler))
            user_channels = []
            for _ in range(NUM_USERS):
                tdl = TDL(
                    model=model,
                    delay_spread=delay_spread,
                    carrier_frequency=FREQ,
                    num_tx_ant=NUM_ANTENNAS,
                    num_rx_ant=1,
                    min_speed=speed_range[0],
                    max_speed=speed_range[1]
                )
                h_t, _ = tdl(batch_size=BATCH_SIZE, num_time_steps=NUM_SLOTS, sampling_frequency=sampling_freq)
                h_t = tf.reduce_mean(h_t, axis=[-1, -2])
                h_t = tf.expand_dims(h_t, axis=1)
                h_beam = apply_beamspace(h_t)
                user_channels.append(tf.squeeze(h_beam, axis=1))
            h = tf.stack(user_channels, axis=1)
        else:
            channel_model = RayleighBlockFading(
                num_rx=NUM_USERS,
                num_rx_ant=1,
                num_tx=1,
                num_tx_ant=NUM_ANTENNAS
            )
            h, _ = channel_model(batch_size=BATCH_SIZE, num_time_steps=1)
            h = apply_beamspace(h)
    else:
        raise ValueError("Unknown channel type")

    h_norm = tf.reduce_mean(tf.abs(h))
    h = h / tf.cast(h_norm + 1e-6, tf.complex64)

    print(f"✅ Shape of h: {h.shape}")
    print(f"✅ Mean |h|: {tf.reduce_mean(tf.abs(h)).numpy():.4f}")
    print(f"✅ Std |h|:  {tf.math.reduce_std(tf.abs(h)).numpy():.4f}")
    return h

if __name__ == "__main__":
    for ch in ["TDL", "Rayleigh", "Mixed"]:
        _ = generate_channel(ch)
