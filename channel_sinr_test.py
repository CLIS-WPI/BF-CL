# channel_sinr_test.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading

# تنظیمات پایه
NUM_ANTENNAS = 64
NUM_USERS = 8
BATCH_SIZE = 4
FREQ = 28e9
NOISE_POWER = 1e-8
POWER = 1.0

# Beamspace projection
def apply_beamspace(h):
    h = tf.reshape(h, [tf.shape(h)[0], -1, NUM_ANTENNAS])  # [B, U, A]
    dft_matrix = tf.signal.fft(tf.eye(NUM_ANTENNAS, dtype=tf.complex64))  # [A, A]
    return tf.einsum("bua,ac->buc", h, dft_matrix)

# کانال تست
def generate_channel(task_name):
    if task_name == "TDL":
        delay_spread = 100e-9
        model = "A"
        print("\n== Testing TDL ==")
        user_channels = []
        for _ in range(NUM_USERS):
            tdl = TDL(
                model=model,
                delay_spread=delay_spread,
                carrier_frequency=FREQ,
                num_tx_ant=NUM_ANTENNAS,
                num_rx_ant=1
            )
            h_t, _ = tdl(batch_size=BATCH_SIZE, num_time_steps=4, sampling_frequency=10000)
            h_t = tf.reduce_mean(h_t, axis=[-1, -2])  # [B, A]
            h_t = tf.expand_dims(h_t, axis=1)
            h_beam = apply_beamspace(h_t)
            user_channels.append(tf.squeeze(h_beam, axis=1))  # [B, A]
        h = tf.stack(user_channels, axis=1)  # [B, U, A]

    elif task_name == "Rayleigh":
        print("\n== Testing Rayleigh ==")
        channel_model = RayleighBlockFading(
            num_rx=NUM_USERS,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=NUM_ANTENNAS
        )
        h, _ = channel_model(batch_size=BATCH_SIZE, num_time_steps=1)
        h = apply_beamspace(h)

    elif task_name == "Mixed":
        print("\n== Testing Mixed ==")
        if np.random.rand() < 0.5:
            return generate_channel("TDL")
        else:
            return generate_channel("Rayleigh")

    h_norm = tf.reduce_mean(tf.abs(h))
    h = h / tf.cast(h_norm + 1e-6, tf.complex64)
    tf.print("✅ Shape of h:", tf.shape(h))
    tf.print("✅ Mean |h|:", tf.reduce_mean(tf.abs(h)), summarize=-1)
    tf.print("✅ Std |h|: ", tf.math.reduce_std(tf.abs(h)), summarize=-1)
    return h

# Beamforming → matched filter
def matched_filter_beamforming(h):
    h_hermitian = tf.transpose(tf.math.conj(h), [0, 2, 1])  # [B, A, U]
    w = h_hermitian
    w = tf.transpose(w, [0, 2, 1])  # [B, U, A]
    w = tf.math.l2_normalize(w, axis=-1) * tf.cast(tf.sqrt(tf.cast(NUM_ANTENNAS, tf.complex64)), tf.complex64)
    return w

# محاسبه SINR
def compute_sinr(h, w):
    B = tf.shape(h)[0]
    U = tf.shape(h)[1]
    h_hermitian = tf.transpose(tf.math.conj(h), [0, 2, 1])  # [B, A, U]

    signal_matrix = tf.matmul(w, h_hermitian)  # [B, U, U]
    desired_signal = tf.linalg.diag_part(signal_matrix)
    desired_power = tf.reduce_mean(tf.abs(desired_signal)**2)

    interference_matrix = tf.abs(signal_matrix)**2
    interference_mask = 1.0 - tf.eye(U, batch_shape=[B])
    interference = tf.reduce_mean(tf.reduce_sum(interference_matrix * interference_mask, axis=-1))

    sinr = desired_power / (interference + NOISE_POWER + 1e-10)
    sinr_db = 10.0 * tf.math.log(sinr + 1e-8) / tf.math.log(10.0)
    throughput = tf.math.log(1.0 + sinr) / tf.math.log(2.0)

    tf.print("📶 Desired Power:", desired_power)
    tf.print("📡 Interference:", interference)
    tf.print("🔊 SINR (dB):", sinr_db)
    tf.print("🚀 Throughput (bps/Hz):", throughput)
    return sinr_db, throughput

# اجرای تست برای هر نوع کانال
for channel_type in ["TDL", "Rayleigh", "Mixed"]:
    h = generate_channel(channel_type)
    w = matched_filter_beamforming(h)
    compute_sinr(h, w)
