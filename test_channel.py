import tensorflow as tf
import numpy as np
import logging

# ثابت‌ها
BATCH_SIZE = 16
NUM_ANTENNAS = 64
NUM_SLOTS = 1
NOISE_POWER = 1e-8
FREQ = 3e9
POWER = 1000.0
EPSILON = 0.1  # regularization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_channel(num_users, batch_size=BATCH_SIZE, num_antennas=NUM_ANTENNAS):
    h = tf.complex(
        tf.random.normal([batch_size, num_users, num_antennas], 0, 1/np.sqrt(2), dtype=tf.float64),
        tf.random.normal([batch_size, num_users, num_antennas], 0, 1/np.sqrt(2), dtype=tf.float64)
    )
    return h

def compute_zf_weights(h_batch, reg=0.01):
    B = tf.shape(h_batch)[0]
    U = tf.shape(h_batch)[1]

    h_hermitian = tf.transpose(tf.math.conj(h_batch), perm=[0, 2, 1])  # [B, A, U]
    hh_h = tf.matmul(h_batch, h_hermitian)  # [B, U, A] x [B, A, U] → [B, U, U]

    # Add regularization (diagonal loading)
    eye = tf.eye(U, batch_shape=[B], dtype=tf.complex64)
    hh_h_reg = hh_h + tf.cast(reg, tf.complex64) * eye

    # Inversion
    hh_h_inv = tf.linalg.inv(hh_h_reg)  # [B, U, U]
    w = tf.matmul(hh_h_inv, h_batch)  # [B, U, A]

    return w  # complex64, GPU-compatible


def compute_sinr_and_loss(h_batch, power=1000.0, noise_power=1e-8):
    h_batch = tf.cast(h_batch, tf.complex64)
    B = tf.shape(h_batch)[0]
    U = tf.shape(h_batch)[1]

    w = compute_zf_weights(h_batch)  # ✅ ZF weights
    w = w * tf.cast(tf.sqrt(power), tf.complex64)

    signal_matrix = tf.matmul(h_batch, w, transpose_b=True)  # [B, U, U]
    desired_signal = tf.linalg.diag_part(signal_matrix)
    desired_power = tf.reduce_mean(tf.abs(desired_signal) ** 2)

    interference_matrix = tf.abs(signal_matrix) ** 2
    mask = 1.0 - tf.eye(U, batch_shape=[B], dtype=tf.float32)
    interference = tf.reduce_mean(tf.reduce_sum(interference_matrix * mask, axis=-1))

    snr = desired_power / (interference + noise_power)
    sinr_db = 10.0 * tf.math.log(snr + 1e-8) / tf.math.log(10.0)
    loss = -tf.reduce_mean(tf.math.log(1.0 + snr))

    tf.print("[GPU-ZF] desired_power:", desired_power)
    tf.print("[GPU-ZF] interference:", interference)
    tf.print("[GPU-ZF] mean SNR (dB):", sinr_db)
    tf.print("[GPU-ZF] loss:", loss)
    tf.print()

    return sinr_db, loss


def test_channel_for_users(user_counts):
    for U in user_counts:
        logger.info(f"\n\n=== Testing with {U} users ===")
        h = generate_channel(num_users=U)
        compute_sinr_and_loss(h)

if __name__ == "__main__":
    test_channel_for_users([1, 2, 4, 8, 16, 32, 64, 128, 256, 500])

