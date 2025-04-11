# -*- coding: utf-8 -*-
"""
Metrics module for the Continual Learning Beamforming project.

This module defines functions to calculate performance metrics such as
SINR, alignment, and throughput.
"""
import tensorflow as tf
import numpy as np
from typing import Tuple, List

def calculate_metrics_for_logging(h: tf.Tensor, w_pred_norm: tf.Tensor,
                                     noise_power: float = 1e-3) -> Tuple[
    tf.Tensor, tf.Tensor, tf.Tensor, List[str]]:
    """
    Calculates SINR (dB), Mean Alignment |dot|, Avg Throughput (bps/Hz)
    for logging during training.

    Args:
        h: The channel tensor of shape [batch_size, num_users, num_antennas].
        w_pred_norm: The normalized predicted weights.
        noise_power: The noise power.

    Returns:
        A tuple containing:
        - Mean SINR in dB (scalar tensor).
        - Mean absolute value of the dot product (scalar tensor).
        - Average throughput (scalar tensor).
        - A list of log lines for detailed user-specific metrics.
    """

    sinr_dB_batch_mean: tf.Tensor = tf.constant(np.nan, dtype=tf.float32)
    mean_dot_abs: tf.Tensor = tf.constant(np.nan, dtype=tf.float32)
    avg_throughput: tf.Tensor = tf.constant(np.nan, dtype=tf.float32)
    log_lines: List[str] = []

    try:
        h_stop: tf.Tensor = tf.stop_gradient(
            tf.convert_to_tensor(h, dtype=tf.complex64))
        w_pred_norm_stop: tf.Tensor = tf.stop_gradient(
            tf.convert_to_tensor(w_pred_norm, dtype=tf.complex64))

        h_norm: tf.Tensor = tf.nn.l2_normalize(h_stop, axis=-1, epsilon=1e-8)
        complex_dots: tf.Tensor = tf.reduce_sum(
            h_norm * tf.math.conj(w_pred_norm_stop), axis=-1)  # [B, U]
        dot_abs_all: tf.Tensor = tf.abs(complex_dots)  # [B, U]
        mean_dot_abs = tf.reduce_mean(dot_abs_all)  # scalar

        num_users: int = tf.shape(h_stop)[1]
        batch_size: int = tf.shape(h_stop)[0]
        if batch_size == 0:
            raise ValueError("Batch size is zero in metric calculation")

        signal_matrix: tf.Tensor = tf.matmul(h_stop, w_pred_norm_stop,
                                            adjoint_b=True)  # [B, U, U]
        desired_signal: tf.Tensor = tf.linalg.diag_part(signal_matrix)  # [B, U]
        desired_power: tf.Tensor = tf.abs(desired_signal) ** 2  # [B, U]

        mask: tf.Tensor = 1.0 - tf.eye(num_users, batch_shape=[batch_size],
                                      dtype=h_stop.dtype.real_dtype)  # [B, U, U]
        interference_power: tf.Tensor = tf.reduce_sum(
            tf.abs(signal_matrix) ** 2 * mask, axis=-1)  # [B, U]

        sinr_linear: tf.Tensor = desired_power / (
            interference_power + noise_power)
        sinr_dB_batch_mean = 10.0 * tf.math.log(
            tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0)

        sinr_per_user_bps: tf.Tensor = tf.math.log(1.0 + sinr_linear) / \
                                       tf.math.log(2.0)
        avg_throughput = tf.reduce_mean(sinr_per_user_bps)

        sinr_user0: tf.Tensor = sinr_per_user_bps[0]
        dot_abs_user0: tf.Tensor = dot_abs_all[0]
        complex_dots_user0: tf.Tensor = complex_dots[0]
        for u in range(num_users):
            complex_dot_u = complex_dots_user0[u]
            dot_real: tf.Tensor = tf.math.real(complex_dot_u)
            dot_abs: tf.Tensor = tf.abs(complex_dot_u)
            angle: tf.Tensor = tf.math.angle(complex_dot_u)
            sinr_u_val: float = sinr_user0[u].numpy()  # requires CPU transfer
            if (dot_abs < 0.8) or (tf.abs(angle) > np.pi / 6):
                line: str = (
                    f"User {u:02d} | dot={dot_real:.4f} | angle={angle:.4f} "
                    f"rad | |dot|={dot_abs:.4f} | SINR={sinr_u_val:.2f} bps/Hz")
                log_lines.append(line)

    except Exception as e:
        tf.print("Error during metric calculation:", e, output_stream=sys.stderr)
        sinr_dB_batch_mean = tf.constant(np.nan, dtype=tf.float32)
        mean_dot_abs = tf.constant(np.nan, dtype=tf.float32)
        avg_throughput = tf.constant(np.nan, dtype=tf.float32)
        log_lines = ["Error in metrics calc."]

    return (tf.identity(sinr_dB_batch_mean), tf.identity(mean_dot_abs),
            tf.identity(avg_throughput), log_lines)


def calculate_metrics(h: tf.Tensor, w_pred_raw: tf.Tensor,
                      noise_power: float = 1e-3) -> Tuple[float, float, float]:
    """
    Calculates |dot|, SINR (dB), Avg Thrpt (bps/Hz) from raw prediction.

    Args:
        h: The channel tensor.
        w_pred_raw: The raw predicted weights.
        noise_power: The noise power.

    Returns:
        A tuple containing:
        - Average SINR in dB (float).
        - Average absolute value of the dot product (float).
        - Average throughput (float).
    """

    w_pred_norm: tf.Tensor = tf.nn.l2_normalize(w_pred_raw, axis=-1,
                                               epsilon=1e-8)
    sinr_dB: float = np.nan
    mean_dot: float = np.nan
    avg_thrpt: float = np.nan

    try:
        # Use tf.stop_gradient to ensure metric calculation
        # doesn't affect training gradients
        h_stop: tf.Tensor = tf.stop_gradient(h)
        w_pred_norm_stop: tf.Tensor = tf.stop_gradient(w_pred_norm)

        # Run calculations potentially outside tf.function context or on CPU
        h_norm: tf.Tensor = tf.nn.l2_normalize(h_stop, axis=-1, epsilon=1e-8)
        complex_dots: tf.Tensor = tf.reduce_sum(
            h_norm * tf.math.conj(w_pred_norm_stop), axis=-1)
        dot_abs_all: tf.Tensor = tf.abs(complex_dots)
        mean_dot = tf.reduce_mean(dot_abs_all).numpy()  # Convert to numpy

        num_users: int = tf.shape(h_stop)[1]
        batch_size: int = tf.shape(h_stop)[0]
        signal_matrix: tf.Tensor = tf.matmul(h_stop, w_pred_norm_stop,
                                            adjoint_b=True)
        desired_signal: tf.Tensor = tf.linalg.diag_part(signal_matrix)
        desired_power: tf.Tensor = tf.abs(desired_signal) ** 2
        mask: tf.Tensor = 1.0 - tf.eye(num_users, batch_shape=[batch_size],
                                      dtype=h_stop.dtype.real_dtype)
        interference_power_masked: tf.Tensor = tf.abs(signal_matrix) ** 2 * mask
        interference_power: tf.Tensor = tf.reduce_sum(
            interference_power_masked, axis=-1)
        sinr_linear: tf.Tensor = desired_power / (
            interference_power + noise_power)
        sinr_dB = 10.0 * tf.math.log(
            tf.reduce_mean(sinr_linear) + 1e-9) / tf.math.log(10.0)
        sinr_dB = sinr_dB.numpy()  # Convert to numpy
        sinr_per_user_bps: tf.Tensor = tf.math.log(1.0 + sinr_linear) / \
                                       tf.math.log(2.0)
        avg_thrpt = tf.reduce_mean(sinr_per_user_bps).numpy()  # Convert to numpy

    except Exception as e:
        print(f"Warning: Error during metric calculation: {e}", file=sys.stderr)

    return sinr_dB, mean_dot, avg_thrpt