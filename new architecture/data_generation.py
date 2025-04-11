# -*- coding: utf-8 -*-
"""
Data generation module for the Continual Learning Beamforming project.

This module provides functions to generate synthetic channel data.
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple

try:
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
except ImportError as e:
    print("Error importing Sionna. Please ensure it is installed correctly.")
    print(e)
    import sys

    sys.exit()


def generate_synthetic_batch(task: Dict, batch_size: int, num_antennas: int,
                             num_users: int, num_slots: int, freq: float) -> tf.Tensor:
    """
    Generates a batch of synthetic channel data.

    Args:
        task: A dictionary defining the task parameters
              (e.g., delay_spread, doppler, channel type).
        batch_size: The number of samples in the batch.
        num_antennas: The number of transmitting antennas.
        num_users: The number of users.
        num_slots: The number of time slots.
        freq: The carrier frequency.

    Returns:
        A TensorFlow tensor of shape [batch_size, num_users, num_antennas]
        representing the complex channel.
    """

    h_users: List[tf.Tensor] = []
    for _ in range(num_users):
        delay: float = np.random.uniform(*task["delay_spread"])
        doppler: float = np.random.uniform(*task["doppler"])
        sampling_freq: int = int(max(1 / (delay + 1e-9), 2 * doppler)) * 10
        h_user: tf.Tensor = None

        if task["channel"] == "TDL":
            tdl = TDL(model=task.get("model", "A"), delay_spread=delay,
                      carrier_frequency=freq, num_tx_ant=num_antennas,
                      num_rx_ant=1, min_speed=task["speed_range"][0],
                      max_speed=task["speed_range"][1])
            h_time, _ = tdl(batch_size=batch_size, num_time_steps=num_slots,
                            sampling_frequency=sampling_freq)
            h_avg_time: tf.Tensor = tf.reduce_mean(h_time, axis=-1)
            h_comb_paths: tf.Tensor = tf.reduce_sum(h_avg_time, axis=-1)
            h_user = tf.squeeze(h_comb_paths, axis=[1, 2])
        elif task["channel"] == "Rayleigh":
            rb = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1,
                                     num_tx_ant=num_antennas)
            h_block, _ = rb(batch_size=batch_size, num_time_steps=1)
            h_user = tf.squeeze(h_block, axis=[1, 2, 3, 5])

        if h_user is None:
            raise ValueError("h_user not defined")

        h_user_reshaped: tf.Tensor = tf.reshape(h_user,
                                               [batch_size, 1, num_antennas])
        h_users.append(h_user_reshaped)

    h_stacked: tf.Tensor = tf.stack(h_users, axis=1)
    h_stacked_squeezed: tf.Tensor = (
        tf.squeeze(h_stacked, axis=2) if tf.rank(h_stacked) == 4 and
                                         tf.shape(h_stacked)[2] == 1
        else h_stacked)
    h_norm: tf.Tensor = h_stacked_squeezed / (
        tf.cast(tf.norm(h_stacked_squeezed, axis=-1, keepdims=True),
                tf.complex64) + 1e-8)

    return h_norm