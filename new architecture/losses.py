# -*- coding: utf-8 -*-
"""
Loss functions for the Continual Learning Beamforming project.

This module defines the loss functions used for training the models,
including the Elastic Weight Consolidation (EWC) loss.
"""

import tensorflow as tf
from typing import Dict


def compute_ewc_loss(model: tf.keras.Model, optimal_params_agg_np: Dict,
                     fisher_info_agg_np: Dict, ewc_lambda: float) -> tf.Tensor:
    """
    Computes the Elastic Weight Consolidation (EWC) loss.

    Args:
        model: The TensorFlow Keras model.
        optimal_params_agg_np: A dictionary of optimal parameters from
                               previous tasks (NumPy arrays).
        fisher_info_agg_np: A dictionary of Fisher information matrices from
                             previous tasks (NumPy arrays).
        ewc_lambda: The EWC regularization strength.

    Returns:
        The EWC loss as a TensorFlow scalar.
    """

    if not optimal_params_agg_np or not fisher_info_agg_np or ewc_lambda == 0:
        return tf.constant(0.0, dtype=tf.float32)

    ewc_loss: tf.Tensor = 0.0
    for var in model.trainable_variables:
        opt_param_np = optimal_params_agg_np.get(var.name)
        fisher_val_np = fisher_info_agg_np.get(var.name)

        if opt_param_np is not None and fisher_val_np is not None:
            try:
                optimal_param: tf.Tensor = tf.cast(
                    tf.convert_to_tensor(opt_param_np), var.dtype)
                fisher_val: tf.Tensor = tf.cast(
                    tf.convert_to_tensor(fisher_val_np), tf.float32)
                param_diff: tf.Tensor = var - optimal_param
                sq_diff_mag: tf.Tensor = (
                    tf.square(tf.math.real(param_diff)) +
                    tf.square(tf.math.imag(param_diff)))
                ewc_loss += tf.reduce_sum(
                    fisher_val * tf.cast(sq_diff_mag, tf.float32))
            except Exception as e:
                tf.print(f"Warning: EWC calc error {var.name}: {e}",
                         output_stream=sys.stderr)

    return 0.5 * ewc_lambda * tf.cast(ewc_loss, dtype=tf.float32)