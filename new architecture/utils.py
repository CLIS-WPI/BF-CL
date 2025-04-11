# -*- coding: utf-8 -*-
"""
Utility functions for the Continual Learning Beamforming project.

This module provides general utility functions that are used across
different parts of the project.
"""

import tensorflow as tf
import numpy as np
from typing import Dict


def aggregate_fisher_and_params(model: tf.keras.Model,
                               current_task_fisher_ema: Dict[str, tf.Variable],
                               optimal_params: Dict[int, Dict[str, np.ndarray]],
                               fisher_information: Dict[int, Dict[str, np.ndarray]],
                               task_idx: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Aggregates Fisher information and optimal parameters from previous tasks.

    This function prepares the Fisher information and optimal parameters from
    previous tasks to be used in the EWC loss calculation for the current task.

    Args:
        model: The TensorFlow Keras model.
        current_task_fisher_ema: The Fisher information EMA for the current task.
        optimal_params: A dictionary storing optimal parameters for previous tasks.
        fisher_information: A dictionary storing Fisher information for previous tasks.
        task_idx: The index of the current task.

    Returns:
        A tuple containing two dictionaries:
        - optimal_params_prev_agg: Aggregated optimal parameters from previous tasks.
        - fisher_info_prev_agg: Aggregated Fisher information from previous tasks.
    """

    optimal_params_prev_agg: Dict[str, np.ndarray] = {}
    fisher_info_prev_agg: Dict[str, np.ndarray] = {}
    if task_idx > 0:
        for prev_idx in range(task_idx):
            if prev_idx in optimal_params and prev_idx in fisher_information:
                for var_name, param_val in optimal_params[prev_idx].items():
                    if var_name not in optimal_params_prev_agg:
                        optimal_params_prev_agg[var_name] = param_val
                    fisher_info_prev_agg[var_name] = (
                        fisher_info_prev_agg.get(var_name, 0.0) +
                        fisher_information[prev_idx].get(var_name, 0.0)
                    )
    return optimal_params_prev_agg, fisher_info_prev_agg


def update_fisher_ema(model: tf.keras.Model,
                      current_task_fisher_ema: Dict[str, tf.Variable],
                      grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]],
                      fisher_ema_decay: float) -> None:
    """
    Updates the Fisher information Exponential Moving Average (EMA).

    This function calculates the squared magnitude of the gradients and updates
    the Fisher information EMA for each trainable variable in the model.

    Args:
        model: The TensorFlow Keras model.
        current_task_fisher_ema: The Fisher information EMA variables for the current task.
        grads_and_vars: A list of (gradient, variable) tuples.
        fisher_ema_decay: The decay rate for the EMA.
    """

    var_to_grad: Dict[tf.Variable, tf.Tensor] = {
        v.ref(): g for g, v in grads_and_vars
    }
    for var_name, fisher_variable in current_task_fisher_ema.items():
        target_var: Optional[tf.Variable] = next(
            (v_model for v_model in model.trainable_variables if
             v_model.name == var_name), None)
        if target_var is not None:
            g: Optional[tf.Tensor] = var_to_grad.get(target_var.ref())
            if g is not None:
                grad_sq_mag: tf.Tensor = tf.cast(
                    tf.math.real(g * tf.math.conj(g)), dtype=tf.float32)
                avg_sq_grad_mag_this_step: tf.Tensor = tf.reduce_mean(
                    grad_sq_mag)
                new_fisher_val: tf.Tensor = (
                    fisher_ema_decay * fisher_variable) + (
                    (1.0 - fisher_ema_decay) * avg_sq_grad_mag_this_step)
                fisher_variable.assign(new_fisher_val)