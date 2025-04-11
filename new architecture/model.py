# -*- coding: utf-8 -*-
"""
Model module for the Continual Learning Beamforming project.

This module defines the MultiHeadCVNNModel class.
"""

import tensorflow as tf
import cvnn.layers as complex_layers
import cvnn.activations as complex_activations

class MultiHeadCVNNModel(tf.keras.Model):
    def __init__(self, num_antennas: int, num_users: int, num_tasks: int):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        self.num_tasks = num_tasks
        hidden_dim1 = 128
        hidden_dim2 = 256
        initializer = 'glorot_uniform'
        activation = complex_activations.cart_relu
        self.dense1 = complex_layers.ComplexDense(hidden_dim1, activation=activation,
                                                kernel_initializer=initializer)
        self.dense2 = complex_layers.ComplexDense(hidden_dim2, activation=activation,
                                                kernel_initializer=initializer)
        self.output_heads = []
        for i in range(num_tasks):
            head = complex_layers.ComplexDense(self.num_antennas, activation='linear',
                                            kernel_initializer=initializer, name=f'head_task_{i}')
            self.output_heads.append(head)

    def call(self, inputs: tf.Tensor, task_idx: int, training: bool = False) -> tf.Tensor:
        x1 = self.dense1(inputs)
        x2 = self.dense2(x1)
        if not (0 <= task_idx < self.num_tasks):
            raise ValueError(f"Invalid task_idx: {task_idx}.")
        selected_head = self.output_heads[task_idx]
        w = selected_head(x2)
        return w  # Return raw weights