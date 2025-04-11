# -*- coding: utf-8 -*-
"""
Configuration file for the Continual Learning Beamforming project.

This file defines global parameters and task specifications.
"""

# --- Basic Settings ---
NUM_ANTENNAS = 64  # Number of antennas
NUM_USERS = 6  # Number of users
FREQ = 28e9  # Frequency
NUM_SLOTS = 10  # Number of time slots for averaging
NOISE_POWER_LIN = 1e-3  # Linear noise power

# --- Task Definitions ---
TASKS = [
    {"name": "Static", "speed_range": [0, 5], "delay_spread": [30e-9, 50e-9],
     "doppler": [10, 50], "channel": "TDL", "model": "A"},
    {"name": "Pedestrian", "speed_range": [5, 10], "delay_spread": [50e-9, 100e-9],
     "doppler": [50, 150], "channel": "Rayleigh"},
    {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": [200e-9, 500e-9],
     "doppler": [500, 2000], "channel": "TDL", "model": "C"},
    {"name": "Aerial", "speed_range": [20, 50], "delay_spread": [100e-9, 300e-9],
     "doppler": [200, 1000], "channel": "TDL", "model": "A"},
]
NUM_TASKS = len(TASKS)

# --- Training Configuration ---
NUM_EPOCHS_PER_TASK = 50
LEARNING_RATE = 1e-4
GLOBAL_BATCH_SIZE = 32  # Global batch size (adjust for multi-GPU)
NUM_BATCHES_PER_EPOCH = 50
ZF_REG = 1e-5  # Zero-Forcing regularization

# --- Continual Learning Configuration ---
EWC_LAMBDA = 1000.0  # EWC regularization strength (TUNING REQUIRED)
FISHER_EMA_DECAY = 0.99  # EMA decay for Fisher info

# --- Evaluation / Metrics ---
EVAL_BATCHES = 20
GPU_POWER_DRAW = 400  # Watts (Approximate for H100)

# --- File Names ---
LOG_FILE_BASE = "cl_beamforming_results"
SUMMARY_LOG_FILE = f"{LOG_FILE_BASE}_summary.log"
EWC_LOG_FILE = f"{LOG_FILE_BASE}_ewc_details.log"
FT_LOG_FILE = f"{LOG_FILE_BASE}_ft_details.log"
RT_LOG_FILE = f"{LOG_FILE_BASE}_rt_details.log"
PLOT_DOT_FILE = f"{LOG_FILE_BASE}_matrix_dot.png"
PLOT_SINR_FILE = f"{LOG_FILE_BASE}_matrix_sinr.png"
PLOT_THRPT_FILE = f"{LOG_FILE_BASE}_matrix_thrpt.png"