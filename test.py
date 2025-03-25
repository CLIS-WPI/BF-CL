import tensorflow as tf
import numpy as np
import sionna

def create_multi_user_channels(batch_size, num_users, num_antennas, carrier_frequency=28e9, delay_spread=100e-9):
    """
    Generate independent TDL channels for multiple users with specified characteristics
    
    Args:
        batch_size (int): Number of channel realizations
        num_users (int): Number of users
        num_antennas (int): Number of base station antennas
        carrier_frequency (float): Carrier frequency in Hz
        delay_spread (float): Delay spread in seconds
    
    Returns:
        Tensor of channel coefficients with shape [batch_size, num_users, num_antennas]
    """
    # User-specific channel configurations
    user_configs = [
        {
            "model": "A",  # TDL-A for static users
            "min_speed": 0.0,
            "max_speed": 1.0
        },
        {
            "model": "B",  # TDL-B for pedestrian users
            "min_speed": 1.0,
            "max_speed": 3.0
        },
        # Add more user categories as needed
    ]
    
    # Initialize list to store channels for each user
    user_channels = []
    
    for user_idx in range(num_users):
        # Cycle through user configurations
        config = user_configs[user_idx % len(user_configs)]
        
        # Create TDL channel for this user
        tdl = sionna.channel.tr38901.TDL(
            model=config["model"],
            delay_spread=delay_spread,
            carrier_frequency=carrier_frequency,
            num_tx_ant=1,  # Single tx antenna per user
            num_rx_ant=1,  # Single virtual rx antenna 
            min_speed=config["min_speed"],
            max_speed=config["max_speed"]
        )
        
        # Generate channel coefficients for this user
        # Use sampling_frequency appropriate for your scenario
        channel_coeff, _ = tdl(batch_size=batch_size, num_time_steps=1, sampling_frequency=30.72e6)
        
        # Process channel coefficients
        # Sum over paths and remove singleton dimensions
        channel_coeff = tf.reduce_sum(channel_coeff, axis=-2)  # Sum over paths
        channel_coeff = tf.squeeze(channel_coeff, axis=[1, 3, 4, -1])
        
        # Simulate multi-antenna reception by repeating single rx path
        channel_coeff = tf.repeat(channel_coeff, repeats=num_antennas, axis=-1)
        
        user_channels.append(channel_coeff)
    
    # Stack channels for all users
    multi_user_channel = tf.stack(user_channels, axis=1)
    
    return multi_user_channel

def main():
    # System parameters
    batch_size = 8
    num_users = 20
    num_antennas = 64
    carrier_frequency = 28e9
    delay_spread = 100e-9
    
    # Generate channels
    channels = create_multi_user_channels(
        batch_size=batch_size,
        num_users=num_users,
        num_antennas=num_antennas,
        carrier_frequency=carrier_frequency,
        delay_spread=delay_spread
    )
    
    print("Channel tensor shape:", channels.shape)
    
    # Visualize some channel characteristics
    channel_magnitudes = tf.abs(channels)
    print("\nChannel Magnitude Statistics:")
    print("Mean channel magnitude:", tf.reduce_mean(channel_magnitudes).numpy())
    print("Min channel magnitude:", tf.reduce_min(channel_magnitudes).numpy())
    print("Max channel magnitude:", tf.reduce_max(channel_magnitudes).numpy())

if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(1)
    np.random.seed(1)
    
    main()