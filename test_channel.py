import numpy as np
import tensorflow as tf
from sionna.phy.channel.tr38901 import TDL

# کاهش تعداد کاربران همزمان برای کاهش تداخل
NUM_ANTENNAS = 64
FREQ = 28e9
NUM_SLOTS = 10
BATCH_SIZE = 16
MAX_USERS_PER_SLOT = 32  # کاهش از 32 به 16
NOISE_POWER = 1e-6
POWER = 1.0

TASK = {"name": "Vehicular", "speed_range": [60, 120], "delay_spread": [200e-9, 500e-9], "channel": "TDL", "model": "C", "doppler": [500, 2000]}

def generate_channel(task, num_slots, batch_size, num_users):
    speeds = np.random.uniform(task["speed_range"][0], task["speed_range"][1], num_users)
    doppler_freq = np.random.uniform(task["doppler"][0], task["doppler"][1])
    delay = np.random.uniform(task["delay_spread"][0], task["delay_spread"][1])
    sampling_freq = int(min(1 / delay, 2 * doppler_freq))

    print("→ Using TDL model:", task["model"])
    user_channels = []
    for user_idx in range(num_users):
        tdl = TDL(
            model=task["model"],
            delay_spread=delay,
            carrier_frequency=FREQ,
            num_tx_ant=NUM_ANTENNAS,
            num_rx_ant=1,
            min_speed=task["speed_range"][0],
            max_speed=task["speed_range"][1]
        )
        h_t, _ = tdl(batch_size=batch_size, num_time_steps=num_slots, sampling_frequency=sampling_freq)
        h_t = tf.reduce_mean(h_t, axis=-1)  # میانگین روی زمان
        h_t = tf.squeeze(h_t)
        user_channels.append(h_t)
    
    h = tf.stack(user_channels, axis=1)
    
    if h.shape[-1] > 1:  # اگر بیش از یک tap وجود دارد
        h = h[..., 0]  # استفاده از اولین tap به جای میانگین‌گیری
    
    print("✅ Final TDL shape [B,U,A]:", h.shape)

    # نرمال‌سازی برای هر کاربر به طور جداگانه
    # این باعث می‌شود همه کاربران قدرت سیگنال یکسانی داشته باشند
    h_norm = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(h)), axis=2, keepdims=True))
    h = h / tf.cast(h_norm + 1e-6, tf.complex64)
    
    # اضافه کردن تصادفی‌سازی برای کاهش همبستگی کانال‌ها
    # می‌تواند به بهبود SINR کمک کند
    phase_noise = tf.random.uniform(h.shape, minval=0, maxval=2*np.pi, dtype=tf.float32)
    h = h * tf.exp(tf.complex(0.0, phase_noise))
    
    return h

# تست با MMSE beamforming بهبود یافته
h = generate_channel(TASK, NUM_SLOTS, BATCH_SIZE, MAX_USERS_PER_SLOT)

# شکل h: [B,U,A]
h_hermitian = tf.transpose(tf.math.conj(h), [0, 2, 1])  # [B,A,U]

# بررسی ابعاد
print("Shape of h:", h.shape)
print("Shape of h_hermitian:", h_hermitian.shape)

# افزایش regularization برای بهبود پایداری MMSE
reg_factor = 0.01  # افزایش ضریب regularization

# محاسبه MMSE با regularization بهبود یافته
covariance = tf.matmul(h_hermitian, h)  # [B,A,A]
identity = tf.eye(NUM_ANTENNAS, dtype=tf.complex64)
reg_matrix = covariance + (NOISE_POWER/POWER + reg_factor) * identity

# محاسبه MMSE precoder: W = (H^H * H + (σ² + λ)I)^(-1) * H^H
w_mmse = tf.matmul(tf.linalg.inv(reg_matrix), h_hermitian)  # [B,A,U]
print("Shape of w_mmse:", w_mmse.shape)

# نرمال‌سازی وزن‌ها برای هر کاربر
# تغییر روش نرمال‌سازی به per-user power constraint
w_squared_norm = tf.reduce_sum(tf.square(tf.abs(w_mmse)), axis=1, keepdims=True)  # [B,1,U]
w_mmse_normalized = w_mmse * tf.cast(tf.sqrt(POWER / (w_squared_norm + 1e-6)), tf.complex64)

# محاسبه SINR
# محاسبه سیگنال دریافتی
y = tf.matmul(h, w_mmse_normalized)  # [B,U,U]
print("Shape of received signal matrix:", y.shape)

# توان سیگنال مطلوب (عناصر قطری)
desired_signal = tf.square(tf.abs(tf.linalg.diag_part(y)))  # [B,U]

# توان تداخل (عناصر غیر قطری)
mask = 1.0 - tf.eye(MAX_USERS_PER_SLOT)
interference = tf.reduce_sum(tf.square(tf.abs(y)) * mask, axis=-1)  # [B,U]

# محاسبه SINR برای هر کاربر
sinr_per_user = desired_signal / (interference + NOISE_POWER)  # [B,U]

# میانگین SINR روی همه کاربران و نمونه‌ها
avg_sinr = tf.reduce_mean(sinr_per_user)
avg_sinr_db = 10.0 * tf.math.log(avg_sinr) / tf.math.log(10.0)

print(f"\nمیانگین SINR با MMSE beamforming بهبود یافته: {avg_sinr_db.numpy():.2f} dB")

# آمار توصیفی SINR برای همه کاربران
all_sinr_db = 10.0 * tf.math.log(sinr_per_user) / tf.math.log(10.0)
min_sinr = tf.reduce_min(all_sinr_db)
max_sinr = tf.reduce_max(all_sinr_db)
std_sinr = tf.math.reduce_std(all_sinr_db)

print(f"حداقل SINR: {min_sinr.numpy():.2f} dB")
print(f"حداکثر SINR: {max_sinr.numpy():.2f} dB")
print(f"انحراف معیار SINR: {std_sinr.numpy():.2f} dB")

# نمایش SINR برای هر کاربر (میانگین روی بسته‌ها)
user_sinr_db = 10.0 * tf.math.log(tf.reduce_mean(sinr_per_user, axis=0)) / tf.math.log(10.0)
print("\nSINR برای هر کاربر (dB):")
for i, s in enumerate(user_sinr_db.numpy()):
    if i < 5:  # فقط 5 کاربر اول را نمایش می‌دهیم
        print(f"کاربر {i+1}: {s:.2f} dB")
print("...")

# بررسی همبستگی کانال بین کاربران (می‌تواند علت تداخل زیاد را نشان دهد)
def channel_correlation(h):
    # میانگین روی بسته‌ها
    h_avg = tf.reduce_mean(h, axis=0)  # [U,A]
    h_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(h_avg)), axis=1, keepdims=True))
    h_normalized = h_avg / tf.cast(h_norm + 1e-6, tf.complex64)
    
    # محاسبه همبستگی
    correlation = tf.abs(tf.matmul(h_normalized, tf.transpose(tf.math.conj(h_normalized), [1, 0])))
    return correlation

# نمایش ماتریس همبستگی کانال (به صورت نمونه برای 5 کاربر اول)
corr_matrix = channel_correlation(h)
print("\nماتریس همبستگی کانال (5 کاربر اول):")
for i in range(min(5, MAX_USERS_PER_SLOT)):
    print([f"{corr_matrix[i, j].numpy():.2f}" for j in range(min(5, MAX_USERS_PER_SLOT))])