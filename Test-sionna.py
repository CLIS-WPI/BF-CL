import os
import tensorflow as tf

# استفاده از فقط GPU:1 (تغییر بده اگر خواستی تست GPU دیگه‌ای انجام بدی)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
        print(f"✅ Using GPU: {gpus[1]}")
    except RuntimeError as e:
        print(f"❌ Error setting visible device: {e}")

# تست ضرب روی GPU
@tf.function
def test_mul_on_gpu():
    a = tf.ones([16, 1], dtype=tf.float32)
    b = a * 15.0
    return b

try:
    with tf.device("/GPU:0"):
        result = test_mul_on_gpu()
        print("✅ Multiplication successful on GPU:", result.numpy())
except Exception as e:
    print("❌ GPU multiplication failed with error:")
    print(e)
