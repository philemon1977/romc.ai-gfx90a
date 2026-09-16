import tensorflow as tf
print("tf", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("visible GPUs:", len(gpus))
assert gpus, "TensorFlow sees no GPU!"
with tf.device("/GPU:0"):
    a = tf.random.normal([2048, 2048])
    c = tf.reduce_sum(a @ a).numpy()
print("matmul 2048^3 ok, checksum:", round(float(c), 1))
print("TENSORFLOW: PASS")
