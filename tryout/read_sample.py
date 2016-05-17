"""Tryout, giving data to TensorFlow."""
import numpy
import tensorflow as tf

filename_queue = tf.train.string_input_producer([
    "fake_data/data1.csv",
    "fake_data/data2.csv"]
)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0], [0]]
col1, col2 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.pack([col1])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        # Retrieve a single instance:
        example, label = sess.run([features, col2])
        print example, label

    coord.request_stop()
    coord.join(threads)
