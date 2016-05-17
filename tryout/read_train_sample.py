"""Neural Network Tryout using TensorFlow."""
# !/usr/bin/python
# -*- coding: utf8 -*-
import numpy as np
import tensorflow as tf

from numpy import genfromtxt
train_data = genfromtxt('fake_data/data1.csv', delimiter=',', dtype='float32')
test_data = genfromtxt('fake_data/data2.csv', delimiter=',', dtype='float32')

n_samples = train_data.shape[0]
n_features = train_data.shape[1] - 1

train_features = np.reshape(
    train_data[:, :n_features],
    (n_samples, n_features))

train_labels = np.reshape(train_data[:, n_features], (n_samples, 1))

n_test_samples = test_data.shape[0]
test_features = np.reshape(
    test_data[:, :n_features],
    (n_test_samples, n_features))

test_labels = np.reshape(test_data[:, n_features], (n_test_samples, 1))
print test_labels.shape


def weight_variable(shape):
    """Initialize weight variables."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Initialize bias variables."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


x = tf.placeholder(tf.float32, shape=[None, n_features])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_t = tf.placeholder(tf.float32, shape=[None, n_features])
y_t = tf.placeholder(tf.float32, shape=[None, 1])

W = weight_variable([n_features, 1])
b = bias_variable([n_features])

y = tf.nn.sigmoid(tf.matmul(x, W) + b)
o = tf.round(y)
o_t = tf.round(tf.nn.sigmoid(tf.matmul(x_t, W) + b))

# cross_entropy = tf.reduce_mean(
#     -tf.reduce_sum((y_ - y) ** 2), reduction_indices=[1])

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

loss = tf.nn.l2_loss(y - y_)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(o, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

filenames = [
    "fake_data/data1.csv",
    "fake_data/data2.csv"
]


sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())

for i in range(2000):
    # batch = input_pipeline(filenames, batch_size=10)
    tt, output, output_test, acc = sess.run([train_step, o, o_t, accuracy], feed_dict={
        x: train_features,
        y_: train_labels,
        x_t: test_features,
        y_t: test_labels})
    if i % 200 == 0:
        # print W.eval()
        # print output
        print acc
        print output_test

# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# acc, output, y = sess.run(
#     [accuracy, o, y],
#     feed_dict={x: test_features, test: test_labels})
# print acc
# print y
# print output

    # print sess.run(accuracy)

# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
