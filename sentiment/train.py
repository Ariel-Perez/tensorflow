"""Train the Classifier using the computed word embeddings."""
import json
import codecs
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell


vocabulary_size = 50000
embedding_size = 128  # Dimension of the embedding vector.

unknown_tag = "UNK"

data_path = "../data/small_corpus.txt"
dictionary_path = "../dictionary.json"
embeddings_path = "../word_embeddings"

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10
max_size = 80

# Network Parameters
n_hidden = 128  # hidden layer num of features
n_classes = 2  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, None, embedding_size])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([embedding_size, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def load_embedding_dictionary(path):
    """Load the dictionary with the word-index mappings."""
    with codecs.open(path, 'r', encoding='utf8') as f:
        obj = json.load(f)

    dictionary = obj['dictionary']
    reverse_dictionary = {
        int(k): v for k, v in
        obj['reverse_dictionary'].items()
    }

    return dictionary, reverse_dictionary


def load_training_data(path):
    """Load the data for training."""
    with codecs.open(path, 'r', encoding='utf8') as f:
        text = f.read()

    samples = text.split(' . ')
    return [sample.split(' ') for sample in samples]


def embed_text(text_samples, embeddings, dictionary):
    """Embed text into vectors."""
    # indices = [[dictionary.get(word, dictionary[unknown_tag])
    #             for word in sample] for sample in text_samples]
    indices = [None] * len(text_samples)
    for i, sample in enumerate(text_samples):
        indices[i] = [dictionary[unknown_tag]] * max_size
        for j, word in enumerate(sample):
            if word in dictionary:
                indices[i][j] = dictionary[word]

    # indices = []
    # for sample in text_samples:
    #     indices.extend([dictionary.get(word, dictionary[unknown_tag])
    #                     for word in sample])

    early_stop = [len(sample) for sample in text_samples]

    embed = [tf.nn.embedding_lookup(embeddings, sample_indices)
             for sample_indices in indices]
    # embed = tf.nn.embedding_lookup(embeddings, indices)
    # print embed.get_shape()
    return embed, early_stop


def RNN(x, weights, biases, lengths=None):
    """Run rnn."""
    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(
        lstm_cell, x, dtype=tf.float32, sequence_length=lengths)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


dictionary, reverse_dictionary = load_embedding_dictionary(dictionary_path)

embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
    name='embeddings'
)
samples = load_training_data(data_path)
embed, early_stop = embed_text(samples, embeddings, dictionary)

pred = RNN(embed, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Add ops to save and restore all the variables.
saver = tf.train.Saver({
    'embeddings': embeddings
})


# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
init = tf.initialize_all_variables()
with tf.Session() as sess:
    # Restore variables from disk.
    sess.run(init)
    saver.restore(sess, embeddings_path)
    print("Model restored.")

    # step = 1
    # # Keep training until reach max iterations
    # while step * batch_size < training_iters:
    #     batch_x, batch_y = mnist.train.next_batch(batch_size)
    #     print np.shape(batch_x), np.shape(batch_y)
    #     # Reshape data to get 28 seq of 28 elements
    #     batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    #     # Run optimization op (backprop)
    #     sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
    #     if step % display_step == 0:
    #         # Calculate batch accuracy
    #         acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
    #         # Calculate batch loss
    #         loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
    #         print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
    #               "{:.6f}".format(loss) + ", Training Accuracy= " + \
    #               "{:.5f}".format(acc)
    #     step += 1
    # print "Optimization Finished!"

    # # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print "Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label})
