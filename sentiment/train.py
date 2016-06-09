"""Train the Classifier using the computed word embeddings."""
import json
import codecs
import numpy as np
import tensorflow as tf


vocabulary_size = 50000
embedding_size = 128  # Dimension of the embedding vector.

unknown_tag = "UNK"

data_path = "../data/training_data.csv"
dictionary_path = "../dictionary.json"
embeddings_path = "../word-embeddings-basic"

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10
max_size = 80

# Network Parameters
n_hidden = 128  # hidden layer num of features
n_classes = 2  # total classes (Positive, Neutral, Negative)

# tf Graph input
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.float32, shape=[None, n_classes])
lengths = tf.placeholder(tf.int32, shape=[None])

# x = tf.placeholder("float", [None, max_size, embedding_size], name="x")
# y = tf.placeholder("float", [None, n_classes], name="y")
# lengths = tf.placeholder("int8", [None, 1], name="length")

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

    samples = [data.split(',') for data in text.split('\n')[1:]]
    sample_texts = [sample[0].split(' ') for sample in samples]
    sample_labels = [sample[1:1 + n_classes] for sample in samples]
    return sample_texts, sample_labels


def embed(indices, embeddings):
    """Embed into vectors."""
    embedded_indices = tf.nn.embedding_lookup(embeddings, indices)

    return embedded_indices  # .split(0, batch_size, embedded_indices)


def RNN(inputs, weights, biases, lengths=None):
    """Run rnn."""
    # Define a lstm cell with tensorflow
    # TODO CHECK THIS IS SPLITTING CORRECTLY
    x = tf.split(0, max_size, inputs)
    print "[Check split is right]"

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
        n_hidden, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(
        lstm_cell, x, dtype=tf.float32)  # , sequence_length=lengths)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


data_index = 0


def generate_batch(samples, labels, batch_size, dictionary, embeddings):
    """Generate batch data."""
    global data_index

    # Pad to batch_size * max_size
    batch_samples = np.zeros((batch_size, max_size), np.int)
    batch_labels = np.zeros((batch_size, n_classes), np.float32)
    early_stop = np.zeros(batch_size, np.int)

    unknown_index = dictionary[unknown_tag]
    for i, sample in enumerate(
            samples[data_index: data_index + batch_size]):

        early_stop[i] = len(sample)
        for j, word in enumerate(sample):
            batch_samples[i][j] = dictionary.get(
                word, unknown_index)

    for i, sample_labels in enumerate(
            labels[data_index: data_index + batch_size]):
        batch_labels[i] = np.array(sample_labels, np.float32)

    return batch_samples, batch_labels, early_stop


print "Loading dictionary..."
dictionary, reverse_dictionary = load_embedding_dictionary(dictionary_path)

embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
    name='embeddings'
)

print "Loading data..."
samples, labels = load_training_data(data_path)

print "Building graph..."
x = embed(train_inputs, embeddings)
pred = RNN(x, weights, biases, lengths)

# Define loss and optimizer
cost = tf.nn.l2_loss(pred - train_labels)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(train_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Add ops to save and restore all the variables.
print "Loading embeddings..."
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

    print("Optimizing..")
    step = 1
    # # Keep training until reach max iterations),
    while step < training_iters:
        print "Step"
        batch_x, batch_y, early_stop = generate_batch(
            samples, labels, batch_size, dictionary, embeddings)

        batch_x = batch_x.reshape((-1))

        feed_dict = {
            train_inputs: batch_x,
            train_labels: batch_y,
            lengths: early_stop
        }

        sess.run(optimizer, feed_dict=feed_dict)

        if True:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict=feed_dict)
            # Calculate batch loss
            loss = sess.run(cost, feed_dict=feed_dict)
            print "Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1

    print "Optimization Finished!"

    # # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print "Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label})
