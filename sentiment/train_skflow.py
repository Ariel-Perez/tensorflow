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
training_iters = 100
batch_size = 8
display_step = 10
max_size = 80

# Network Parameters
n_hidden = 128  # hidden layer num of features
n_classes = 2  # total classes (Positive, Neutral, Negative)

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
    sample_labels = [[sample[1], sample[3]] for sample in samples]
    return sample_texts, sample_labels


def embed(indices, embeddings):
    """Embed into vectors."""
    embedded_vectors = tf.nn.embedding_lookup(embeddings, [[0, 16], [6, 14]])
    return embedded_vectors  # .split(0, batch_size, embedded_indices)


data_index = 0


def get_indices(samples, labels, dictionary):
    """Generate batch data."""
    n = len(samples)
    batch_samples = np.zeros((n, max_size), np.int)
    batch_labels = np.zeros((n, n_classes), np.float32)
    early_stop = np.zeros(n, np.int)

    unknown_index = dictionary[unknown_tag]
    for i in range(n):
        sample = samples[i]
        sample_labels = labels[i]

        early_stop[i] = len(sample)
        for j, word in enumerate(sample):
            batch_samples[i][j] = dictionary.get(word, unknown_index)

        batch_labels[i] = np.array(sample_labels, np.float32)

    return batch_samples, batch_labels, early_stop


def normalize(x):
    """Normalize the data."""
    mean, variance = tf.nn.moments(x, [0])
    return (x - mean) / variance

print "Loading dictionary..."
dictionary, reverse_dictionary = load_embedding_dictionary(dictionary_path)

embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
    name='embeddings'
)

print "Loading data..."
samples, labels = load_training_data(data_path)

print "Building graph..."
samples, labels, early_stop = get_indices(samples, labels, dictionary)
for i in range(3):
    print early_stop[i], list(samples[i][:early_stop[i]])
embedded_input = embed(samples, embeddings)

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
    print("Embeddings restored.")

    x = sess.run(embedded_input)
    print x
