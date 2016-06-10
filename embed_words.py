"""Load and use embeddings on dataset."""
import tensorflow as tf
import json
import codecs

# Create some variables.
vocabulary_size = 50000
embedding_size = 128  # Dimension of the embedding vector.

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
# n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 80  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 2  # total classes (positive / negative)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, embedding_size])
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

filename = 'data/corpus.txt'


def load_dictionary(path):
    """Load the dictionary for the embeddings."""
    with codecs.open(path, 'r', encoding='utf8') as f:
        obj = json.load(f)
        dictionary = obj['dictionary']
        reverse_dictionary = {
            key: value for key, value in obj['reverse_dictionary'].items()
        }

    return dictionary, reverse_dictionary


def load_data(path):
    """Load the data."""
    with codecs.open(path, 'r', encoding='utf8') as f:
        data = f.read().split('\n')
        samples = [d.split(',') for d in data[1:]]

        text = [None] * len(samples)
        vectors = [None] * len(samples)
        positive_classifications = [None] * len(samples)
        negative_classifications = [None] * len(samples)

        for i, sample in enumerate(samples):
            text[i] = sample[0]
            positive_classifications[i] = sample[1]
            negative_classifications[i] = sample[2]

            words = text[i].split(' ')
            vectors[i] = [
                dictionary[x] if x in dictionary else dictionary["UNK"]
                for x in words
            ]

        return (
            text,
            vectors,
            positive_classifications,
            negative_classifications
        )

dictionary, reverse_dictionary = load_dictionary("dictionary.json")
text, vectors, positive_classifications, negative_classifications =\
    load_data('data/training_data.csv')

# Input data.
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 2])

embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
    name='embeddings'
)

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Add ops to save and restore all the variables.
saver = tf.train.Saver({'embeddings': embeddings})
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, 'word-embeddings-basic')
    print('Model restored.')
    # Do some work with the model

    print("Optimizing..")
    step = 1
    # # Keep training until reach max iterations
    while step < training_iters:
        sess.run(
            [embed],
            feed_dict={
                train_inputs: [1] * batch_size,
                train_labels: [[0, 1]] * batch_size
            })

        print "."
