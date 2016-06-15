import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
    # the size of the hidden state for the lstm
    # (notice the lstm uses 2x of this amount
    #  so actually lstm will have state of size 2)
    size = 1
    # 2 different sequences total
    batch_size = 2
    # the maximum steps for both sequences is 10
    n_steps = 10
    # each element of the sequence has dimension of 2
    seq_width = 2

    # the first input is to be stopped at 4 steps, the second at 6 steps
    e_stop = np.array([4, 6])

    initializer = tf.random_uniform_initializer(-1, 1)

    # the sequences, has n steps of maximum size
    seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
    # what timesteps we want to stop at
    # snotice it's different for each batch hence dimension of [batch]
    early_stop = tf.placeholder(tf.int32, [batch_size])

    # inputs for rnn needs to be a list, each item being a timestep.
    # we need to split our input into each timestep, and reshape it
    # because split keeps dims by default
    inputs = [tf.squeeze(i) for i in tf.split(0, n_steps, seq_input)]

    cell = tf.nn.rnn_cell.LSTMCell(
        size, seq_width, initializer=initializer, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)

    # ========= This is the most important part ==========
    # output will be of length 4 and 6
    # the state is the final state at termination (stopped at step 4 and 6)
    outputs, state = tf.nn.rnn(
        cell, inputs,
        initial_state=initial_state,
        sequence_length=early_stop
    )

    # tf.dynamic_stitch([0, 1], [early_stop - 1, [0, 1]])
    real_outputs = tf.squeeze(tf.gather(outputs, early_stop - 1))
    true_outputs = tf.pack([real_outputs[i, i] for i in range(2)])

    # usual crap
    iop = tf.initialize_all_variables()
    session = tf.Session()
    session.run(iop)
    feed = {
        early_stop: e_stop,
        seq_input: np.random.rand(
            n_steps, batch_size, seq_width).astype('float32')
    }

    print "outputs, should be 2 things one of length 4 and other of 6"
    outs = session.run(outputs, feed_dict=feed)
    print np.shape(outs)
    for xx in outs:
        print xx

    real_outs = session.run(real_outputs, feed_dict=feed)
    print "real output"
    print np.shape(real_outs)
    print real_outs
    true_outs = session.run(true_outputs, feed_dict=feed)
    print true_outs

    print "states, 2 things total both of size 2 " \
        "which is the size of the hidden state"
    st = session.run(state, feed_dict=feed)
    print st
