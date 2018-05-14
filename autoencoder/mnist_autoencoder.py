#!/usr/bin/env python

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from utils import add_noise
from tf_utils import form_image_grid, weight_variable, bias_variable, fc_layer

BATCH_SIZE = 50
GRID_ROWS = 5
GRID_COLS = 10
USE_RELU = True

L0_SIZE = 28*28
L1_SIZE = 50
L2_SIZE = 50
LATENT_SIZE = 30
N_BATCH = 200001
NOISE = "mask-0.9"

def autoencoder(x, x_original):
    # first fully connected layer with L1_SIZE neurons using tanh activation
    l1 = tf.nn.tanh(fc_layer(x, L0_SIZE, L1_SIZE))
    # second fully connected layer with 50 neurons using tanh activation
    l2 = tf.nn.tanh(fc_layer(l1, L1_SIZE, L2_SIZE))
    # third fully connected layer with 2 neurons
    l3 = fc_layer(l2, L2_SIZE, LATENT_SIZE)
    # fourth fully connected layer with 50 neurons and tanh activation
    l4 = tf.nn.tanh(fc_layer(l3, LATENT_SIZE, L2_SIZE))
    # fifth fully connected layer with 50 neurons and tanh activation
    l5 = tf.nn.tanh(fc_layer(l4, L2_SIZE, L1_SIZE))
    # readout layer
    if USE_RELU:
        out = tf.nn.relu(fc_layer(l5, L1_SIZE, L0_SIZE))
    else:
        out = fc_layer(l5, L1_SIZE, L0_SIZE)
    # let's use an l2 loss on the output image
    loss = tf.reduce_mean(tf.squared_difference(x_original, out))
    return loss, out, l3


def layer_grid_summary(name, var, image_dims):
    prod = np.prod(image_dims)
    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod]), [GRID_ROWS, 
        GRID_COLS], image_dims, 1)
    return tf.summary.image(name, grid)


def create_summaries(loss, x, latent, output):
    writer = tf.summary.FileWriter("./logs")
    tf.summary.scalar("Loss", loss)
    layer_grid_summary("Input", x, [28, 28])
    layer_grid_summary("Encoder", latent, [LATENT_SIZE, 1])
    layer_grid_summary("Output", output, [28, 28])
    return writer, tf.summary.merge_all()


def make_image(name, var, image_dims):
    prod = np.prod(image_dims)
    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod]), [GRID_ROWS, 
        GRID_COLS], image_dims, 1)
    s_grid = tf.squeeze(grid, axis=0)

    # This reproduces the code in: tensorflow/core/kernels/summary_image_op.cc
    im_min = tf.reduce_min(s_grid)
    im_max = tf.reduce_max(s_grid)

    kZeroThreshold = tf.constant(1e-6)
    max_val = tf.maximum(tf.abs(im_min), tf.abs(im_max))

    offset = tf.cond(
            im_min < tf.constant(0.0),
            lambda: tf.constant(128.0),
            lambda: tf.constant(0.0)
            )
    scale = tf.cond(
            im_min < tf.constant(0.0),
            lambda: tf.cond(
                max_val < kZeroThreshold,
                lambda: tf.constant(0.0),
                lambda: tf.div(127.0, max_val)
                ),
            lambda: tf.cond(
                im_max < kZeroThreshold,
                lambda: tf.constant(0.0),
                lambda: tf.div(255.0, im_max)
                )
            )
    s_grid = tf.cast(tf.add(tf.multiply(s_grid, scale), offset), tf.uint8)
    enc = tf.image.encode_jpeg(s_grid)

    fwrite = tf.write_file(name, enc)
    return fwrite


def main():
    # initialize the data
    mnist = input_data.read_data_sets('/tmp/MNIST_data')

    # placeholders for the images
    x = tf.placeholder(tf.float32, shape=[None, L0_SIZE])
    x_original = tf.placeholder(tf.float32, shape=[None, L0_SIZE])

    # build the model
    loss, output, latent = autoencoder(x, x_original)

    # and we use the Adam Optimizer for training
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # We want to use Tensorboard to visualize some stuff
    writer, summary_op = create_summaries(loss, x, latent, output)

    first_batch = mnist.test.next_batch(BATCH_SIZE)
    first_batch = (add_noise(first_batch[0], NOISE),first_batch[1])

    # Run the training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(make_image("images/input.jpg", x, [28, 28]), feed_dict={x : 
            first_batch[0]})
        for i in range(int(N_BATCH)):
            batch_original = mnist.train.next_batch(BATCH_SIZE)       
            batch = (add_noise(batch_original[0], NOISE),batch_original[1])

            feed = {x : batch[0], x_original: batch_original[0]}
            if i % 500 == 0:
                summary, train_loss = sess.run([summary_op, loss], 
                        feed_dict=feed)
                print("step %d, training loss: %g" % (i, train_loss))

                writer.add_summary(summary, i)
                writer.flush()

            if i % 1000 == 0:
                sess.run(make_image("images/output_%06i.jpg" % i, output, [28, 
                    28]), feed_dict={x : first_batch[0]})

            train_step.run(feed_dict=feed)

        # Save latent space
        pred = sess.run(latent, feed_dict={x : mnist.test._images})
        pred = np.asarray(pred)
        pred = np.reshape(pred, (mnist.test._num_examples, 2))
        labels = np.reshape(mnist.test._labels, (mnist.test._num_examples, 1))
        pred = np.hstack((pred, labels))
        if USE_RELU:
            fname = "latent_relu.csv"
        else:
            fname = "latent_default.csv"
        np.savetxt(fname, pred, delimiter=",", fmt="%1.5f")


if __name__ == '__main__':
    main()
