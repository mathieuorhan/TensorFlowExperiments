#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns; sns.set()
import os


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
LATENT_SIZE = 2
N_BATCH = 200001
NOISE = "mask-0.0"
NICE_NAME = "Autoencoder reconstruction with latent space of dimension 2"
RUN_NAME = "autoencoder_dim2_reconstruct"
os.makedirs("./logs/"+RUN_NAME+"/train")
os.makedirs("./logs/"+RUN_NAME+"/test")

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
    # let's use an l2 loss on the output image and the image WITHOUT noise
    loss = tf.reduce_mean(tf.squared_difference(x_original, out))
    return loss, out, l3


def acp(x, x_original):
    # A linear autoencoder is equivalent to a PCA

    # first fully connected layer, linear
    l1 = fc_layer(x, L0_SIZE, LATENT_SIZE)

    # readout layer
    if USE_RELU:
        out = tf.nn.relu(fc_layer(l1, LATENT_SIZE, L0_SIZE))
    else:
        out = fc_layer(l1, LATENT_SIZE, L0_SIZE)
    # let's use an l2 loss on the output image and the image WITHOUT noise
    loss = tf.reduce_mean(tf.squared_difference(x_original, out))
    return loss, out, l1


def layer_grid_summary(name, var, image_dims):
    prod = np.prod(image_dims)
    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod]), [GRID_ROWS, 
        GRID_COLS], image_dims, 1)
    return tf.summary.image(name, grid)



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
    # x_original is the image without noise, and x with noise
    x = tf.placeholder(tf.float32, shape=[None, L0_SIZE])
    x_original = tf.placeholder(tf.float32, shape=[None, L0_SIZE])

    # build the model (acp | autoencoder)
    loss, output, latent = autoencoder(x, x_original)

    # and we use the Adam Optimizer for training
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # We want to use Tensorboard to visualize some stuff

    ref_batch = mnist.train.next_batch(BATCH_SIZE)
    ref_batch = (add_noise(ref_batch[0], NOISE),ref_batch[1])

    # Run the training loop
    with tf.Session() as sess:
        tf.summary.scalar("Loss", loss)
        layer_grid_summary("Original", x_original, [28, 28])
        layer_grid_summary("Input", x, [28, 28])
        layer_grid_summary("Output", output, [28, 28])
        merged = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter("./logs/"+RUN_NAME+"/train",sess.graph)
        writer_test = tf.summary.FileWriter("./logs/"+RUN_NAME+"/test",sess.graph)

        sess.run(tf.global_variables_initializer())
        sess.run(make_image("images/input.jpg", x, [28, 28]), feed_dict={x : 
            ref_batch[0]})
        for i in range(int(N_BATCH)):
                    
            batch_original_train = mnist.train.next_batch(BATCH_SIZE)       
            batch_train = (add_noise(batch_original_train[0], NOISE),batch_original_train[1])
            feed_dict_train = {x : batch_train[0], x_original: batch_original_train[0]}

            if i % 500 == 0:

                batch_original_test = mnist.test.next_batch(BATCH_SIZE)       
                batch_test = (add_noise(batch_original_test[0], NOISE),batch_original_test[1])

                feed_dict_test = {x : batch_test[0], x_original: batch_original_test[0]}
                
                summary_str_train, train_error = sess.run(fetches=[merged, loss], feed_dict=feed_dict_train)
                summary_str_test,  test_error = sess.run(fetches=[merged, loss], feed_dict=feed_dict_test)
                
                print("step %d, train loss:%g, test loss: %g" % (i, train_error, test_error))

                writer_train.add_summary(summary_str_train, i)
                writer_test.add_summary(summary_str_test, i)
                
                writer_train.flush()
                writer_test.flush()
                

            """if i % 1000 == 0:
                sess.run(make_image("images/output_%06i.jpg" % i, output, [28, 
                    28]), feed_dict={x : ref_batch[0]})
            """

            train_step.run(feed_dict=feed_dict_train)

        # Visualize latent space
        if LATENT_SIZE==2:
            pred = sess.run(latent, feed_dict={x : mnist.test._images})
            pred = np.asarray(pred)
            pred = np.reshape(pred, (mnist.test._num_examples, 2))
            labels = np.reshape(mnist.test._labels, (mnist.test._num_examples, 1))

            plt.figure(figsize=(4,4))
            plt.subplot(111)
            plt.title(NICE_NAME)
            color_map = ListedColormap(sns.color_palette("hls", 10))
            plt.scatter(pred[:5000,0], pred[:5000,1], c=labels[:5000], s=8, cmap=color_map)
            plt.gca().get_xaxis().set_ticklabels([])
            plt.gca().get_yaxis().set_ticklabels([])

            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    main()
