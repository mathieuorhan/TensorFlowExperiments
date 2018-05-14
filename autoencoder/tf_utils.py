#!/usr/bin/env python
import numpy as np
import tensorflow as tf

def form_image_grid(input_tensor, grid_shape, image_shape, num_channels):
    """Arrange a minibatch of images into a grid to form a single image.
    Args:
        input_tensor: Tensor. Minibatch of images to format, either 4D
            ([batch size, height, width, num_channels]) or flattened
            ([batch size, height * width * num_channels]).
        grid_shape: Sequence of int. The shape of the image grid,
            formatted as [grid_height, grid_width].
        image_shape: Sequence of int. The shape of a single image,
            formatted as [image_height, image_width].
        num_channels: int. The number of channels in an image.
    Returns:
        Tensor representing a single image in which the input images have been
        arranged into a grid.
    Raises:
        ValueError: The grid shape and minibatch size don't match, or the image
            shape and number of channels are incompatible with the input tensor.
    """
    if grid_shape[0] * grid_shape[1] != int(input_tensor.get_shape()[0]):
        raise ValueError('Grid shape incompatible with minibatch size.')
    if len(input_tensor.get_shape()) == 2:
        num_features = image_shape[0] * image_shape[1] * num_channels
        if int(input_tensor.get_shape()[1]) != num_features:
            raise ValueError('Image shape and number of channels incompatible with '
                        'input tensor.')
    elif len(input_tensor.get_shape()) == 4:
        if (int(input_tensor.get_shape()[1]) != image_shape[0] or
            int(input_tensor.get_shape()[2]) != image_shape[1] or
            int(input_tensor.get_shape()[3]) != num_channels):
            raise ValueError('Image shape and number of channels incompatible with '
                        'input tensor.')
    else:
        raise ValueError('Unrecognized input tensor format.')
    height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
    input_tensor = tf.reshape(
        input_tensor, grid_shape + image_shape + [num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2, 4])
    input_tensor = tf.reshape(
        input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
    input_tensor = tf.reshape(
        input_tensor, [1, height, width, num_channels])
    return input_tensor


def weight_variable(shape):
    # From the mnist tutorial
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def fc_layer(previous, input_size, output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(previous, W) + b