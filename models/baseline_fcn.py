"""Baseline convolutional architecture used for heart rate estimation
from ballistocardiographic signals.

The CNN features 6 1D-convolutional layers which reduce the input of
size 400 (8 seconds @ 50Hz) to a single output value.

`create` returns the compiled keras model implementation obtained with
`get`.  By default, Adam optimizer is used to minimize mean squared
error (MSE).

Network architecture:

    +--------+--------------+-------+------+-------------------+
    | layer  | output shape | #filt | size | additional info   |
    +--------+--------------+-------+------+-------------------+
    | input  |        400x1 |     - |    - | 8s BCG patch      |
    +--------+--------------+-------+------+-------------------+
    | conv1  |       190x8a |    8a |   21 | s=2,BN,LReLU      |
    +--------+--------------+-------+------+-------------------+
    | conv2  |       86x16a |   16a |   19 | s=2,BN,LReLU      |
    +--------+--------------+-------+------+-------------------+
    | conv3  |       36x16a |   16a |   15 | s=2,BN,LReLU      |
    +--------+--------------+-------+------+-------------------+
    | conv4  |       11x16a |   16a |   15 | s=2,BN,LReLU      |
    +--------+--------------+-------+------+-------------------+
    | conv5  |         1x8a |    8a |   11 | s=1,LReLU         |
    +--------+--------------+-------+------+-------------------+
    | conv6  |          1x1 |     1 |    1 | linear activation |
    +--------+--------------+-------+------+-------------------+
    | output |            1 |     - |    - | x_bpm = 68+100*x  |
    +--------+--------------+-------+------+-------------------+
"""

import tensorflow as tf

from .blocks import conv_block


def get(input_shape=(400, 1), kernel_regularizer=None, separable=False,
        batchnorm=True, scale_output=100., offset_output=68., alpha=0.1,
        dropout=0., enlarge=1):
    """Define the Keras implementation of the baseline CNN architecture

    Args:
        input_shape (tuple of int): input shape without batch dimension
        kernel_regularizer (`keras regularizer`): keras regularizer for
            all convolutional layers
        separable (bool): use separable convolutions
        batchnorm (bool): use Batch Normalization for first four layers
        scale_output (float): multiply output by factor
        offset_output (float): offset output
        alpha (float): slope of leaky ReLU for x < 0
        dropout (float): dropout rate for convolutional layers
        enlarge (int): increase number of filters for all layers

    Returns:
        `keras model`: model architecture
    """
    input_layer = tf.keras.layers.Input(shape=input_shape)

    x = conv_block(input_layer, 8*enlarge, 21, strides=2, padding="valid",
                   batchnorm=batchnorm, kernel_regularizer=kernel_regularizer,
                   separable=separable, alpha=alpha, dropout=dropout)

    x = conv_block(x, 16*enlarge, 19, strides=2, padding="valid",
                   batchnorm=batchnorm, kernel_regularizer=kernel_regularizer,
                   separable=separable, alpha=alpha, dropout=dropout)

    x = conv_block(x, 16*enlarge, 15, strides=2, padding="valid",
                   batchnorm=batchnorm, kernel_regularizer=kernel_regularizer,
                   separable=separable, alpha=alpha, dropout=dropout)

    x = conv_block(x, 16*enlarge, 15, strides=2, padding="valid",
                   batchnorm=batchnorm, kernel_regularizer=kernel_regularizer,
                   separable=separable, alpha=alpha, dropout=dropout)

    x = conv_block(x, 8*enlarge, 11, strides=1, padding="valid",
                   batchnorm=False, kernel_regularizer=kernel_regularizer,
                   separable=separable, alpha=alpha)

    x = tf.keras.layers.Conv1D(1, 1, activation="linear")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Lambda(lambda x_: offset_output + x_ * scale_output)(x)

    return tf.keras.models.Model(inputs=[input_layer], outputs=[x])


def create(input_shape=(400, 1), batchnorm=True, kernel_regularizer=None,
           separable=False, scale_output=100., alpha=0.1, dropout=0.,
           enlarge=1, optimizer="adam", loss="mse", metrics=()):
    """Return compiled model architecture of baseline CNN
    """
    model = get(input_shape, batchnorm=batchnorm, dropout=dropout, alpha=alpha,
                enlarge=enlarge, scale_output=scale_output, separable=separable,
                kernel_regularizer=kernel_regularizer)

    model.compile(loss=loss, optimizer=optimizer, metrics=list(metrics))

    return model
