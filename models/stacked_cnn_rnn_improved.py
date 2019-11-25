"""Modified stacked CNN-GRU architecture for heart rate estimation from
ballistocardiographic signals.


The network features four convolutional layers followed by 4 recurrent
layers with gated recurrent units (GRUs).  The convolutional part
reduces the patch size from 400 (eight seconds @ 50Hz) to 21 samples.
This results in significantly smaller computational load of the
subsequent recurrent part.

Model architecture:

    +--------+--------------+-------+------+------------------------+
    | layer  | output shape | #filt | size | additional info        |
    +--------+--------------+-------+------+------------------------+
    | input  |        400x1 |     - |    - | eight second BCG patch |
    +--------+--------------+-------+------+------------------------+
    | conv1  |      196x64a |   64a |    9 | s=2, BN, LReLU         |
    +--------+--------------+-------+------+------------------------+
    | conv2  |       96x32a |   32a |    5 | s=2, BN, LReLU         |
    +--------+--------------+-------+------+------------------------+
    | conv3  |       46x32a |   32a |    5 | s=2, BN, LReLU         |
    +--------+--------------+-------+------+------------------------+
    | conv4  |       21x32a |   32a |    5 | s=2, BN, LReLU         |
    +--------+--------------+-------+------+------------------------+
    | GRU 1  |       21x32a |   32a |    - | returns sequence       |
    +--------+--------------+-------+------+------------------------+
    | GRU 2  |       21x16a |   16a |    - | returns sequence       |
    +--------+--------------+-------+------+------------------------+
    | GRU 3  |        21x8a |    8a |    - | returns sequence       |
    +--------+--------------+-------+------+------------------------+
    | GRU 4  |           4a |    4a |    - |                        |
    +--------+--------------+-------+------+------------------------+
    | dense  |            1 |     1 |    - | linear activation      |
    +--------+--------------+-------+------+------------------------+
    | output |            1 |     - |    - | x_bpm = 68 + 100*x     |
    +--------+--------------+-------+------+------------------------+
"""

import tensorflow as tf

from .blocks import conv_block


def get(input_shape=(400, 1), kernel_regularizer=None, separable=False,
        batchnorm=True, scale_output=100., offset_output=68., enlarge=1,
        alpha=0.1):
    """Define the Keras implementation of the modified stacked CNN-LSTM

    Args:
        input_shape (tuple of int): input shape without batch dimension
        kernel_regularizer (`keras regularizer`): keras regularizer for
            all convolutional layers
        separable (bool): use separable convolutions (unused here,
            provided for compatibility with other models)
        batchnorm (bool): use Batch Normalization for first eight layers
        scale_output (float): multiply output by factor
        offset_output (float): offset output
        enlarge (int): increase number of filters for all layers
        alpha (float): slope of leaky ReLU for x < 0

    Returns `keras model`: model architecture
    """

    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = conv_block(input_layer, int(64*enlarge), 9, strides=2, padding="valid",
                   kernel_regularizer=kernel_regularizer, batchnorm=batchnorm,
                   alpha=alpha)
    x = conv_block(x, int(32*enlarge), 5, strides=2, padding="valid",
                   kernel_regularizer=kernel_regularizer, alpha=alpha)
    x = conv_block(x, int(32*enlarge), 5, strides=2, padding="valid",
                   kernel_regularizer=kernel_regularizer, alpha=alpha)
    x = conv_block(x, int(32*enlarge), 5, strides=2, padding="valid",
                   kernel_regularizer=kernel_regularizer, alpha=alpha)

    x = tf.keras.layers.CuDNNGRU(int(32*enlarge), return_sequences=True)(x)
    x = tf.keras.layers.CuDNNGRU(int(16*enlarge), return_sequences=True)(x)
    x = tf.keras.layers.CuDNNGRU(int(8*enlarge), return_sequences=True)(x)
    x = tf.keras.layers.CuDNNGRU(int(4*enlarge), return_sequences=False)(x)

    x = tf.keras.layers.Dense(1)(x)

    x = tf.keras.layers.Lambda(lambda x_: offset_output + x_ * scale_output)(x)

    return tf.keras.models.Model(inputs=[input_layer], outputs=[x])


def create(input_shape=(400, 1), batchnorm=True, kernel_regularizer=None,
           separable=False, scale_output=100., offset_output=68., enlarge=1,
           optimizer="adam", loss="mse", metrics=()):
    """Return compiled model architecture of the stacked CNN-LSTM
    """

    model = get(input_shape, batchnorm=batchnorm, separable=separable,
                kernel_regularizer=kernel_regularizer, enlarge=enlarge,
                scale_output=scale_output, offset_output=offset_output)

    model.compile(loss=loss, optimizer=optimizer, metrics=list(metrics))

    return model
