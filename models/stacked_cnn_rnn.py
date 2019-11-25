"""Stacked convolutional and LSTM network for heart rate estimation from
ballistocardiographic signals originally from [1]_.

Since the paper does not provide a complete description of the network,
some features have to be estimated. Also, because input patch size is
larger in our setting, the first convolutional layer uses a stride of 2.

Model architecture:

+----------+--------------+-------+------+----------------------------+
| layer    | output shape | #filt | size | additional info            |
+----------+--------------+-------+------+----------------------------+
| input    | 400x1        | -     | -    | input BCG patch            |
+----------+--------------+-------+------+----------------------------+
| conv1    | 200x40       | 40    | 5    | s=2, linear activation     |
+----------+--------------+-------+------+----------------------------+
| maxpool1 | 100x40       | -     | 2    | s=2                        |
+----------+--------------+-------+------+----------------------------+
| conv2    | 100x32       | 32    | 3    | s=1, linear activation     |
+----------+--------------+-------+------+----------------------------+
| maxpool2 | 50x32        | -     | 2    | s=2                        |
+----------+--------------+-------+------+----------------------------+
| LSTM 1   | 50x32        | 32    | -    | returns sequ., dropout=0.5 |
+----------+--------------+-------+------+----------------------------+
| LSTM 2   | 50x16        | 16    | -    | returns sequences          |
+----------+--------------+-------+------+----------------------------+
| LSTM 3   | 50x4         | 4     | -    |                            |
+----------+--------------+-------+------+----------------------------+
| dense    | 1            | 1     | -    | linear activation          |
+----------+--------------+-------+------+----------------------------+
| output   | 1            | -     | -    | x_bpm = 68 + 100*x         |
+----------+--------------+-------+------+----------------------------+

.. [1] Tan, Jen Hong, Yuki Hagiwara, Winnie Pang, Ivy Lim, Shu Lih Oh,
   Muhammad Adam, Ru San Tan, Ming Chen, and U. Rajendra Acharya. 2018.
   Application of Stacked Convolutional and Long Short-Term Memory
   Network for Accurate Identification of CAD ECG Signals.
   Computers in Biology and Medicine 94 (March): 19–26.
   https://doi.org/10.1016/j.compbiomed.2017.12.023.
"""
import tensorflow as tf


def get(input_shape=(400, 1), kernel_regularizer=None, separable=False,
        batchnorm=False, scale_output=100., offset_output=68., enlarge=1,
        activations=False):
    """Define the Keras implementation of the stacked CNN-LSTM

    Args:
        input_shape (tuple of int): input shape without batch dimension
        kernel_regularizer (`keras regularizer`): keras regularizer for
            all convolutional layers
        separable (bool): use separable convolutions (unused)
        batchnorm (bool): use Batch Normalization for first eight layers
            (unused here, provided for compatibility with other models)
        scale_output (float): multiply output by factor
        offset_output (float): offset output
        enlarge (int): increase number of filters for all layers
        activations (bool): use ReLU instead of linear activations in
            convolutional part

    Returns `keras model`: model architecture
    """

    input_layer = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(40*enlarge, 5, kernel_regularizer=kernel_regularizer,
                               strides=2, padding="same",
                               activation="relu" if activations else None)(input_layer)

    x = tf.keras.layers.MaxPool1D(2)(x)

    x = tf.keras.layers.Conv1D(32*enlarge, 3, kernel_regularizer=kernel_regularizer,
                               padding="same",
                               activation="relu" if activations else None)(x)
    x = tf.keras.layers.MaxPool1D(2)(x)

    x = tf.keras.layers.CuDNNLSTM(32*enlarge, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.CuDNNLSTM(16*enlarge, return_sequences=True)(x)
    x = tf.keras.layers.CuDNNLSTM(4*enlarge, return_sequences=False)(x)

    x = tf.keras.layers.Dense(1)(x)

    x = tf.keras.layers.Lambda(lambda x_: offset_output + x_ * scale_output)(x)

    return tf.keras.models.Model(inputs=[input_layer], outputs=[x])


def create(input_shape=(400, 1), batchnorm=True, kernel_regularizer=None,
           separable=False, scale_output=100., offset_output=68., enlarge=1,
           activations=False, optimizer="adam", loss="mse", metrics=()):
    """Return compiled model architecture of the stacked CNN-LSTM
    """

    model = get(input_shape, batchnorm=batchnorm, separable=separable,
                kernel_regularizer=kernel_regularizer, enlarge=enlarge,
                scale_output=scale_output, offset_output=offset_output,
                activations=activations)

    model.compile(loss=loss, optimizer=optimizer, metrics=list(metrics))

    return model
