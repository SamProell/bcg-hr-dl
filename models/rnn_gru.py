"""Recurrent architecture used for heart rate estimation from
ballistocardiographic signals.

The RNN features 3 layers with gated recurrent units (GRUs) to obtain a
single output value for an arbitrary lenght input stream.

`create` returns the compiled keras model implementation obtained with
`get`.  By default, Adam optimizer is used to minimize mean squared
error (MSE).

Network architecture:

    +--------+--------------+-------+------+-----------------------+
    | layer  | output shape | #filt | size | additional info       |
    +--------+--------------+-------+------+-----------------------+
    | input  |          nx1 |     - |    - | variable length patch |
    +--------+--------------+-------+------+-----------------------+
    | GRU 1  |         nx8a |    8a |    - | returns sequence      |
    +--------+--------------+-------+------+-----------------------+
    | GRU 2  |        nx16a |   16a |    - | returns sequence      |
    +--------+--------------+-------+------+-----------------------+
    | GRU 3  |          32a |   32a |    - |                       |
    +--------+--------------+-------+------+-----------------------+
    | dense  |            1 |     1 |    - | linear activation     |
    +--------+--------------+-------+------+-----------------------+
    | output |            1 |     - |    - | x_bpm = 68+100*x      |
    +--------+--------------+-------+------+-----------------------+
"""
import tensorflow as tf


def get(input_shape=(None, 1), scale_output=100., offset_output=68., enlarge=1,
        use_lstm=False):
    """Define the Keras implementation of the deeper CNN architecture

    Args:
        input_shape (tuple of int): input shape without batch dimension
        scale_output (float): multiply output by factor
        offset_output (float): offset output
        enlarge (int): increase number of filters for all layers
        use_lstm (bool): use LSTM cells instead of GRUs

    Returns:
        `keras model`: model architecture
    """

    patchin = tf.keras.layers.Input(shape=input_shape)

    if use_lstm:
        x = tf.keras.layers.CuDNNLSTM(8*enlarge, return_sequences=True)(patchin)
        x = tf.keras.layers.CuDNNLSTM(16*enlarge, return_sequences=True)(x)
        x = tf.keras.layers.CuDNNLSTM(32*enlarge, return_sequences=False)(x)
    else:
        x = tf.keras.layers.CuDNNGRU(8*enlarge, return_sequences=True)(patchin)
        x = tf.keras.layers.CuDNNGRU(16*enlarge, return_sequences=True)(x)
        x = tf.keras.layers.CuDNNGRU(32*enlarge, return_sequences=False)(x)

    x = tf.keras.layers.Dense(1, activation="linear")(x)

    x = tf.keras.layers.Lambda(lambda x_: offset_output + x_ * scale_output)(x)

    return tf.keras.models.Model(inputs=[patchin], outputs=[x])


def create(input_shape=(None, 1), scale_output=100., offset_output=68.,
           enlarge=1, use_lstm=False, optimizer="adam", loss="mse", metrics=()):
    """Return compiled model architecture of recurrent network
    """

    model = get(input_shape=input_shape, scale_output=scale_output,
                offset_output=offset_output, enlarge=enlarge, use_lstm=use_lstm)

    model.compile(loss=loss, optimizer=optimizer, metrics=list(metrics))

    return model
