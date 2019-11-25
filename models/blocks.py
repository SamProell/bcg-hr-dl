import tensorflow as tf


def conv_block(prev, filters, kernel_size=3, strides=1, padding="valid",
               kernel_initializer="glorot_uniform", kernel_regularizer=None,
               separable=False, batchnorm=True, depth_multiplier=1, alpha=0.1,
               dropout=0., dilation_rate=1):
    """Conv1D layer followed by BatchNorm and LeakyReLU layer

    Args:
        prev (`keras.Layer`): previous layer (or network) for block
        filters (int): number of filters
        kernel_size (int): size of convolutional kernels
        strides (int): stride of convolutional kernels
        padding (str): padding mode
        kernel_initializer (str): initializer for Conv1D layer
        kernel_regularizer (`keras.regularizer`): regularizer for
            convolutional layer
        separable (bool): use separable convolutions
        batchnorm (bool): use BatchNormalization layer
        depth_multiplier (int): number of output channels will be
            `depth_multiplier * filters_in`
        alpha (float): slope of LeakyReLU for x < 0
        dropout (float): percentage of neurons to dropout
        dilation_rate (int): dilation rate for dilated convolution

    Returns:
        `tf.keras.layer`: output layer of convolutional block
    """

    if not separable:
        x = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size,
                                   strides=strides, padding=padding,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   dilation_rate=dilation_rate)(prev)
    else:
        x = tf.keras.layers.SeparableConv1D(filters, kernel_size=kernel_size,
                                            strides=strides, padding=padding,
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            depth_multiplier=depth_multiplier,
                                            dilation_rate=dilation_rate)(prev)

    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)

    if dropout > 0:
        x = tf.keras.layers.Dropout(rate=dropout)(x)

    return x


def dense_block(prev, units, batchnorm=True,
                kernel_initializer="glorot_uniform",
                kernel_regularizer=None):
    """Dense layer followed by BatchNorm, LeakyReLU and Dropout layer

    Args:
        prev (`keras.Layer`): input layer (preceding network) for block
        units (int): number of neurons
        batchnorm (bool): use BatchNormalization layer
        kernel_initializer (str): initializer for Dense layer
        kernel_regularizer (`keras.regularizer`): regularizer for dense
            layer

    Returns:
        `keras.layer`: output layer of dense block
    """

    x = tf.keras.layers.Dense(units, kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer)(prev)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    # x = LeakyReLU(alpha=alpha)(x)
    x = tf.keras.layers.Activation("tanh")(x)

    return x
