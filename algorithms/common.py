
import numpy as np
import scipy.signal as sgnl


def filter_bandpass(x, f, cutoff1, cutoff2, order=2):
    """Filter signal forwards and backwards with Butterworth bandpass
    """
    coeffs = sgnl.butter(N=order, Wn=np.divide([cutoff1, cutoff2],
                                               f/2.), btype="bandpass")

    return sgnl.filtfilt(coeffs[0], coeffs[1], x)


def filter_lowpass(x, f, cutoff, order=2):
    """Filter signal forwards and backwards with Butterworth lowpass
    """
    coeffs = sgnl.butter(N=order, Wn=np.divide(cutoff, f/2.),
                         btype="lowpass")

    return sgnl.filtfilt(coeffs[0], coeffs[1], x)


def get_padded_window(x, i, n, nafter=None, padding_value=0.):
    """Get padded window from signal at specified location

    Args:
        x (`np.ndarray`): signal from which to extract window
        i (int): index location of window
        n (int): window size in samples (if asymmetric: number of
            samples before specified index)
        nafter (int): number of samples after specified index if
            asymmetric window
        padding_value (float or list): value used for padding at start
            and end of signal.  If None, nearest neighbor is used.

    Returns:
        `np.ndarray`: padded window within signal
    """

    x = np.asarray(x)

    # check inputs
    if i < 0 or i >= len(x):
        raise IndexError("Index %d out of range for array with length %d" %
                         (i, len(x)))
    if len(x.shape) not in {1, 2}:
        raise ValueError("x has to be 1- or 2-dimensional")
    if (len(x.shape) == 2 and hasattr(padding_value, "__len__") and
            len(padding_value) != x.shape[1]):
        raise ValueError("padding_value must be a single float or of same "
                         "length as x")

    # calculate left and right margins
    if not nafter:
        nbefore = int(np.ceil(n/2.))
        nafter = int(n // 2)
    else:
        nbefore = n

    # determine left and right padding values
    if padding_value is None:
        left_pad = x[max(0, i-nbefore)]
        right_pad = x[min(i+nafter, len(x))-1]
    else:
        left_pad = right_pad = padding_value
    left_pad = [left_pad]*max(0, nbefore - i)
    right_pad = [right_pad]*max(0, i+nafter-len(x))
    if len(x.shape) == 2:
        left_pad = np.reshape(left_pad, (-1, x.shape[1]))
        right_pad = np.reshape(right_pad, (-1, x.shape[1]))

    return np.concatenate([left_pad,
                           x[max(0, i-nbefore):min(i+nafter, len(x))],
                           right_pad])
