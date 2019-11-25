"""Dispersion maximum method introduced by Choe & Cho (2017)

S. T. Choe and W. D. Cho, 'Simplified real-time heartbeat detection
in ballistocardiography using a dispersion-maximum method,' Biomed.
Res. India, vol. 28, no. 9, pp. 3974-3985, 2017
"""

import numpy as np
import pandas as pd


def moving_average(x, f, window_length=0.05):
    """Smooth signal with a moving average with given window length
    """

    winsize = int(window_length * f)

    return pd.Series(x).rolling(winsize, min_periods=1,
                                center=True).mean().values


def moving_absolute_deviation(x, f, window_length=0.05):
    """Calculate moving absolute deviation (MAD) from raw signal

    The original paper suggests a `window_length` of 0.05.  However,
    we find 0.15 seconds more appropriate for our signals, although the
    resulting indices lie in the transition from J to K.

    Args:
        x (`1d array-like`): raw signal
        f (float): in Hz; sampling rate of signal
        window_length (float): in seconds; window length for MAD
            calculation

    Returns:
        `1d array`: moving absolute deviation signal
    """

    winsize = int(window_length * f)
    a = moving_average(x, f, window_length=window_length)
    d = pd.Series(np.subtract(x, a)).abs().rolling(winsize, min_periods=1,
                                                   center=True).mean().values

    return d


def moving_maximum(x, f, window_length=0.4):
    """Calculate moving maximum signal with given window length
    """
    winsize = int(window_length * f)

    return pd.Series(x).rolling(winsize, min_periods=1, center=False).max()


def detect_peaks(x, f, window_length=0.4):
    """Detect peaks in dispersion maximum signal by searching for peaks
    that remain the largest ones for at least `window_length` seconds

    Args:
        x (`1d array-like`): moving maximum signal
        f (float): in Hz; sampling rate of signal
        window_length (float): in s; window length used for rolling max.

    Returns:
        `1d array`: peak location indices
    """
    winsize = int(window_length * f)
    timer = np.ones(len(x))

    for i in range(1, len(timer)):
        if x[i-1] == x[i]:
            timer[i] = timer[i-1] + 1

    return np.arange(len(timer))[timer == winsize] - winsize + 1


def choe(x, f, mad_window_length=0.15, max_window_length=0.4):
    """Dispersion maximum method for heart beat detection in BCG signals
    by Choe & Cho (2017)

    Args:
        x (`1d array-like`): BCG signal
        f (float): in Hz; sampling rate of BCG signal
        mad_window_length (float): in s; window length for moving
            absolute deviation calculation
        max_window_length (float): in s; window length for moving
            maximum calculation and peak search

    Returns:
        `1d array`: indices of heart beat locations
    """
    mad_signal = moving_absolute_deviation(x, f, mad_window_length)
    max_signal = moving_maximum(mad_signal, f, max_window_length)

    indices = detect_peaks(max_signal, f, max_window_length)

    return indices
