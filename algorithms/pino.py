"""J wave detection by Pino et al. (2015)

E. J. Pino, J. A. Chavez, and P. Aqueveque, 'Noninvasive ambulatory
measurement system of cardiac activity,' in Conf. Proc. IEEE Eng.
Med. Biol. Soc., 2015, pp. 7622-7625
"""

import numpy as np
import scipy
import scipy.ndimage
import scipy.signal as sgnl
import pandas as pd
import pywt

from . import common


def wavelet_signal_separation(x, wavelet="db6", levels=8,
                              details=(3, 4, 5, 6)):
    """Extract BCG signal from raw data using wavelet decomposition

    The paper by Pino et al. suggests to keep details 4 to 7 which
    corresponds to indices 3-6.

    Args:
        x (`1d array`): raw signal
        wavelet (str): wavelet mother to use
        levels (int): number of levels
        details (tuple(int)): indices of levels to keep

    Returns:
        `1d array`: extracted BCG signal
    """

    coeffs = pywt.wavedec(x, wavelet, level=levels)
    # set all unused details to 0
    for i in range(levels):
        if i not in details:
            coeffs[i][:] = 0.

    return pywt.waverec(coeffs, wavelet)


def length_transform(x, f, window_length=0.3, center=True):
    """Apply length transform to preprocessed signal

    Length of returned signal is 1 less than original due to difference
    operation.

    Args:
        x (`1d array-like`): preprocessed BCG signal
        f (float): in Hz; sampling rate of signal
        window_length (float): in seconds; length of moving window
        center (bool): center window to avoid relocation later

    Returns:
        `1d array`: length transform of given signal
    """

    winsize = int(f * window_length)
    xs = pd.Series(np.sqrt((x[1:] - x[:-1]) ** 2 + 1))

    return xs.rolling(winsize, min_periods=1, center=center).sum().values


def smoothing(x, f, window_length=0.3):
    """Apply smoothing with moving average window

    Args:
        x (`1d array-like`): signal
        f (float): in Hz; sampling rate of signal
        window_length (float): in seconds; length of moving window

    Returns:
        `1d array`: smoothed signal
    """
    winsize = int(f * window_length)

    return scipy.ndimage.convolve1d(x, np.divide(np.ones(winsize), winsize),
                                    mode="nearest")


def first_elimination(lt, f, indices, window_length=0.3):
    """Eliminate peaks that are not true maxima within a small window

    Args:
        lt (`1d array-like`): length-transformed signal
        f (float): sampling rate of input signal
        indices (`list(int)`): list of detected peaks
        window_length (float): in seconds; window length for maximum
            search

    Returns:
        `list(int)`: list of filtered peak indices
    """

    def is_maximum(i):
        winmax = common.get_padded_window(lt, i, int(f*window_length)).max()
        return winmax > lt[i]

    # first elimination
    return list(filter(lambda i: not is_maximum(i), indices))


def relocate_indices(x, f, indices, search_window=0.4):
    """Refine peak locations to adjust for small errors length-transform
    calculation

    Args:
        x (`1d array-like`): preprocessed BCG signal
        f (float): in Hz; sampling rate of input signal
        indices (`list(int)`): list of detected peak locations
        search_window (float): in seconds; window length for peak
            correction

    Returns:
        `list(int)`: refined J peak locations
    """

    winsize = int(f*search_window)
    js = indices[:]
    for i, ind in enumerate(indices):
        js[i] = (ind - winsize // 2
                 + np.argmax(common.get_padded_window(x, ind, winsize)))

    return js


def second_elimination(bcg, f, indices, dist=0.3):
    """Discard J wave locations that are too close to each other

    Args:
        bcg (`1d array-like`): preprocessed BCG signal
        f (float): in Hz; sampling rate of input signal
        indices (`1d array-like`): list of detected peak indices
        dist (float): in seconds; minimum distance between peaks

    Returns:
        `list(int)`: list of filtered peak indices
    """

    dist = int(f * dist)
    inds = indices[:]
    i = 1
    while i < len(inds):
        if inds[i] - inds[i-1] <= dist:
            if bcg[inds[i]] > bcg[inds[i-1]]:
                del inds[i]
            else:
                del[inds[i-1]]
        else:
            i += 1
    return inds


def pino(x, f, low_cutoff=30., lt_window=0.3, smoothing_window=0.3,
         order=2, mother="db6", levels=8, details=(4, 5, 6, 7),
         elimination_window=0.6, search_window=0.4, min_dist=0.3):
    """J wave detection by Pino et al. (2015)

    Args:
        x (`1d array-like`): raw BCG signal
        f (float): in Hz; sampling rate of input signal
        low_cutoff (float): in Hz; cutoff frequency of lowpass filter
        lt_window (float): in seconds; window size for length-transform
        smoothing_window (float): in seconds; window size for moving
            average
        order (int): order of Butterworth lowpass filter
        mother (str): wavelet base
        levels (int): number of levels for wavelet decomposition
        details (int): number of details for wavelet decomposition
        elimination_window (float): in seconds; window length for first
            elimination
        search_window (float): in seconds; window length for peak
            refinement
        min_dist (float): in seconds; minimum distance of peaks

    Returns:
        `list(int)`: list of detected J wave locations
    """

    # filter with 30Hz lowpass
    x = common.filter_lowpass(x, f, low_cutoff, order=order)

    # separate BCG signal:
    bcg = wavelet_signal_separation(x, wavelet=mother, levels=levels,
                                    details=details)

    # calculate smoothed length transform
    lt = length_transform(bcg, f, window_length=lt_window, center=True)
    lt = smoothing(lt, f, window_length=smoothing_window)

    # find local maxima
    indices = sgnl.find_peaks(lt)[0]

    indices = first_elimination(lt, f, indices, elimination_window)
    j_indices = relocate_indices(bcg, f, indices, search_window)
    j_indices = second_elimination(bcg, f, j_indices, dist=min_dist)

    return j_indices
