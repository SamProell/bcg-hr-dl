"""J peak detection algorithm by Choi et al. (2009)


B. H. Choi, G. S. Chung, J. S. Lee, D. U. Jeong, and K. S. Park, 'Slow-
wave sleep estimation on a load-cell-installed bed: A non-constrained
method,' Physiol. Meas., vol. 30, no. 11, pp. 1163-1170, 2009
"""

import numpy as np
import scipy.signal as sgnl


def preprocessing(x, f, f1=0.2, f2=8., order=2):
    """Apply bandpass filter for preprocessing

    Args:
        x (`1d array-like`): raw BCG signal
        f (float): in Hz; sampling rate of BCG signal
        f1 (float): in Hz; lower cutoff frequency of bandpass filter
        f2 (float): in Hz; higher cutoff frequency of bandpass filter
        order (int): order of Butterworth bandpass

    Returns:
        `1d array`: preprocessed signal
    """

    bandpass = sgnl.butter(N=order, Wn=np.divide([f1, f2], f / 2.),
                           btype="bandpass")

    return sgnl.filtfilt(bandpass[0], bandpass[1], x)


def get_segment_maxima(x, f, hbi=1.):
    """Determine maxima of given signal for each segment of length
    `hbi`/4

    Args:
        x (`1d array-like`): preprocessed BCG signal
        f (float):  in Hz; sampling rate of input signal
        hbi (float): in seconds; heart beat interval length

    Returns:
        `list(int)`: list of maxima locations
    """

    seglen = int(np.ceil(hbi * f / 4.))
    maxs = []
    for i in range(0, len(x), seglen):
        maxs.append(int(i+np.argmax(x[i:min(i+seglen, len(x))])))

    return maxs


def get_local_maxima(x, f, hbi=1.):
    """estimate local maxima by calculating segment maxima and comaring
    3 successive maxima from list

    Args:
        x (`1d array-like`): preprocessed BCG signal
        f (float): in Hz; sampling rate of input signal
        hbi (float): in seconds; heart beat interval length

    Returns:
        `list(int)`: list of local maximum locations
    """

    maxs = get_segment_maxima(x, f, hbi=hbi)
    # maxvals = np.asarray(x)[np.r_[0, maxs, len(x)-1]]
    maxvals = np.asarray(x)[maxs]
    local_maxs = []
    for i in range(1, len(maxs)-1):
        if maxvals[i] > maxvals[i-1] and maxvals[i] > maxvals[i+1]:
            local_maxs.append(maxs[i])  # local_maxs.append(maxs[i-1])
    return local_maxs


def eliminate_false_peaks(x, f, maxima, hbi=1.):
    """Eliminate detected peaks that are to close together

    Args:
        x (`1d array-like`): preprocessed BCG signal
        f (float): in Hz; sampling rate of input signal
        maxima (`1d array-like`): list of detected local maxima
        hbi (float): in seconds; heart beat interval

    Returns:
        `list(int)`: list of filtered heart beat locations
    """

    maxima = maxima[:]
    min_dist = np.ceil(hbi * f / 4.)*2
    i = 0
    while i < len(maxima) - 2:
        e = np.argmin(x[maxima[i:i+3:2]])*2 + i
        if np.abs(maxima[i+1] - maxima[e]) < min_dist:
            del maxima[e]
        else:
            i += 1

    return maxima


def choi(x, f, f1=1., f2=8., hbi=1., order=2):
    """J peak detection by Choi et al. (2009)

    Args:
        x (`1d array-like`): raw BCG signal
        f (float): in Hz; sampling rate of input signal
        f1 (float): in Hz; lower cutoff frequency of bandpass filter
        f2 (float): in Hz; higher cutoff frequency of bandpass filter
        hbi (float): in seconds; initial heart beat interval length
        order (int): order of Butterworth bandpass filter

    Returns:
        `list(int)`: list of J wave locations
    """
    x = preprocessing(x, f, f1=f1, f2=f2, order=order)

    local_maxima = get_local_maxima(x, f, hbi=hbi)
    indices = eliminate_false_peaks(x, f, local_maxima, hbi=hbi)

    return indices
