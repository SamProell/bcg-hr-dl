"""Inter-beat interval estimation by Brueser et al. (2013)

C. Brueser, S. Winter, and S. Leonhardt, 'Robust inter-beat interval
estimation in cardiac vibration signals,' Physiol. Meas., vol. 34, no.
2, pp. 123-138, 2013
"""

import numpy as np
import scipy.signal as sgnl

from .common import get_padded_window

epsilon = 1e-7


def normalize_pdf(pdf):
    """Normalize probability density to be non-negative and sum to 1

    Args:
        pdf (`np.ndarray`): probability density

    Returns:
        `np.ndarray`: normalized probability density
    """

    if np.min(pdf) == np.max(pdf):
        return np.ones_like(pdf) / len(pdf)

    pdf = pdf - np.min(pdf)
    return np.divide(pdf, np.sum(pdf) + epsilon)


def preprocessing(x, f, f1=2., f2=20., order=2, savgol_width=0.3,
                  savgol_order=8):
    """Preprocess BCG signals using Butterworth bandpass and
    Savitzky-Golay filter

    Args:
        x (`1d array-like`): raw BCG signal
        f (float): in Hz; sampling rate of input signal
        f1 (float): in Hz; lower cutoff frequency of Butterworth filter
        f2 (float): in Hz; higher cutoff frequency of Butterworth filter
        order (int): order of Butterworth filter
        savgol_width (float): in s; width parameter for SavGol filter
        savgol_order (int): order parameter for SavGol filter

    Returns:
        `1d array`: preprocessed signal
    """

    bandpass = sgnl.butter(N=order, Wn=np.divide([f1, f2], f / 2.),
                           btype="bandpass")
    xfilt = sgnl.filtfilt(bandpass[0], bandpass[1], x)

    savgol_width = int(savgol_width*f)

    # ensure odd width
    if savgol_width % 2 == 0:
        savgol_width += 1

    xfilt = sgnl.savgol_filter(xfilt, window_length=savgol_width,
                               polyorder=savgol_order, deriv=1)

    return xfilt


def check_amplitude(xwin, rmax=15.):
    """Simple amplitude queck: returns True if no sample greater `rmax`
    """
    return np.max(np.abs(xwin)) <= rmax


def get_index_maps(winsize, ns):
    """Get index maps for optimized estimator calculation

    Due to the modified versions of autocorrelation etc. in the
    estimator definition, different window sizes are used for the
    probability estimation of every inter-beat interval length.
    For more efficient calculations (as opposed to loops), index maps
    are defined beforehand which allows the use of numpy array-indexing.

    For example, S_amfd is 1/N*sum(w[v]*w[v-N] for v in 0,...,N).
    Every row of the left map contains v-N for all v in 0,...,N;
    (centered around the middle of the window - winsize//2)
    every row of the right map contains v for all v in 0,...,N;
    each row corresponds to a N in `Nrange`

    S_amfd can then be computed as
    ``np.divide(np.sum(w[right]*w[left], axis=1), Nrange)``

    Resulting speed-up is around 5-10x

    Args:
        winsize (int): size of the signal snippet
        ns (`1d array-like`): inter-beat interval lengths for which
            estimators will be computed

    Returns:
        `np.ndarray, np.ndarray`: left and right index maps
    """

    left = -1*np.ones((len(ns), winsize // 2 + 1), dtype=int)
    right = -1*np.ones_like(left)
    for i, n in enumerate(ns):
        left[i, np.arange(n)] = winsize//2 + np.arange(n) - n
        right[i, np.arange(n)] = winsize//2 + np.arange(n)

    return left, right


def get_estimators_opt(xwin, ns, left, right):
    """Optimized estimator calculation

    Args:
        xwin (`1d array-like`): signal window
        ns (list of int): in samples; inter-beat interval lengths
        left (`np.ndarray`): left index map
        right (`np.ndarray`): right index map

    Returns:
        `np.ndarray, np.ndarray, np.ndarray, np.ndarray`:
        normalized estimators s_corr, s_amfd, s_map and s_tot
    """

    xwin = np.concatenate((xwin, [0]))

    xleft = xwin[left]
    xright = xwin[right]

    s_corr = np.divide(np.sum(xleft * xright, axis=1), ns)
    s_amfd = np.divide(ns,
                       np.sum(np.abs(xright - xleft), axis=1) + epsilon)
    s_map = np.max(xleft + xright, axis=1)

    s_corr = normalize_pdf(s_corr)
    s_amfd = normalize_pdf(s_amfd)
    s_map = normalize_pdf(s_map)

    s_tot = normalize_pdf(s_corr * s_amfd * s_map)

    return s_corr, s_amfd, s_map, s_tot


def estimate_local_interval_size(xwin, ns, lefti, righti,
                                 rel_score=False):
    """Estimate inter-beat interval length for current window position

    Relative score refers to an alternative quality measure that is
    calculated from PDF peak heights of the two interval lengths with
    highest likelihood.

    Args:
        xwin (`1d array-like`): current signal window (preprocessed)
        ns (`1d array-like`): list of potential interval lengths
        lefti (`np.ndarray`): index map for optimized estimator calc.
        righti (`np.ndarray`): index map for optimized estimator calc.
        rel_score (bool): use relative quality score instead of standard

    Returns:
        `(float, float)`: interval length with highest likelihood and
        corresponding quality measure
    """

    _, _, _, s_tot = get_estimators_opt(xwin, ns, lefti, righti)
    s_peaks = sorted(s_tot[sgnl.find_peaks(s_tot)[0]])
    if len(s_peaks) < 1:
        return -1, -1

    quality = s_peaks[-1]
    if rel_score:
        if len(s_peaks) > 1:
            quality = s_peaks[-1]/s_peaks[-2]
        else:
            quality = 10

    return ns[np.argmax(s_tot)], quality


def get_interval_anchor(xwin, ni):
    """Determine anchor point for current window position

    Args:
        xwin (`1d array-like`): current signal window (preprocessed)
        ni (int): estimated inter-beat interval length

    Returns:
        int: index of interval anchor for current window position
    """

    winsize = len(xwin)

    # find local maxima in second half of window
    mi = winsize//2 + sgnl.find_peaks(xwin[winsize//2:])[0]
    mi = mi[mi-winsize//2 < ni]
    if len(mi) < 1:
        return -1

    return mi[np.argmax(xwin[mi] + xwin[mi-ni])]


def ni_to_hr(ni, f):
    """Calculate heart rate in beat/min from estimated interval length

    Args:
        ni (int): estimated inter-beat interval length
        f (float): in Hz; sampling rate of input signal

    Returns:
        float: heart rate in beat/min
    """
    if ni == -1:
        return -1
    return 60. * f / ni


def brueser_extended(x, f,
                     f1=2.,
                     f2=20.,
                     order=2,
                     savgol_width=0.3,
                     savgol_order=8,
                     tmin=0.4,
                     tmax=1.5,
                     nstep=1,
                     delta_t=0.4,
                     rmax=15.,
                     rel_score=False,
                     ):

    """Full inter-beat interval estimation algorithm introduced by
    Brueser et al. (2013)

    Args:
        x (`1d array-like`): raw BCG signal of arbitrary length
        f (float): in Hz; sampling rate of BCG signal
        f1 (float): in Hz; lower cutoff frequency of bandpass filter
        f2 (float): in Hz; higher cutoff frequency of bandpass filter
        order (int): order of Butterworth bandpass filter
        savgol_width (float): in s; width parameter for SavGol filter
        savgol_order (int): order parameter for SavGol filter
        tmin (float): in s; minimum inter-beat interval length
        tmax (float): in s; maximum inter-beat interval length, sliding
            window will be of length 2*tmax
        nstep (int): step size for tested interval lenghts
        delta_t (float): in s; step size of sliding window
        rmax (float): maximum signal amplitude
        rel_score (bool): use alternative quality measure

    Returns:
        `np.ndarray`: n x 3 array containing unique interval anchor
        locations (column 1) and corresponding interval length (column
        0) and quality measure (column 2)
    """

    xfilt = preprocessing(x, f, f1=f1, f2=f2, order=order,
                          savgol_width=savgol_width, savgol_order=savgol_order)

    winsize = int(2 * tmax * f)
    ns = np.arange(int(tmin*f), int(tmax*f), step=nstep, dtype=int)
    lefti, righti = get_index_maps(winsize, ns)

    # iterate sliding window over input signal
    ni_anchor_quality = []
    for loc in range(winsize//2, len(xfilt)-winsize//2, int(delta_t*f)):
        window = get_padded_window(xfilt, loc, winsize)

        if not check_amplitude(window, rmax=rmax):
            continue

        ni, qi = estimate_local_interval_size(window, ns, lefti, righti,
                                              rel_score=rel_score)
        if ni == -1:
            ni_anchor_quality.append((-1, -1, -1))
            continue
        pi = get_interval_anchor(window, ni)
        if pi == -1:  # no anchor found
            continue
        ni_anchor_quality.append((ni, loc-winsize//2 + pi, qi))

    if len(ni_anchor_quality) == 0:
        return None

    ni_anchor_quality = np.array(ni_anchor_quality)

    # for each unique interval anchor, calculate median heart rate and
    # corresponding quality measure
    ni_anchor_quality_unique = []
    for anchor in np.unique(ni_anchor_quality[:, 1]):
        tmp = ni_anchor_quality[ni_anchor_quality[:, 1] == anchor, :]
        median_index = np.argsort(tmp[:, 0])[len(tmp)//2]
        hr = tmp[median_index, 0]
        q = tmp[median_index, 2]
        ni_anchor_quality_unique.append((hr, anchor, q))

    return np.array(ni_anchor_quality_unique)


def brueser(x, f, f1=2., f2=20., order=2, savgol_width=0.3, savgol_order=8,
            tmin=0.42, tmax=1.5, nstep=1, delta_t=0.21, rmax=15.,
            rel_score=False):
    """Estimate heart rate for given BCG input signal (8s window)

    From all interval anchors found within the signal, the heart rate
    with the highest quality measure is chosen.
    If calculations are invalid (-1, -1) is returned.

    We use a delta_t of 1 second our evaluations. rmax is set to a value
    higher than all possible values (movements are excluded before
    application of this algoirthms). Hyper-parameters for Butterworth
    and SavGol filters were not optimized exhaustively for our BCG data.

    Args:
        x (`1d array-like`): raw BCG signal of arbitrary length
        f (float): in Hz; sampling rate of BCG signal
        f1 (float): in Hz; lower cutoff frequency of bandpass filter
        f2 (float): in Hz; higher cutoff frequency of bandpass filter
        order (int): order of Butterworth bandpass filter
        savgol_width (float): in s; width parameter for SavGol filter
        savgol_order (int): order parameter for SavGol filter
        tmin (float): in s; minimum inter-beat interval length
        tmax (float): in s; maximum inter-beat interval length, sliding
            window will be of length 2*tmax
        nstep (int): step size for tested interval lenghts
        delta_t (float): in s; step size of sliding window
        rmax (float): maximum signal amplitude
        rel_score (bool): use alternative quality measure

    Returns:
        `(float, float)`: estimated heart rate for signal window and
        corresponding quality
    """

    naq = brueser_extended(x, f,
                           f1=f1,
                           f2=f2,
                           order=order,
                           savgol_width=savgol_width,
                           savgol_order=savgol_order,
                           tmin=tmin,
                           tmax=tmax,
                           nstep=nstep,
                           delta_t=delta_t,
                           rmax=rmax,
                           rel_score=rel_score,
                           )
    if naq is None:
        return -1, -1

    return ni_to_hr(naq[np.argmax(naq[:, 2]), 0], f), np.max(naq[:, 2])


def brueser2(x, f, f1=2., f2=20., order=2, savgol_width=0.3, savgol_order=8,
             tmin=0.42, tmax=1.5, nstep=1, delta_t=0.21, rmax=15.,
             rel_score=False):
    """For debugging: return heart rates and quality for all unique
    interval anchors
    """
    naq = brueser_extended(x, f,
                           f1=f1,
                           f2=f2,
                           order=order,
                           savgol_width=savgol_width,
                           savgol_order=savgol_order,
                           tmin=tmin,
                           tmax=tmax,
                           nstep=nstep,
                           delta_t=delta_t,
                           rmax=rmax,
                           rel_score=rel_score,
                           )
    if naq is None:
        return [], []

    return naq[:, 0], naq[:, 2]
