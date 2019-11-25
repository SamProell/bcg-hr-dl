"""Our BCG segmetation algorithm introduced in

Pröll, Samuel M., Stefan Hofbauer, Christian Kolbitsch, Rainer Schubert,
and Karl D. Fritscher. 2019. 'Ejection Wave Segmentation for
Contact-Free Heart Rate Estimation from Ballistocardiographic Signals.'
In 2019 41st Annual International Conference of the IEEE Engineering in
Medicine and Biology Society (EMBC). IEEE.
https://doi.org/10.1109/embc.2019.8857731.
"""

import numpy as np
import pandas as pd
import scipy.signal as sgnl

from .common import get_padded_window


def enhance_signal(x, f, f1=2., f2=10., f3=5., order=2):
    """Preprocess BCG and enhance ejection wave amplitudes by cubing the
    signal and computing filtered second derivative

    Args:
        x (`1d array-like`): raw BCG signal
        f (float): in Hz; sampling rate of input signal
        f1 (float): in Hz; lower cutoff frequency of bandpass filter
        f2 (float): in Hz; higher cutoff frequency of bandpass filter
        f3 (float): in Hz; cutoff frequency of lowpass filter for cubed
            signal
        order (int): order of Butterworth filters

    Returns:
        `(1d array, 1d array)`: bandpass-filtered and enhanced signals
    """

    coeffs1 = sgnl.butter(N=order, Wn=np.divide([f1, f2], f / 2.),
                          btype="bandpass")
    coeffs2 = sgnl.butter(N=order, Wn=np.divide(f3, f / 2.), btype="lowpass")

    # basic preprocessing
    x_filt = sgnl.filtfilt(coeffs1[0], coeffs1[1], x)

    # IJK enhancement
    x_enhanced = sgnl.filtfilt(coeffs2[0], coeffs2[1], x_filt ** 3)
    x_enhanced = -np.gradient(np.gradient(x_enhanced))

    return x_filt, x_enhanced


def renormalize_signal(x, f, window_length=1.):
    """Re-normalize signal by division with a moving stddev signal

    Args:
        x (`1d array-like`): input signal
        f (float): in Hz; sampling rate of input signal
        window_length (float): in seconds; window length for moving
            stddev calculation

    Returns:
        `1d-array`: re-normalized signal
    """

    rolling_std = pd.Series(x).rolling(int(f * window_length),
                                       center=True,
                                       min_periods=1).std()

    rolling_std[rolling_std == 0] = 1.  # avoid divide by zero
    x = np.divide(x, rolling_std)

    return x


def get_coarse_signal(x, f, cutoff=1.5, order=4):
    """Calculate coarse BCG signal from enhanced BCG

    Args:
        x (`1d array-like`): enhanced BCG signal
        f (float): in Hz; sampling rate of input signal
        cutoff (float): in Hz; cutoff frequency of lowpass filter
        order (int): order of Butterworth filter

    Returns:
        `1d array`: coarse BCG signal
    """

    coeffs = sgnl.butter(N=order, Wn=np.divide(cutoff, f / 2.), btype="lowpass")

    return sgnl.filtfilt(coeffs[0], coeffs[1], np.abs(x))


def find_ijk(x, f, ws, wave_dist=0.045):
    """Find IJK indices in small window around peaks in coarse signal

    Args:
        x (`1d array-like`): BCG signal
        f (float): in Hz; sampling rate of BCG signal
        ws (`tuple(float)`): weights for waves
        wave_dist (float): in seconds; minimum distance between peaks

    Returns:
        `1d array`: locations of one peak tuple
    """

    n = len(ws)
    waves = sgnl.find_peaks(np.abs(x), distance=int(wave_dist*f))[0]
    if len(waves) < n:
        return None
    maxi = np.argmax(np.correlate(x[waves], ws, mode="valid"))
    return np.asarray(waves)[np.arange(n) + maxi]


def refine_ijk(x, f, ijk, ws, window_length=0.045):
    """Refine IJK locations detected in enhanced signal

    Args:
        x (`1d array-like`): preprocessed BCG signal
        f (float): in Hz; sampling rate of input signal
        ijk (`tuple(int)`): I, J and K indices
        ws (`tuple(float)`): weights for each wave
        window_length (float): in seconds; small window length for peak
            refinement

    Returns:
        `tuple(int)`: refined IJK peaks
    """

    winsize = int(window_length * f)
    for i, p in enumerate(ijk):
        win = np.sign(ws[i]) * get_padded_window(x, p, winsize,
                                                 padding_value=np.nan)
        maxi = np.nanargmax(win)
        ijk[i] = p - np.ceil(winsize / 2).astype(int) + maxi

    return ijk


def segmenter(x, f, f1=2., f2=10., f3=5., f4=1.35, order=2, renorm_window=1.,
              coarse_dist=0.3, coarse_window=0.5, ws=(-1, 1, -1),
              refine_window=0.1, coarse_order=3, renorm=False, refine=True):
    """BCG segmentation algorithm

    With (-1, 1, -1) the algorithm searches for a valley-peak-valley
    sequence in the BCG signal around coarse BCG locations.
    The valley-peak-valley sequence likely corresponds to I, J and K
    waves.  By using different weights, one could search for more or
    fewer waves (coarse window should be adjusted accordingly).

    Args:
        x (`1d array-like`): raw BCG signal
        f (float): in Hz; sampling rate of input signal
        f1 (float): in Hz; lower cutoff frequency of bandpass filter
        f2 (float): in Hz; higher cutoff frequency of bandpass filter
        f3 (float): in Hz; cutoff frequency of lowpass filter (enhanced
            BCG calculation)
        f4 (float):in Hz; cutoff frequency for lowpass filter (coarse
            BCG calculation)
        order (int): order of Butterworth filters in BCG enhancement
        renorm_window (float): in seconds; window size for signal
            re-normalization
        coarse_dist (float): in seconds; minimum distance of coarse
            peaks
        coarse_window (float): in seconds; window length for detection
            of I, J and K peaks around coarse locations
        ws (`tuple(float)`): weights for weighted sum calculation
        refine_window (float): in seconds; small window length for peak
            refinement
        coarse_order (int): order of Butterworth lowpass for coarse BCG
            calculation
        renorm (bool): apply re-normalization of enhanced signal
        refine (bool): apply peak refinement

    Returns:
        `array`: n x m array, with n being the number of detected
        peak complexes and m being the number of waves per complex
        (length of `ws`)
    """

    x_filt, x_enhanced = enhance_signal(x, f, f1=f1, f2=f2, f3=f3, order=order)
    if renorm:
        x_enhanced = renormalize_signal(x_enhanced, f,
                                        window_length=renorm_window)
    x_coarse = get_coarse_signal(x_enhanced, f, cutoff=f4, order=coarse_order)

    coarse_indices = sgnl.find_peaks(x_coarse, distance=int(coarse_dist*f))[0]

    window_size = int(coarse_window * f)
    ijk_indices = []
    for ci in coarse_indices:
        win = get_padded_window(x_enhanced, ci, window_size)
        ijk = find_ijk(win, f, ws=ws, wave_dist=1./f)
        if ijk is None:
            win = get_padded_window(x_filt, ci, window_size)
            ijk = find_ijk(win, f, ws=ws, wave_dist=1./f)
        if ijk is None:
            continue

        ijk_indices.append(np.asarray(ijk) + max(0, ci-np.ceil(window_size/2.)))

    if refine:
        for i in range(len(ijk_indices)):
            ijk_indices[i] = refine_ijk(x_filt, f, ijk_indices[i].astype(int),
                                        ws=ws, window_length=refine_window)

    return np.array(ijk_indices).astype(int)
