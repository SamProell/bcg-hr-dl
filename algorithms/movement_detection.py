
import numpy as np
import pandas as pd

from .common import filter_bandpass


def detect_movements(x, f, f1=2., f2=10., th0=125., thf1=2.5, order=2,
                     percentile=90, stdwin=2., th2=3., margin=1.):
    """Detect movements in raw BCG signal by simple applying simple
    thresholds

    Every data point in the input signal is checked against different
    thresholds and bits corresponding to different thresholds are set if
    the values exceed ceratin thresholds.
    If the value of the movement signal is greater 0, movements were
    detected at that point.

    Applied thresholds are:

       - raw amplitude: signal should not exceed `th0`
       - deviation: bandpass-filtered signal should not exceed `thf1`
         times the `percentile`th percentile of the bandpass-filtered
         signal
       - moving deviation: moving stddev with window size `stdwin` of
         bandpass-filtered signal should not exceed `th2`

    After thresholding, a moving maximum calculation is applied to
    provide a 'safety margin' for detected movements.

    Args:
        x (`1d array-like`): raw BCG signal
        f (float): in Hz; sampling rate of input signal
        f1 (float): in Hz; lower cutoff frequency of bandpass filter
        f2 (float): in Hz; higher cutoff frequency of bandpass filter
        th0 (float): maximum raw amplitude value
        thf1 (float): thresholding factor for deviation threshold
        order (int): order of Butterworth bandpass filter
        percentile (float): percentile used for deviation thresholding
        stdwin (float): in seconds; window length for moving stddev
            calculation
        th2 (float): moving deviation threshold
        margin (float): in seconds; window size for moving maximum
            calculation of safety margin.  `margin` is applied to both
            sides of detected movements.

    Returns:
        `1d array`: movement signal containing values greater 0 where
        movements were detected
    """

    xfilt = filter_bandpass(x, f, cutoff1=f1, cutoff2=f2, order=order)
    xstd = pd.Series(xfilt).rolling(int(f*stdwin), center=True, min_periods=1
                                    ).std()

    # check all four thresholds
    raw_amplitude_bit = np.greater(np.abs(x), th0) * 1
    deviation_bit = np.greater(np.abs(xfilt),
                               thf1 * np.percentile(np.abs(xfilt), percentile)
                               ) * 2
    moving_deviation_bit = np.greater(xstd.values, th2) * 4

    # combine thresholds to single movement signal
    movement_signal = np.zeros_like(x, dtype=int)
    for bits in [raw_amplitude_bit, deviation_bit, moving_deviation_bit]:
        movement_signal = np.bitwise_or(movement_signal, bits.astype(int))

    # apply 'safety' margin
    return pd.Series(movement_signal).rolling(int(2*f*margin), center=True,
                                              min_periods=1
                                              ).max().values
