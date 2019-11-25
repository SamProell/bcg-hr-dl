
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length


def heartrate_from_indices(indices, f, max_std_seconds=float("inf"),
                           min_num_peaks=2, use_median=False):
    """Calculate heart rate from given peak indices

    Args:
        indices (`np.ndarray`): indices of detected peaks
        f (float): in Hz; sampling rate of BCG signal
        min_num_peaks (int): minimum number of peaks to consider valid
        max_std_seconds (float): in seconds; maximum standard deviation
            of peak distances
        use_median (bool): calculate heart rate with median instead of
            mean

    Returns:
        float: mean heartrate estimation in beat/min
    """

    if len(indices) < min_num_peaks:
        return -1

    diffs = np.diff(indices).astype(float) / f
    if np.std(diffs) > max_std_seconds:
        return -1

    if use_median:
        return 60. / np.median(diffs)

    return 60. / np.mean(diffs)


def get_heartrate_pipe(segmenter, max_std_seconds=float("inf"), min_num_peaks=2,
                       use_median=False, index=None):
    """build function that estimates heart rate from detected peaks in
    input signal

    If stddev of peak distances exceeds `max_std_seconds` or less than
    `min_.num_peaks` peaks are found, input signal is marked as invalid
    by returning -1.
    If the `segmenter` returns tuples of wave indices (e.g. IJK instead
    of just J) the wave used for calculations has to be specified with
    `index`.

    Args:
        segmenter (function): BCG segmentation algorithm
        max_std_seconds (float): maximum stddev of peak distances
        min_num_peaks (int): minimum number of detected peaks
        use_median (bool): calculate heart rate from median of peak
            distances instead of mean
        index (int): index of wave used for calculations

    Returns:
        `function`: full heart rate estimation algorithm
    """

    def pipe(x, f, **args):
        indices = segmenter(x, f, **args)
        if index is not None:
            indices = indices[:, index]
        hr = heartrate_from_indices(indices, f,
                                    max_std_seconds=max_std_seconds,
                                    min_num_peaks=min_num_peaks,
                                    use_median=use_median)
        return hr

    return pipe


def get_heartrate_score_pipe(segmenter, use_median=False, index=None):
    """build function that estimates heart rate from detected peaks in
    input signal and return both heart rate and stddev of peak distances

    If the `segmenter` returns tuples of wave indices (e.g. IJK instead
    of just J) the wave used for calculations has to be specified with
    `index`.

    Args:
        segmenter (function): BCG segmentation algorithm
        use_median (bool): calculate heart rate from median of peak
            distances instead of mean
        index (int): index of wave used for calculations

    Returns:
        `function`: full heart rate estimation algorithm that returns
        both estimated heart rate and stddev of peak distances for given
        signal
    """

    def pipe(x, f, **args):
        indices = segmenter(x, f, **args)
        if index is not None:
            indices = indices[:, index]
        n = len(indices)
        if n < 2:
            return -1, -1
        diffs_std = np.std(np.diff(indices) / f)
        hr = heartrate_from_indices(indices, f, max_std_seconds=float("inf"),
                                    min_num_peaks=2, use_median=use_median)

        return hr, diffs_std

    return pipe


def get_valid_hrs(y_true, y_pred, squeeze=True):
    """Get all valid heart rate estimations and corresponding true
    values

    Args:
        y_true (`1d array-like`): true heart rates
        y_pred (`1d array-like`): estimated heart rates
        squeeze (bool): squeeze data

    Returns:
        `1d array, 1d array`: true heart rates and corresponding
        predictions
    """

    if squeeze:
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
    check_consistent_length(y_true, y_pred)
    yt = np.asarray(y_true)[np.greater(y_pred, 0)]
    yp = np.asarray(y_pred)[np.greater(y_pred, 0)]

    return yt, yp


def hr_mape(y_true, y_pred, squeeze=True, **kwargs):
    """Calculate mean absolute percentage error for predictions
    """
    yt, yp = get_valid_hrs(y_true, y_pred, squeeze=squeeze)
    if len(yp) > 0:
        return np.mean(np.divide(np.abs(yp - yt), yt)) * 100
    else:
        return -1


def hr_mae(y_true, y_pred, squeeze=True, **kwargs):
    """Calculate mean absolute error for predictions
    """
    yt, yp = get_valid_hrs(y_true, y_pred, squeeze=squeeze)
    if len(yp) > 0:
        return np.mean(np.abs(yp - yt))
    else:
        return -1


def hr_coverage(y_true, y_pred, **kwargs):
    """Get percentage of valid predictions (valid if > 0)
    """
    return np.count_nonzero(np.greater(y_pred, 0))/float(len(y_pred)) * 100.
