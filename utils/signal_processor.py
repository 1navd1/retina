"""Signal processing utilities for PPG-like signals."""

from typing import Iterable

import numpy as np
from scipy.signal import butter, detrend, filtfilt

DEFAULT_FS = 30.0
LOW_CUT_HZ = 0.5
HIGH_CUT_HZ = 4.0
FILTER_ORDER = 3


def _bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    nyquist = 0.5 * fs
    low = LOW_CUT_HZ / nyquist
    high = HIGH_CUT_HZ / nyquist
    if not 0 < low < high < 1:
        raise ValueError("Invalid bandpass frequencies for the given sampling rate.")
    b, a = butter(FILTER_ORDER, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


def _remove_trend(signal: np.ndarray) -> np.ndarray:
    return detrend(signal, type="linear")


def process_signal(raw_signal: Iterable[float], fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Clean a raw green-channel signal using detrending + bandpass filtering.

    Args:
        raw_signal: Iterable of raw samples.
        fs: Sampling rate in Hz.

    Returns:
        Filtered 1D NumPy array.
    """
    if fs <= 0:
        raise ValueError("Sampling rate fs must be positive.")

    signal = np.asarray(list(raw_signal), dtype=float)
    if signal.size < max(15, FILTER_ORDER * 3):
        raise ValueError("Signal is too short for filtering.")

    signal = signal - np.mean(signal)
    signal = _remove_trend(signal)
    return _bandpass_filter(signal, fs)
