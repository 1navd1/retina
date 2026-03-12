"""Signal processing utilities for PPG-like signals."""

from typing import Iterable

import numpy as np
from scipy.signal import butter, detrend, filtfilt

DEFAULT_FS = 30.0
LOW_CUT_HZ = 0.5
HIGH_CUT_HZ = 4.0
FILTER_ORDER = 3


def _bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_cut_hz: float,
    high_cut_hz: float,
    order: int,
) -> np.ndarray:
    nyquist = 0.5 * fs
    low = low_cut_hz / nyquist
    high = high_cut_hz / nyquist
    if not 0 < low < high < 1:
        raise ValueError("Invalid bandpass frequencies for the given sampling rate.")
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


def _remove_trend(signal: np.ndarray) -> np.ndarray:
    return detrend(signal, type="linear")

def normalize_ppg(raw_signal: Iterable[float], window_size: int = 30) -> np.ndarray:
    """
    Remove the DC component using moving-average subtraction and optional min-max scaling.
    """
    signal = np.asarray(list(raw_signal), dtype=float)
    if signal.size == 0:
        raise ValueError("Signal must be non-empty.")
    if window_size <= 1:
        raise ValueError("window_size must be > 1.")

    kernel = np.ones(window_size, dtype=float) / float(window_size)
    baseline = np.convolve(signal, kernel, mode="same")
    ac_component = signal - baseline

    span = np.max(ac_component) - np.min(ac_component)
    if span != 0:
        return (ac_component - np.min(ac_component)) / span
    return ac_component


def process_signal(raw_signal: Iterable[float], fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Clean a raw green-channel signal using DC normalization + detrending + bandpass filtering.

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

    signal = normalize_ppg(signal)
    signal = signal - np.mean(signal)
    signal = _remove_trend(signal)
    return _bandpass_filter(signal, fs, LOW_CUT_HZ, HIGH_CUT_HZ, FILTER_ORDER)


def apply_filter(
    raw_signal: Iterable[float],
    low_cut_hz: float,
    high_cut_hz: float,
    fs: float,
    order: int = FILTER_ORDER,
    remove_trend: bool = True,
) -> np.ndarray:
    """
    Backward-compatible filter helper used by the Streamlit app.
    """
    if fs <= 0:
        raise ValueError("Sampling rate fs must be positive.")

    signal = np.asarray(list(raw_signal), dtype=float)
    if signal.size < max(15, order * 3):
        raise ValueError("Signal is too short for filtering.")

    signal = signal - np.mean(signal)
    if remove_trend:
        signal = _remove_trend(signal)
    return _bandpass_filter(signal, fs, low_cut_hz, high_cut_hz, order)
