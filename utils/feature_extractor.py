"""Feature extraction utilities for PPG-like signals."""

from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np
from scipy.signal import find_peaks


def center_roi(frame: np.ndarray, box_size: int = 100) -> Tuple[int, int, int, int]:
    """
    Return center ROI coordinates as (x0, y0, x1, y1).

    Args:
        frame: BGR frame array (H, W, C).
        box_size: Desired square ROI size in pixels.
    """
    if frame is None or frame.ndim < 2:
        raise ValueError("Frame must be a valid image array.")
    if box_size <= 0:
        raise ValueError("box_size must be positive.")

    h, w = frame.shape[:2]
    size = min(box_size, h, w)
    x0 = (w - size) // 2
    y0 = (h - size) // 2
    x1 = x0 + size
    y1 = y0 + size
    return x0, y0, x1, y1


def mean_green_from_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    """
    Compute the mean green-channel intensity within ROI.
    """
    x0, y0, x1, y1 = roi
    region = frame[y0:y1, x0:x1]
    if region.size == 0:
        raise ValueError("ROI is empty for the given frame.")
    return float(np.mean(region[:, :, 1]))


def init_signal_buffer(fs: int = 30, duration_sec: int = 30) -> Deque[float]:
    """
    Create rolling buffer for the latest `duration_sec` of samples.
    """
    if fs <= 0 or duration_sec <= 0:
        raise ValueError("fs and duration_sec must be positive.")
    return deque(maxlen=fs * duration_sec)


def append_signal_sample(buffer: Deque[float], green_value: float) -> None:
    """
    Append one green-channel sample to rolling buffer.
    """
    buffer.append(float(green_value))


def extract_features(clean_sig: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract simple features from a filtered signal.

    Args:
        clean_sig: Filtered 1D signal array.
        fs: Sampling rate (Hz).

    Returns:
        Dict with keys: hr_bpm, std_dev, mean_val.
    """
    if clean_sig is None or len(clean_sig) == 0:
        raise ValueError("Signal must be a non-empty 1D array.")
    if fs <= 0:
        raise ValueError("Sampling rate fs must be positive.")

    peaks, _ = find_peaks(clean_sig, distance=max(1, int(0.5 * fs)))
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        hr_bpm = 60.0 / np.mean(rr_intervals)
    else:
        hr_bpm = 0.0

    std_dev = float(np.std(clean_sig))
    mean_val = float(np.mean(clean_sig))

    return {
        "hr_bpm": float(hr_bpm),
        "std_dev": std_dev,
        "mean_val": mean_val,
    }
