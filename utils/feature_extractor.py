"""Feature extraction utilities for PPG-like signals."""

from typing import Dict

import numpy as np
from scipy.signal import find_peaks


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
