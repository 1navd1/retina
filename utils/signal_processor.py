"""Signal processing utilities for PPG-like waveforms."""

from typing import Tuple

import numpy as np
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Butterworth bandpass filter coefficients.

    Args:
        lowcut: Low cutoff frequency (Hz).
        highcut: High cutoff frequency (Hz).
        fs: Sampling rate (Hz).
        order: Filter order.

    Returns:
        (b, a) filter coefficients.
    """
    if fs <= 0:
        raise ValueError("Sampling rate fs must be positive.")
    if not (0 < lowcut < highcut < fs / 2):
        raise ValueError("Cutoffs must satisfy 0 < lowcut < highcut < fs/2.")
    if order <= 0:
        raise ValueError("Filter order must be positive.")

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to the signal.

    Args:
        data: 1D signal array.
        lowcut: Low cutoff frequency (Hz).
        highcut: High cutoff frequency (Hz).
        fs: Sampling rate (Hz).
        order: Filter order.

    Returns:
        Filtered signal array.
    """
    if data is None or len(data) == 0:
        raise ValueError("Data must be a non-empty 1D array.")
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
