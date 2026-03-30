# TODO: Build this

from typing import Generator
import numpy as np

import core.spectrogram as sp

class SpectralPeak:
    """
    Represents a single spectral peak detected in a spectrogram frame.

    Attributes:
    - frequency_hz: Frequency of the peak in Hz
    - magnitude_db: Magnitude of the peak in decibels
    - bin_index: Index of the corresponding FFT bin
    - frame_index: Index of the source spectrogram frame

    Semantics:
    - frequency_hz is derived from bin_index and sample_rate
    - magnitude_db is a logarithmic measure of peak strength
    - SpectralPeak is a local property of a single spectrogram frame

    Invariants:
    - bin_index corresponds to frequency_hz via sample_rate and FFT size
    - SpectralPeak contains no temporal context beyond frame_index

    Non-goals:
    - No temporal tracking or smoothing of peaks across frames
    - No musical interpretation or pitch inference
    """
    
    def __init__(self, frequency_hz: float, magnitude: float, bin_index: int, frame_index: int):
        self.frequency_hz = frequency_hz
        self.magnitude = magnitude
        self.bin_index = bin_index
        self.frame_index = frame_index
        
        
class SpectralPeakFrame:
    """
    Immutable container for spectral peaks of a single frame.

    Attributes:
    - frame_index
    - peaks: list[SpectralPeak]

    Semantics:
    - One object per spectrogram frame
    - No temporal context
    - No harmonic reasoning
    """

    def __init__(self, frame_index: int, peaks: list[SpectralPeak]):
        self.frame_index = frame_index
        self.peaks = peaks


def extract_spectral_peaks(
    spectrogram: sp.Spectrogram,
    magnitude_threshold: float,
    min_bin: int = 1,
) -> Generator[SpectralPeakFrame, None, None]:
    """
    Deterministic spectral peak extraction.

    Parameters:
    - spectrogram: Spectrogram instance
    - magnitude_threshold: Absolute magnitude threshold
    - min_bin: Ignore bins below this index (avoid DC)

    Yields:
    - SpectralPeakFrame per spectrogram frame

    Notes:
    - Strictly local maxima detection (to be implemented)
    - No temporal smoothing
    - No harmonic grouping
    """
    magnitude = spectrogram.magnitude
    frequency_axis = spectrogram.frequency_axis

    if magnitude.ndim != 2:
        raise ValueError("spectrogram magnitude must be a 2D array.")

    n_frames, n_bins = magnitude.shape

    if n_bins == 0:
        return

    start_bin = max(int(min_bin), 1)
    end_bin = n_bins - 1

    if start_bin >= end_bin:
        for frame_index in range(n_frames):
            yield SpectralPeakFrame(frame_index=frame_index, peaks=[])
        return

    for frame_index in range(n_frames):
        frame_magnitude = magnitude[frame_index]
        center = frame_magnitude[start_bin:end_bin]

        is_peak = (
            (center >= magnitude_threshold)
            & (center > frame_magnitude[start_bin - 1:end_bin - 1])
            & (center >= frame_magnitude[start_bin + 1:end_bin + 1])
        )

        peak_bins = np.flatnonzero(is_peak) + start_bin

        peaks = [
            SpectralPeak(
                frequency_hz=float(frequency_axis[bin_index]),
                magnitude=float(frame_magnitude[bin_index]),
                bin_index=int(bin_index),
                frame_index=frame_index,
            )
            for bin_index in peak_bins
        ]

        yield SpectralPeakFrame(frame_index=frame_index, peaks=peaks)
