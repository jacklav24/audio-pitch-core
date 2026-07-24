#TODO : build
# =============================================================================
# Spectral Peak → Discrete Note Projection
#
# Deterministic frequency-to-MIDI mapping.
#
# This module:
#   - Does NOT perform harmonic grouping
#   - Does NOT collapse octaves
#   - Does NOT introduce temporal smoothing
#   - Does NOT infer chords
#
# Frame-local only.
# =============================================================================

from typing import Iterable, Generator
import numpy as np

from analysis.spectral_peaks import SpectralPeakFrame


class NoteFrame:
    """
    Immutable per-frame discrete note hypotheses.

    Attributes:
    - frame_index
    - midi_numbers: list[int]
    - frequencies_hz: list[float]
    - magnitudes: list[float]

    Semantics:
    - One entry per accepted spectral peak
    - No harmonic collapsing
    - No octave folding
    - No temporal context
    """

    def __init__(
        self,
        frame_index: int,
        midi_numbers: list[int],
        frequencies_hz: list[float],
        magnitudes: list[float],
    ):
        self.frame_index = frame_index
        self.midi_numbers = midi_numbers
        self.frequencies_hz = frequencies_hz
        self.magnitudes = magnitudes


A4_FREQ = 440.0
A4_MIDI = 69


def frequency_to_midi(frequency_hz: float) -> int:
    """
    Deterministic nearest-MIDI mapping.

    Raises:
    - ValueError if frequency <= 0
    """

    if frequency_hz <= 0:
        raise ValueError("Frequency must be positive.")

    return int(np.rint(A4_MIDI + 12.0 * np.log2(frequency_hz / A4_FREQ)))


def project_peaks_to_notes(
    peak_frames: Iterable[SpectralPeakFrame],
    f_min: float,
    f_max: float,
) -> Generator[NoteFrame, None, None]:
    """
    Deterministic per-frame note projection.

    Parameters:
    - peak_frames: Iterable of SpectralPeakFrame
    - f_min: Minimum allowed frequency
    - f_max: Maximum allowed frequency

    Yields:
    - NoteFrame per frame

    Notes:
    - No harmonic grouping
    - No temporal smoothing
    - No chord inference
    """

    if f_min <= 0 or f_max <= 0:
        raise ValueError("f_min and f_max must be positive.")
    if f_min > f_max:
        raise ValueError("f_min must be less than or equal to f_max.")

    midi_from_frequency = frequency_to_midi

    for peak_frame in peak_frames:
        midi_numbers: list[int] = []
        frequencies_hz: list[float] = []
        magnitudes: list[float] = []

        for peak in peak_frame.peaks:
            frequency_hz = peak.frequency_hz
            if f_min <= frequency_hz <= f_max:
                midi_numbers.append(midi_from_frequency(frequency_hz))
                frequencies_hz.append(frequency_hz)
                magnitudes.append(peak.magnitude)

        yield NoteFrame(
            frame_index=peak_frame.frame_index,
            midi_numbers=midi_numbers,
            frequencies_hz=frequencies_hz,
            magnitudes=magnitudes,
        )
