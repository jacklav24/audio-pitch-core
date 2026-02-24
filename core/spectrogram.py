# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================

# =============================================================================
# Spectrogram
#
# Deterministic STFT-based time-frequency representation.
#
# This class is purely representational. It performs:
#   - No masking
#   - No smoothing
#   - No pitch estimation
#   - No musical interpretation
#   - No psychoacoustic scaling
#   - No normalization beyond NumPy FFT conventions
#
# The object is immutable after construction.
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import Literal

from core.audio_buffer import AudioBuffer
from core.framing import build_frames


class Spectrogram:
    """
    Immutable STFT-based time-frequency representation.

    Shape convention:
        complex_spectrum.shape == (n_frames, n_bins)

    Where:
        n_bins = fft_size // 2 + 1   (rFFT convention)

    No normalization beyond NumPy FFT defaults.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_audio_buffer(
        cls,
        buffer: AudioBuffer,
        window_size: int,
        hop_size: int,
        fft_size: int | None = None,
        window: Literal["hann", "rectangular"] = "hann",
        center: bool = False,
    ) -> "Spectrogram":

        # ----------------------------
        # Parameter Validation
        # ----------------------------

        if center:
            raise NotImplementedError(
                "center=True is not supported in v1 (no implicit padding allowed)."
            )

        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        if not isinstance(hop_size, int) or hop_size <= 0:
            raise ValueError("hop_size must be a positive integer.")

        if fft_size is None:
            fft_size = window_size

        if not isinstance(fft_size, int) or fft_size <= 0:
            raise ValueError("fft_size must be a positive integer.")

        if fft_size < window_size:
            raise ValueError("fft_size must satisfy fft_size >= window_size.")

        if window not in ("hann", "rectangular"):
            raise ValueError("window must be 'hann' or 'rectangular'.")

        # ----------------------------
        # Window Construction
        # ----------------------------

        if window == "hann":
            window_vector = np.hanning(window_size)
        else:  # rectangular
            window_vector = np.ones(window_size)

        assert window_vector.shape == (window_size,)
        window_vector = window_vector.astype(buffer.data.dtype)

        # ----------------------------
        # Frame Extraction
        # ----------------------------

        frames = list(
            build_frames(
                audio_buffer=buffer,
                frame_size_samples=window_size,
                hop_size_samples=hop_size,
            )
        )

        if len(frames) == 0:
            raise ValueError("Signal too short for even one frame.")

        # ----------------------------
        # STFT Computation
        # ----------------------------

        spectra = []

        for frame in frames:
            samples = frame.samples

            if samples.shape[0] != window_size:
                raise ValueError("Frame size mismatch during STFT computation.")

            # Apply window
            windowed = samples * window_vector

            # rFFT
            spectrum = np.fft.rfft(windowed, n=fft_size)

            spectra.append(spectrum)

        complex_spectrum = np.vstack(spectra)

        # ----------------------------
        # Axis Construction
        # ----------------------------

        n_frames = complex_spectrum.shape[0]
        n_bins = complex_spectrum.shape[1]

        expected_bins = fft_size // 2 + 1
        if n_bins != expected_bins:
            raise ValueError("Unexpected rFFT bin count.")

        sample_rate = buffer.sample_rate

        # Time axis: use frame center times for consistency
        time_axis = np.array([f.time_seconds for f in frames])

        # Frequency axis (Hz)
        frequency_axis = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        
        
        assert frequency_axis.shape[0] == n_bins
        assert time_axis.shape[0] == n_frames
        
        covered_length = hop_size*(n_frames-1) + window_size
        
        if covered_length > len(buffer.data):
            raise ValueError("Covered length exceeds original signal length.")
        # ----------------------------
        # Construct Instance
        # ----------------------------

        obj = cls.__new__(cls)

        obj._complex_spectrum = complex_spectrum
        obj._window_size = window_size
        obj._hop_size = hop_size
        obj._fft_size = fft_size
        obj._window_vector = window_vector
        obj._sample_rate = sample_rate
        obj._original_length = len(buffer.data)
        obj._time_axis = time_axis
        obj._frequency_axis = frequency_axis
        obj._covered_length = covered_length
        # Enforce immutability of stored arrays
        obj._complex_spectrum.setflags(write=False)
        obj._window_vector.setflags(write=False)
        obj._time_axis.setflags(write=False)
        obj._frequency_axis.setflags(write=False)

        return obj

    @property
    def complex_spectrum(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Complex-valued array of shape (n_frames, n_bins).

        Notes
        -----
        - rFFT representation (positive frequencies only).
        - No normalization applied.
        - Returned array must not be mutated.
        """
        return self._complex_spectrum


    @property
    def magnitude(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Magnitude spectrum |X| of shape (n_frames, n_bins).

        Notes
        -----
        - Computed deterministically from complex_spectrum.
        - No log compression.
        - No scaling.
        """
        return np.abs(self._complex_spectrum)


    @property
    def phase(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Phase spectrum (angle in radians) of shape (n_frames, n_bins).

        Notes
        -----
        - Computed via np.angle.
        - Range: (-π, π].
        """
        return np.angle(self._complex_spectrum)

    @property
    def original_length(self) -> int:
        """_summary_

        Return the original AudioBuffer length in samples.

        """
        return self._original_length
    
    
    
    @property
    def time_axis(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            1D array of frame center times in seconds.
            Shape: (n_frames,)

        Notes
        -----
        - Deterministically derived from hop_size and sample_rate.
        - No centering offset unless center=True.
        """
        return self._time_axis


    @property
    def frequency_axis(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            1D array of frequency bin centers in Hz.
            Shape: (n_bins,)

        Notes
        -----
        - Derived from fft_size and sample_rate.
        - Positive frequencies only (rFFT convention).
        """
        return self._frequency_axis

    @property
    def covered_length(self) -> int:
        """
        Returns the total length in samples covered by the spectrogram frames.

        This is given by:
            covered_length = hop_size * (n_frames - 1) + window_size

        This represents the span of the original signal that contributes to the
        spectrogram, accounting for frame overlap. It may be less than or equal
        to the original signal length, depending on framing parameters.
        """
        return self._covered_length
    # -------------------------------------------------------------------------
    # Inverse Transformation
    # -------------------------------------------------------------------------

    def inverse(self) -> AudioBuffer:
        """
        Deterministic inverse STFT reconstruction using overlap-add.
        """

        # ----------------------------
        # Retrieve Stored State
        # ----------------------------

        complex_spectrum = self._complex_spectrum
        window_vector = self._window_vector
        hop_size = self._hop_size
        fft_size = self._fft_size
        sample_rate = self._sample_rate
        window_size = self._window_size
        covered_length = self._covered_length

        if complex_spectrum.ndim != 2:
            raise ValueError("Invalid complex_spectrum shape.")

        n_frames, n_bins = complex_spectrum.shape

        expected_bins = fft_size // 2 + 1
        if n_bins != expected_bins:
            raise ValueError("Spectrum bin count inconsistent with fft_size.")

        # ----------------------------
        # Prepare Output Buffer
        # ----------------------------

        # Total reconstructed length before trimming
        total_length = covered_length

        output = np.zeros(total_length, dtype=np.float64)
        window_energy = np.zeros(total_length, dtype=np.float64)

        # ----------------------------
        # Overlap-Add Reconstruction
        # ----------------------------

        for frame_index in range(n_frames):

            start = frame_index * hop_size
            end = start + window_size

            # Inverse FFT
            frame_time = np.fft.irfft(
                complex_spectrum[frame_index],
                n=fft_size
            )

            if frame_time.shape[0] != fft_size:
                raise ValueError("irfft output length mismatch.")

            # Remove zero padding if fft_size > window_size
            frame_time = frame_time[:window_size]

            # Reapply window
            windowed = frame_time * window_vector

            # Overlap-add
            output[start:end] += windowed

            # Accumulate window energy for normalization
            window_energy[start:end] += window_vector ** 2

        # ----------------------------
        # Normalize Overlap
        # ----------------------------

        nonzero = window_energy > 1e-12
        output[nonzero] /= window_energy[nonzero]

        output = output.astype(np.float32)


        output = output[:covered_length]


        # ----------------------------
        # Return AudioBuffer
        # ----------------------------

        return AudioBuffer(output, sample_rate)