# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================



from typing import Generator, Iterable
from core.framing import Frame
import numpy as np

class PitchFrame:
    """
    Immutable per-frame pitch hypothesis produced by a pitch estimator.

    A PitchFrame represents the result of applying a pitch estimation algorithm
    to a single Frame. It encodes a local hypothesis about fundamental frequency
    without any temporal context or smoothing.

    Attributes:
    - frame_index: Index of the source Frame
    - f0_hz: Estimated fundamental frequency in Hz, or None if no pitch is detected
    - confidence: Non-negative scalar indicating estimator self-consistency
    - method: Identifier of the pitch estimation method used
    - raw_score: Optional method-specific diagnostic score

    Semantics:
    - f0_hz is None indicates estimator failure or intentional abstention
    - confidence reflects local structural evidence, not correctness
    - confidence == 0.0 always implies f0_hz is None
    - confidence values are comparable only within the same method

    Invariants:
    - frame_index corresponds exactly to the source Frame
    - PitchFrame contains no temporal context beyond frame_index
    - PitchFrame does not infer or propagate pitch across frames

    Non-goals:
    - No temporal smoothing
    - No musical interpretation
    - No octave correction
    - No continuity assumptions

    PitchFrame is the atomic unit of pitch analysis and serves as the input
    to all downstream sequence-level processing.
    """
    
    def __init__(self, frame_index: int, f0_hz : float | None, confidence : float, method : str, raw_score: float | None):
        self.frame_index = frame_index
        self.f0_hz = f0_hz
        self.confidence = confidence
        self.method = method
        self.raw_score = raw_score
        
        
        
def estimate_pitch_autocorr(frame : Frame, sample_rate: int, f_min : float = 30.0, f_max : float = 500.0) -> PitchFrame:
    """
    Estimate the fundamental frequency of a single Frame using autocorrelation.

    This function applies a deterministic, frame-local autocorrelation-based
    pitch estimation algorithm. Each Frame is processed independently with
    no temporal context or smoothing.

    Algorithm outline:
    - Remove DC offset
    - Reject silent frames via RMS energy threshold
    - Compute normalized autocorrelation
    - Restrict search to lags corresponding to [f_min, f_max]
    - Detect local maxima in the autocorrelation function
    - Select the strongest valid peak
    - Convert lag to frequency

    Confidence semantics:
    - confidence is the normalized height of the selected autocorrelation peak
    - confidence reflects local periodic structure, not pitch correctness
    - confidence == 0.0 indicates estimator failure or abstention

    Failure behavior:
    - Returns f0_hz=None and confidence=0.0 if:
        - RMS energy is below threshold
        - No valid autocorrelation peak is found
        - Peak strength is below the minimum threshold
        - Lag range is invalid

    Parameters:
    - frame: Input Frame to analyze
    - sample_rate: Sample rate of the source AudioBuffer
    - f_min: Minimum allowable fundamental frequency (Hz)
    - f_max: Maximum allowable fundamental frequency (Hz)

    Returns:
    - PitchFrame containing the per-frame pitch hypothesis

    Non-goals:
    - No temporal smoothing
    - No octave correction
    - No continuity assumptions
    - No pitch inference across silence or ambiguity
    """
    
    samples = frame.samples
    
    # Remove DC offset
    samples = samples - np.mean(samples)
    
    # RMS energy check for silence
    rms_energy = np.sqrt(np.mean(samples**2))
    ENERGY_THRESHOLD = 1e-4
    
    if rms_energy < ENERGY_THRESHOLD:
        return PitchFrame(
            frame_index=frame.frame_index,
            f0_hz=None,
            confidence=0.0,
            method="autocorr",
            raw_score=None
        )
    
    # Copute autocorrelation
    autocorr = np.correlate(samples, samples, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Keep only non-negative lags
    
    # Normalize by zero-lag value
    zero_lag = autocorr[0]
    autocorr = autocorr / zero_lag

    # Return no pitch if less than zero, since the signal is silent
    if zero_lag <= 0:
        return PitchFrame(frame.frame_index, None, 0.0, "autocorr", None)
    
    lag_min = int(np.ceil(sample_rate / f_max))
    lag_max = int(np.floor(sample_rate / f_min))
    
    # guard against invalid lag range
    if lag_min >= lag_max or lag_max >= len(autocorr):
        return PitchFrame(frame.frame_index, None, 0.0, "autocorr", None)
    
    search_region = autocorr[lag_min:lag_max]
    
    peaks = []
    for i in range(1, len(search_region) - 1):
        if search_region[i] > search_region[i - 1] and search_region[i] > search_region[i + 1]:
            peaks.append((i + lag_min, search_region[i]))  # (lag, value)
            
    if not peaks:
        return PitchFrame(frame.frame_index, None, 0.0, "autocorr", None)
    
    # select the strongest peak 
    best_lag, best_value = max(peaks, key=lambda x: x[1])
    #ts peak

    # Minimum peak strength threshold
    PEAK_THRESHOLD = 0.3

    if best_value < PEAK_THRESHOLD:
        return PitchFrame(
            frame_index=frame.frame_index,
            f0_hz=None,
            confidence=0.0,
            method="autocorr",
            raw_score=best_value,
        )

    # Convert lag to frequency
    f0_hz = sample_rate / best_lag

    # Confidence is proportional to normalized peak height
    confidence = float(best_value)

    return PitchFrame(
        frame_index=frame.frame_index,
        f0_hz=f0_hz,
        confidence=confidence,
        method="autocorr",
        raw_score=best_value,
    )
    



def estimate_pitch_sequence_autocorr(frames: Iterable[Frame], sample_rate: int, f_min: float = 30.0, f_max: float = 500.0,) -> Generator[PitchFrame, None, None]:
    """
    Apply autocorrelation-based pitch estimation to a sequence of Frames.

    This function processes each Frame independently using
    `estimate_pitch_autocorr` and yields a corresponding PitchFrame.
    Frame ordering and frame_index values are preserved exactly.

    Parameters:
    - frames: Iterable of Frame objects
    - sample_rate: Sample rate of the source AudioBuffer
    - f_min: Minimum allowable fundamental frequency (Hz)
    - f_max: Maximum allowable fundamental frequency (Hz)

    Yields:
    - PitchFrame objects in the same order as the input Frames

    Invariants:
    - One PitchFrame is produced per input Frame
    - No temporal context is introduced
    - No smoothing or post-processing is applied

    This function defines the per-frame pitch estimation stage
    of the analysis pipeline.
    """
    
    for frame in frames:
        yield estimate_pitch_autocorr(frame, sample_rate, f_min, f_max)
        
        
def estimate_pitch_sequence(
    frames: Iterable[Frame],
    sample_rate: int,
    method: str = "autocorr",
    **kwargs,
    ) -> Generator[PitchFrame, None, None]:
    """
    Dispatch a sequence of Frames to a selected pitch estimation method.

    This function provides a unified interface for applying different
    pitch estimation algorithms to a Frame sequence. The dispatcher
    itself performs no analysis and introduces no temporal context.

    Supported methods:
    - "autocorr": Autocorrelation-based pitch estimation

    Parameters:
    - frames: Iterable of Frame objects
    - sample_rate: Sample rate of the source AudioBuffer
    - method: Identifier of the pitch estimation method
    - kwargs: Method-specific parameters forwarded to the estimator

    Returns:
    - Generator yielding PitchFrame objects

    Raises:
    - ValueError if the requested method is not supported

    Invariants:
    - Output ordering matches input ordering
    - Exactly one PitchFrame is produced per input Frame
    - No smoothing or interpretation is performed

    This function defines the estimator selection boundary and isolates
    downstream analysis from estimator-specific details.
    """
    
    if method == "autocorr":
        return estimate_pitch_sequence_autocorr(frames, sample_rate, **kwargs)
    else:
        raise ValueError(f"Unknown pitch estimation method: {method}")
