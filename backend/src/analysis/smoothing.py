# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================



from typing import List
from analysis.pitch_frame import PitchFrame
import numpy as np

def smooth_pitch_frames(frames: List[PitchFrame], confidence_min: float = 0.6, window_size: int = 5) -> List[PitchFrame]:
    """
    Apply confidence-gated, segment-local temporal smoothing to a PitchFrame sequence.

    This function performs deterministic post-processing on a sequence of PitchFrame
    objects to improve pitch stability while preserving estimator failure semantics.
    Smoothing operates strictly on contiguous segments of confident frames and never
    infers pitch where the estimator abstained.

    Conceptual model:
    - Each PitchFrame represents an independent, frame-local pitch hypothesis.
    - Confidence is used as a hard gate to determine eligibility for smoothing.
    - Only contiguous sequences of eligible frames are smoothed.
    - Frames outside eligible segments are passed through unchanged.

    Eligibility rule:
    A frame is eligible for smoothing if and only if:
        - frame.f0_hz is not None
        - frame.confidence >= confidence_min

    Segment behavior:
    - Eligible frames are grouped into maximal contiguous segments.
    - Smoothing is applied independently within each segment.
    - No smoothing window ever crosses a segment boundary.
    - Segments of any length (including length 1) are permitted.

    Smoothing behavior:
    - Only f0_hz values are smoothed.
    - Confidence, method, raw_score, frame_index, and timing semantics are preserved.
    - Median filtering is applied using a fixed odd window size.
    - At segment boundaries, the window is shrunk rather than padded or extrapolated.

    Output guarantees:
    - The output list has the same length as the input list.
    - Output PitchFrames appear in the same order as input PitchFrames.
    - frame_index values are preserved exactly.
    - Frames deemed ineligible are unchanged in value.
    - New PitchFrame objects are returned for all positions (immutability preserved).

    Non-goals:
    - No pitch interpolation across gaps or silence
    - No octave correction
    - No modification of confidence values
    - No musical interpretation or note segmentation
    - No temporal prediction or stateful filtering

    Parameters:
    - frames: Ordered list of PitchFrame objects
    - confidence_min: Minimum confidence required for a frame to participate in smoothing
    - window_size: Odd integer defining the median filter window size

    Returns:
    - A new list of PitchFrame objects representing the smoothed sequence

    This function defines the boundary between frame-local pitch estimation and
    sequence-level refinement. It preserves honest failure semantics while allowing
    local temporal consolidation where sufficient evidence exists.
    """
    
    # ensure window size is odd
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")


    segments : List[List[int]] = split_into_confident_segments(frames, confidence_min)
    
    # get f0_hz values from all frames
    smoothed_f0_by_index = [frame.f0_hz for frame in frames]
    # iterate through segmeents
    for segment in segments:
        segment_f0s = [frames[i].f0_hz for i in segment]
        # smooth only frames from the valid segments.
        smoothed_segment_f0s = median_smooth_f0(segment_f0s, window_size)

        # assert we haven't messed up indexes
        assert len(segment_f0s) == len(smoothed_segment_f0s), (
            "Median smoothing must return one value per input f0"
        )
        
        # write the new smoothed values to the original f0 values
        for index, smoothed_f0 in zip(segment, smoothed_segment_f0s):
            smoothed_f0_by_index[index] = smoothed_f0
    
    # write new pitchframes with smoothed f0 values.       
    new_frames = [
        PitchFrame(
            frame_index=frame.frame_index,
            f0_hz = smoothed_f0_by_index[i],
            confidence= frame.confidence,
            method=frame.method,
            raw_score = frame.raw_score,
        )
        for i, frame in enumerate(frames)
    ]
    
    return new_frames


def split_into_confident_segments(frames: List[PitchFrame], confidence_min: float) -> List[List[int]]:
    """
    Identify maximal contiguous segments of PitchFrames eligible for smoothing.

    This function scans an ordered sequence of PitchFrame objects and groups
    frame indices into contiguous segments where each frame satisfies the
    smoothing eligibility criteria.

    Eligibility rule:
    A frame at index i is eligible if and only if:
        - frames[i].f0_hz is not None
        - frames[i].confidence >= confidence_min

    Segment semantics:
    - Each segment is a list of frame indices in strictly increasing order.
    - Segments are maximal: they are terminated by ineligible frames or sequence boundaries.
    - Ineligible frames act as hard barriers and do not appear in any segment.
    - Segments of length 1 are permitted.

    Parameters:
    - frames: Ordered list of PitchFrame objects
    - confidence_min: Minimum confidence required for eligibility

    Returns:
    - A list of segments, where each segment is a list of integer indices into `frames`

    Invariants:
    - Every eligible frame index appears in exactly one segment
    - No ineligible frame index appears in any segment
    - Segment ordering matches input ordering

    This function defines the structural boundaries within which temporal
    smoothing is permitted. It performs no smoothing itself.
    """
    
    segments = []
    current_segment = []
    for i, frame in enumerate(frames):
        # if this frame is valid, append its index to the segment.
        if frame.f0_hz is not None and frame.confidence >= confidence_min:
            current_segment.append(i)
        # if it isn't valid, do other stuff    
        else:
            # if we have a segment currently, add the whole segment to the list of segments.
            if current_segment:
                segments.append(current_segment)
                # clear the current segment
                current_segment.clear()
    # if we have remnants of a segment when we reach the end, we make sure to append it to the list of segments.
    if current_segment: 
        segments.append(current_segment)
        
    return segments


def median_smooth_f0(f0_values: List[float], window_size: int = 5) -> List[float]:
    """
    Apply median smoothing to a sequence of f0 values using a fixed odd window.

    This function performs local, deterministic median filtering on a sequence
    of numeric f0 values. It assumes the input represents a single confidence-
    gated segment and contains no missing or invalid values.

    Window behavior:
    - The window is symmetric with radius r = window_size // 2.
    - At sequence boundaries, the window is shrunk rather than padded.
    - No values outside the input sequence are ever referenced.

    Parameters:
    - f0_values: Ordered list of f0 values (floats), length >= 1
    - window_size: Odd integer defining the median filter window size

    Returns:
    - A list of smoothed f0 values with the same length and ordering as the input

    Invariants:
    - len(output) == len(f0_values)
    - Each output value depends only on local neighbors within the window
    - No extrapolation or interpolation is performed

    Non-goals:
    - No confidence handling
    - No temporal prediction
    - No cross-segment smoothing

    This function is a pure numeric operation and is agnostic to pitch semantics.
    """

    smoothed: List[float] = [] # chatGPT generated
    
    # setting values for the windows
    r = window_size // 2
    N = len(f0_values)
    
    for i in range(N):
        if i < r:
            # left boundary, shrink window forward
            window = f0_values[0:i+r+1]
        elif i >=(N-r):
            # right boundary, shrink window back
            window = f0_values[i-r:N]
        else:
            # full boundary
            window = f0_values[i-r:i+r+1]        
        
        # add smoothed value to the thing. No matter what, a value is added to preserve length.
        smoothed.append(float(np.median(window)))
        
    return smoothed
