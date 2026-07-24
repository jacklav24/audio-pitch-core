# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================

"""
Diagnostics only.
Does not modify PitchFrames.
Does not infer correctness.
"""

from dataclasses import dataclass
from typing import List

from analysis.pitch_frame import PitchFrame



@dataclass(frozen=True)
class VoicingMetrics:
    """
    Diagnostics describing when the pitch estimator
    produces hypotheses vs abstains.

    All metrics are frame-based and make no claims
    about pitch correctness.
    """
    voiced_ratio: float
    failure_clustering: float
    
def is_voiced_frame(pf: PitchFrame) -> bool:
    # A frame is voiced if the estimator gives a pitch hypothesis
    return pf.f0_hz is not None

def compute_voiced_ratio(frames: List[PitchFrame]) -> float:
    if not frames:
        raise ValueError("Cannot compute voiced ratio on empty frame list")
    
    is_voiced = sum(1 for pf in frames if is_voiced_frame(pf))
    
    return is_voiced/len(frames)

def get_voiced_segment_lengths(frames: List[PitchFrame]) -> List[int]:
    segments: List[int] = []
    
    current_length = 0
    prev_frame_index = None
    
    for pf in frames:
        is_voiced = is_voiced_frame(pf)
        if (is_voiced 
            and prev_frame_index is not None 
            and prev_frame_index == pf.frame_index - 1
        ):
            current_length +=1
        elif is_voiced:
            if current_length > 0:
                segments.append(current_length)
            current_length = 1
        else:
            if current_length > 0:
                segments.append(current_length)
                current_length = 0
                
        prev_frame_index = pf.frame_index
    
    if current_length > 0:
        segments.append(current_length)
    
    
    return segments

def get_unvoiced_segment_lengths(frames: List[PitchFrame]) -> List[int]:
    """
    Return the lengths (in frames) of all contiguous unvoiced segments.
    """
    segments: List[int] = []

    current_length = 0
    prev_frame_index = None

    for pf in frames:
        is_voiced = is_voiced_frame(pf)
        if (
            not is_voiced
            and prev_frame_index is not None
            and pf.frame_index == prev_frame_index + 1
        ):
            current_length += 1

        elif not is_voiced:
            if current_length > 0:
                segments.append(current_length)
            current_length = 1

        else:
            if current_length > 0:
                segments.append(current_length)
                current_length = 0

        prev_frame_index = pf.frame_index

    if current_length > 0:
        segments.append(current_length)

    return segments


def compute_failure_clustering(frames: List[PitchFrame]) -> float:
    if not frames:
        raise ValueError("Cannot compute failure clustering on empty frame list.")

    unvoiced_segments = get_unvoiced_segment_lengths(frames)
    num_unvoiced_frames = sum(1 for pf in frames if pf.f0_hz is None)

    if num_unvoiced_frames == 0:
        return 0.0  # No failures → no clustering

    mean_unvoiced_segment_length = (
        sum(unvoiced_segments) / len(unvoiced_segments)
        if unvoiced_segments
        else 0.0
    )

    unvoiced_ratio = num_unvoiced_frames / len(frames)

    return mean_unvoiced_segment_length / unvoiced_ratio


def return_voicing_metrics(frames: List[PitchFrame]) -> VoicingMetrics:
    return VoicingMetrics(
        voiced_ratio=compute_voiced_ratio(frames),
        failure_clustering=compute_failure_clustering(frames)
    )
