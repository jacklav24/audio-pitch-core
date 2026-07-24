# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================
# Segment:
#   start_time: float
#   end_time: float

#   frames: tuple[PitchFrame]   # immutable reference

#   voiced_ratio: float         # fraction with f0_hz != None
#   confidence_stats:
#     mean: float
#     median: float
#     dispersion: float         # IQR or MAD, not variance

#   pitch_stats:
#     median_f0: float | None
#     dispersion_hz: float | None


from typing import List, Optional, Sequence
from analysis.pitch_frame import PitchFrame
import numpy as np

class Segment:
    
    """
    Represents a contiguous interval of PitchFrames over frame indices.

    All temporal coordinates are expressed in frame indices, not seconds.
    A Segment asserts internal coherence sufficient for downstream
    summarization, but makes no musical or perceptual claims.
    """
    
    def __init__(self, start_frame: int, end_frame: int, pitch_frames: list[PitchFrame]):
        assert 0 <= start_frame < end_frame, "Segment indices must satisfy 0 <= start < end"
        
        assert len(pitch_frames) == (end_frame - start_frame), \
            "Frame count must match index span"
        
        assert len(pitch_frames) > 0, "Segment must contain at least one frame"
        
        self.start_index = start_frame
        self.end_index = end_frame
        self.pitch_frames = tuple(pitch_frames)
        
        

    @property
    def voiced_ratio(self) -> float:
        # A Segment may contain frames with f0 is None; such frames contribute to duration and confidence statistics but not pitch statistics.
        
        return sum(f.f0_hz is not None for f in self.pitch_frames) / len(self.pitch_frames)

    @property
    def mean_confidence(self) -> float:
        return sum(f.confidence for f in self.pitch_frames) / len(self.pitch_frames)
    @property
    def median_f0(self) -> float | None:
        f0s = [f.f0_hz for f in self.pitch_frames if f.f0_hz is not None]
        if not f0s:
            return None
        return np.median(f0s)
    @property
    def dispersion_hz(self) -> float | None:
        f0s = [f.f0_hz for f in self.pitch_frames if f.f0_hz is not None]
        if len(f0s) < 2:
            return None
        f0s = np.array(f0s)
        median = np.median(f0s)
        return np.median(np.abs(f0s - median))  # MAD
    


    @property
    def support(self) -> int:
        return len(self.pitch_frames)

from typing import Sequence, List, Optional

def segment_by_failure(
    pitch_frames: Sequence[PitchFrame],
    max_gap: int,
    min_length: Optional[int] = None
) -> List[Segment]:
    segments: List[Segment] = []

    current_frames: List[PitchFrame] = []
    segment_start: Optional[int] = None
    failure_run = 0

    for i, frame in enumerate(pitch_frames):
        is_failure = frame.f0_hz is None

        if is_failure:
            failure_run += 1
        else:
            failure_run = 0

        if segment_start is None:
            segment_start = i
        current_frames.append(frame)
        if failure_run > max_gap:
            # Terminate current segment (if any)
            segment_frames = current_frames[:-failure_run]
            if segment_frames:
                end_index = segment_start + len(segment_frames)
                segments.append(
                    Segment(
                        start_frame=segment_start,
                        end_frame=end_index,
                        pitch_frames=segment_frames
                    )
                )

            # Reset state
            current_frames = []
            segment_start = None
            failure_run = 0

 

    # Final segment
    if current_frames and segment_start is not None:
        segments.append(
            Segment(
                start_frame=segment_start,
                end_frame=segment_start + len(current_frames),
                pitch_frames=current_frames
            )
        )

    # Optional minimum-length filter
    if min_length is not None:
        segments = [
            seg for seg in segments
            if len(seg.pitch_frames) >= min_length
        ]

    return segments
