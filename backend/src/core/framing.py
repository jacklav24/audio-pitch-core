# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================



from typing import Generator
from core.audio_buffer import AudioBuffer 
import numpy as np


class Frame:
    """
    Immutable view into a fixed-size window of an AudioBuffer.

    A Frame represents a contiguous block of samples extracted from an AudioBuffer,
    with explicit indexing and exact time alignment. Frames are defined entirely
    in sample space and derive their time semantics from the parent AudioBuffer.

    Attributes:
    - frame_index: Sequential index of the frame (starting at 0)
    - start_sample: Index of the first sample in the AudioBuffer
    - samples: 1D numpy.ndarray view or copy of audio samples
    - time_seconds: Center time of the frame in seconds

    Invariants:
    - samples is 1D numpy.ndarray
    - samples corresponds to audio_buffer.get_data()[start_sample : start_sample + frame_size]
    - time_seconds corresponds to the center of the frame
    - frame_index uniquely identifies ordering, not time

    Non-goals:
    - Frames do not own audio data
    - Frames do not perform analysis
    - Frames do not encode hop size or frame size semantics

    Frame is a structural object used to support deterministic,
    inspectable frame-based analysis.
    """
    def __init__(self, frame_index: int, start_sample: int, samples: np.ndarray, time_seconds: float):
        self.frame_index = frame_index
        self.start_sample = start_sample
        self.samples = samples
        self.time_seconds = time_seconds


def build_frames(audio_buffer: AudioBuffer, frame_size_samples: int, hop_size_samples: int) -> Generator[Frame, None, None]:
    """
    Generate fixed-size, overlapping Frames from an AudioBuffer.

    Frames are constructed deterministically in sample space using the provided
    frame size and hop size. Only frames that fit entirely within the AudioBuffer
    are generated; no padding or extrapolation is performed.

    Frame indexing and timing:
    - frame_index starts at 0 and increments by 1
    - start_sample = frame_index * hop_size_samples
    - time_seconds corresponds to the center of the frame:
          time_seconds = start_sample / sample_rate + (frame_size_samples / sample_rate) / 2

    The number of frames is determined by:
        last_frame_index = floor((num_samples - frame_size_samples) / hop_size_samples)

    Parameters:
    - audio_buffer: Source AudioBuffer
    - frame_size_samples: Number of samples per frame (must be positive)
    - hop_size_samples: Number of samples between frame starts (must be positive)

    Yields:
    - Frame objects in increasing frame_index order

    Non-goals:
    - No padding at signal boundaries
    - No windowing or weighting
    - No analysis or interpretation

    This function defines the temporal discretization used by all downstream
    frame-based analysis.
    """
    
    
    audio_data = audio_buffer.get_data()
    sample_rate = audio_buffer.get_sample_rate()

    last_frame_index = (len(audio_data) - frame_size_samples) // hop_size_samples

    
    for frame_index in range(last_frame_index + 1):
        start_sample = frame_index * hop_size_samples
        end_sample = start_sample + frame_size_samples
        samples = audio_data[start_sample:end_sample]
        time_seconds = start_sample / sample_rate + frame_size_samples / sample_rate / 2.0  # center time of the frame
        yield Frame(frame_index, start_sample, samples, time_seconds)
