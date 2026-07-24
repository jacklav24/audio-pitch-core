# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================

# PitchSummary:
#   start_time: float
#   end_time: float

#   f0_estimate: float | None
#   uncertainty_hz: float | None

#   confidence: float
#   support: int              # number of contributing frames


from typing import Optional


class PitchSummary:
    # start_frame
    # end_frame
    # f0_estimate
    # uncertainty_hz
    # confidence
    # support
    def __init__(self, start_frame: int, end_frame: int, f0_estimate: Optional[float], uncertainty_hz: Optional[float], confidence: float, support: int):
        assert 0 <= start_frame < end_frame, "Invalid frame interval"
        assert support >= 0, "Support must be non-negative"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in [0, 1]"
        
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.f0_estimate = f0_estimate
        self.uncertainty_hz = uncertainty_hz
        self.confidence = confidence
        self.support = support
