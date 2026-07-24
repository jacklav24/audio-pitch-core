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

"""
    This file is purely an orchestration layer. It exists to accept a List of PitchFrames,
    call metric calculators from the folder, and assemble a single "diagnostics" object.
    NO MATH ALLOWED!! (awww)
"""


from dataclasses import dataclass
from typing import List, Optional

from analysis.pitch_frame import PitchFrame
from analysis.diagnostics.voicing import (
    VoicingMetrics,
    return_voicing_metrics,
)
from analysis.diagnostics.stability import (
    PitchStabilityMetrics,
    return_pitch_stability_metrics,
)
from analysis.diagnostics.confidence import (
    ConfidenceMetrics,
    return_confidence_metrics,
)


@dataclass(frozen=True)
class PitchDiagnostics:
    """
    Aggregate diagnostics for a pitch track.

    This object summarizes structural and statistical
    properties of a PitchFrame sequence without modifying it
    or asserting correctness.
    """
    voicing: VoicingMetrics
    stability: PitchStabilityMetrics
    confidence: ConfidenceMetrics
    
def summarize_pitch_track(
    frames: List[PitchFrame],
) -> PitchDiagnostics:
    """
    Summarize diagnostic properties of a pitch track.

    Accepts raw or smoothed PitchFrames.
    Assumes a single pitch estimation method.
    """
    if not frames:
        raise ValueError("Cannot summarize empty PitchFrame list.")

    # Enforce homogeneous method (confidence semantics depend on this)
    methods = {pf.method for pf in frames}
    if len(methods) != 1:
        raise ValueError(
            f"PitchDiagnostics requires a single method, found: {methods}"
        )

    voicing = return_voicing_metrics(frames)
    stability = return_pitch_stability_metrics(frames)
    confidence = return_confidence_metrics(frames)

    return PitchDiagnostics(
        voicing=voicing,
        stability=stability,
        confidence=confidence,
    )
