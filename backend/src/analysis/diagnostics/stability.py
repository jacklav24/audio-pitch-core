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

import numpy as np

from analysis.pitch_frame import PitchFrame


@dataclass(frozen=True)
class PitchStabilityMetrics:
    """
    Diagnostics describing short-term pitch variability
    conditional on voicing.

    All metrics operate only on adjacent voiced frames
    and make no claims about pitch correctness.
    """
    median_abs_delta_hz: float
    p90_abs_delta_hz: float
    continuity_ratio: float


def _adjacent_voiced_deltas(frames: List[PitchFrame]) -> List[float]:
    deltas: List[float] = []

    prev_pf = None

    for pf in frames:
        if (
            prev_pf is not None
            and prev_pf.f0_hz is not None
            and pf.f0_hz is not None
            and pf.frame_index == prev_pf.frame_index + 1
        ):
            deltas.append(abs(pf.f0_hz - prev_pf.f0_hz))

        prev_pf = pf

    return deltas

def _continuity_ratio(frames: List[PitchFrame]) -> float:
    if len(frames) < 2:
        return 0.0

    possible_pairs = 0
    voiced_pairs = 0

    for prev_pf, pf in zip(frames[:-1], frames[1:]):
        if pf.frame_index == prev_pf.frame_index + 1:
            possible_pairs += 1
            if prev_pf.f0_hz is not None and pf.f0_hz is not None:
                voiced_pairs += 1

    return voiced_pairs / possible_pairs if possible_pairs > 0 else 0.0


def return_pitch_stability_metrics(
    frames: List[PitchFrame],
) -> PitchStabilityMetrics:
    """
    Compute pitch stability diagnostics for a pitch track.
    """
    deltas = _adjacent_voiced_deltas(frames)

    if not deltas:
        return PitchStabilityMetrics(
            median_abs_delta_hz=0.0,
            p90_abs_delta_hz=0.0,
            continuity_ratio=_continuity_ratio(frames),
        )

    arr = np.asarray(deltas)

    return PitchStabilityMetrics(
        median_abs_delta_hz=float(np.median(arr)),
        p90_abs_delta_hz=float(np.percentile(arr, 90)),
        continuity_ratio=_continuity_ratio(frames),
    )
