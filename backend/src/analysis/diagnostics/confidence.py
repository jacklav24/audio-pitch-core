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
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from analysis.pitch_frame import PitchFrame


@dataclass(frozen=True)
class ConfidenceMetrics:
    """
    Diagnostics describing the behavior of the estimator's
    confidence signal.
    """
    mean_conf_voiced: float
    mean_conf_unvoiced: float
    conf_stability_corr: float
    conf_autocorr_lag1: float
    high_conf_median_abs_delta_hz: float
    conf_delta_monotonicity: float


def _confidence_vs_delta_correlation(frames: List[PitchFrame]) -> float:
    confidences = []
    deltas = []

    prev_pf = None

    for pf in frames:
        if (
            prev_pf is not None
            and prev_pf.f0_hz is not None
            and pf.f0_hz is not None
            and pf.frame_index == prev_pf.frame_index + 1
        ):
            confidences.append(pf.confidence)
            deltas.append(abs(pf.f0_hz - prev_pf.f0_hz))

        prev_pf = pf

    if len(confidences) < 2:
        return 0.0

    corr, _ = spearmanr(confidences, deltas)
    return float(corr) if not np.isnan(corr) else 0.0


def _confidence_autocorr_lag1(frames: List[PitchFrame]) -> float:
    if len(frames) < 2:
        return 0.0

    c0 = []
    c1 = []

    for prev_pf, pf in zip(frames[:-1], frames[1:]):
        if pf.frame_index == prev_pf.frame_index + 1:
            c0.append(prev_pf.confidence)
            c1.append(pf.confidence)

    if len(c0) < 2:
        return 0.0

    corr = np.corrcoef(c0, c1)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def _confidence_delta_pairs(
    frames: List[PitchFrame],
) -> List[Tuple[float, float]]:
    """
    Collect (confidence, |Δf0|) pairs from adjacent voiced frames.

    Each pair corresponds to a single frame-to-frame transition
    where both frames are voiced and frame_index increments by 1.

    Confidence is taken from the *current* frame.
    """
    pairs: List[Tuple[float, float]] = []

    prev_pf = None

    for pf in frames:
        if (
            prev_pf is not None
            and prev_pf.f0_hz is not None
            and pf.f0_hz is not None
            and pf.frame_index == prev_pf.frame_index + 1
        ):
            delta = abs(pf.f0_hz - prev_pf.f0_hz)
            pairs.append((pf.confidence, delta))

        prev_pf = pf

    return pairs


def _high_conf_median_delta(
    frames: List[PitchFrame],
) -> float:
    """
    Compute median |Δf0| restricted to high-confidence frames
    (defined as top percentile of confidence values).
    """
    pairs = _confidence_delta_pairs(frames)
    if len(pairs) < 2:
        return 0.0
    
    confidences = np.array([c for c, _ in pairs])
    deltas = np.array([d for _, d in pairs])
    
    threshold = np.percentile(confidences, 90)
    
    mask = confidences >= threshold
    if not np.any(mask):
        return 0.0
    return float(np.median(deltas[mask]))
    
    
def _confidence_delta_monotonicity(
    frames: List[PitchFrame],
) -> float:
    """
    Measure whether pitch volatility decreases monotonically
    as confidence increases.

    Confidence values are partitioned into fixed quantile bins,
    and median |Δf0| is computed per bin. The returned value
    is the Spearman correlation between confidence level
    and volatility.
    """
    pairs = _confidence_delta_pairs(frames)
    if len(pairs) < 4:
        return 0.0

    confidences = np.array([c for c, _ in pairs])
    deltas = np.array([d for _, d in pairs])

    # Fixed binning: quartiles
    quantiles = np.quantile(confidences, [0.0, 0.25, 0.5, 0.75, 1.0])

    bin_centers = []
    bin_medians = []

    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        mask = (confidences >= lo) & (confidences <= hi)
        if np.any(mask):
            bin_centers.append((lo + hi) / 2.0)
            bin_medians.append(np.median(deltas[mask]))

    if len(bin_centers) < 2:
        return 0.0

    corr, _ = spearmanr(bin_centers, bin_medians)
    return float(corr) if not np.isnan(corr) else 0.0



def return_confidence_metrics(
    frames: List[PitchFrame],
) -> ConfidenceMetrics:
    """
    Compute diagnostics describing confidence behavior.
    """
    voiced_conf = [pf.confidence for pf in frames if pf.f0_hz is not None]
    unvoiced_conf = [pf.confidence for pf in frames if pf.f0_hz is None]

    mean_voiced = float(np.mean(voiced_conf)) if voiced_conf else 0.0
    mean_unvoiced = float(np.mean(unvoiced_conf)) if unvoiced_conf else 0.0

    return ConfidenceMetrics(
        mean_conf_voiced=mean_voiced,
        mean_conf_unvoiced=mean_unvoiced,
        conf_stability_corr=_confidence_vs_delta_correlation(frames),
        conf_autocorr_lag1=_confidence_autocorr_lag1(frames),
        high_conf_median_abs_delta_hz=_high_conf_median_delta(frames),
        conf_delta_monotonicity=_confidence_delta_monotonicity(frames),
    )
