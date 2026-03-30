# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================


"""
main.py

This file is an exploratory and diagnostic entry point used to:
- validate core invariants
- exercise pitch estimation and refinement
- run diagnostic summaries
- visualize behavior during development

It is not intended as a production API or CLI.
"""

import core.audio_buffer as ab
import core.framing as fr
import analysis.pitch_frame as pf
import analysis.smoothing as sm
import core.spectrogram as sp
import matplotlib.pyplot as plt
import numpy as np
from analysis.diagnostics import summarize_pitch_track
import analysis.spectral_peaks as spk
import analysis.note_projection as npj

def inspect_frames_around_time(frames, pitch_frames, target_time, window=0.25):
    print(f"\nInspecting frames around {target_time:.2f}s (±{window:.2f}s):\n")

    for f, p in zip(frames, pitch_frames):
        if abs(f.time_seconds - target_time) <= window:
            print(
                f"Frame {f.frame_index:5d} | "
                f"time={f.time_seconds:7.3f}s | "
                f"f0={p.f0_hz if p.f0_hz is not None else 'None':>7} | "
                f"conf={p.confidence:.3f}"
            )

def plot_pitch_window(frames, pitch_frames, t_start, t_end):
    import numpy as np
    import matplotlib.pyplot as plt

    times = []
    f0s = []
    confidences = []

    for f, p in zip(frames, pitch_frames):
        if t_start <= f.time_seconds <= t_end:
            times.append(f.time_seconds)
            f0s.append(p.f0_hz if p.f0_hz is not None else np.nan)
            confidences.append(p.confidence)

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(times, f0s, ".", markersize=4)
    plt.ylabel("f0 (Hz)")
    plt.title(f"Pitch estimates from {t_start:.2f}s to {t_end:.2f}s")
    plt.ylim(0, 200)

    plt.subplot(2, 1, 2)
    plt.plot(times, confidences)
    plt.ylabel("Confidence")
    plt.xlabel("Time (s)")
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


def debug_frame_function(audio, frames, frame_size_samples, duration_seconds):
    
    for f in frames[:5]:
        print(
            f"Frame {f.frame_index}: "
            f"start_sample={f.start_sample}, "
            f"time={f.time_seconds:.4f}s, "
            f"samples_shape={f.samples.shape}"
        )
        
    last = frames[-1]
    print(
        f"Last frame {last.frame_index}: "
        f"start_sample={last.start_sample}, "
        f"time={last.time_seconds:.4f}s"
    )

    end_time_estimate = (
        last.start_sample + frame_size_samples
    ) / audio.sample_rate

    print(f"End of last frame ≈ {end_time_estimate:.4f}s")
    print(f"Audio duration      = {duration_seconds:.4f}s")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 3))
    plt.plot(audio.data, alpha=0.5)
    for f in frames[:10]:
        plt.axvline(f.start_sample, color='r', alpha=0.3)
    plt.title("Audio with first few frame start positions")
    plt.show()
    

def debug_pitch_frame_function(pitch_frames, frames):
    
    # Check 1: alignment
    assert len(pitch_frames) == len(frames), "PitchFrames and Frames length mismatch"

    # Check 2: frame index consistency
    for f, p in zip(frames[:10], pitch_frames[:10]):
        print(
            f"Frame {f.frame_index}: "
            f"time={f.time_seconds:.4f}s, "
            f"f0={p.f0_hz}, "
            f"confidence={p.confidence:.3f}"
        )
    last_frame = frames[-1]
    last_pitch = pitch_frames[-1]

    print(
        f"Last Frame {last_frame.frame_index}: "
        f"time={last_frame.time_seconds:.4f}s"
    )

    print(
        f"Last PitchFrame: "
        f"f0={last_pitch.f0_hz}, "
        f"confidence={last_pitch.confidence:.3f}"
    )
    times = [f.time_seconds for f in frames]
    f0s = [p.f0_hz for p in pitch_frames]
    confidences = [p.confidence for p in pitch_frames]
    import matplotlib.pyplot as plt
    import numpy as np

    f0s_plot = [f if f is not None else np.nan for f in f0s]

    plt.figure(figsize=(12, 4))
    plt.plot(times, f0s_plot, '.', markersize=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Estimated f0 (Hz)")
    plt.title("Autocorrelation Pitch Estimates")
    plt.ylim(0, 600)
    plt.show()

def plot_raw_vs_smoothed_pitch_window(
    frames,
    raw_pitch_frames,
    smoothed_pitch_frames,
    t_start,
    t_end,
):
    import numpy as np
    import matplotlib.pyplot as plt

    times = []
    raw_f0s = []
    smooth_f0s = []
    confidences = []

    for f, p_raw, p_smooth in zip(frames, raw_pitch_frames, smoothed_pitch_frames):
        if t_start <= f.time_seconds <= t_end:
            times.append(f.time_seconds)
            raw_f0s.append(p_raw.f0_hz if p_raw.f0_hz is not None else np.nan)
            smooth_f0s.append(p_smooth.f0_hz if p_smooth.f0_hz is not None else np.nan)
            confidences.append(p_raw.confidence)

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(times, raw_f0s, ".", markersize=3, label="Raw f0", alpha=0.6)
    plt.plot(times, smooth_f0s, "-", linewidth=1.5, label="Smoothed f0")
    plt.ylabel("f0 (Hz)")
    plt.title(f"Raw vs Smoothed Pitch ({t_start:.2f}s–{t_end:.2f}s)")
    plt.legend()
    plt.ylim(0, 300)

    plt.subplot(2, 1, 2)
    plt.plot(times, confidences)
    plt.ylabel("Confidence")
    plt.xlabel("Time (s)")
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.show()

def plot_smoothing_delta_window(
    frames,
    raw_pitch_frames,
    smoothed_pitch_frames,
    t_start,
    t_end,
):
    import numpy as np
    import matplotlib.pyplot as plt

    times = []
    deltas = []

    for f, p_raw, p_smooth in zip(frames, raw_pitch_frames, smoothed_pitch_frames):
        if t_start <= f.time_seconds <= t_end:
            if p_raw.f0_hz is not None and p_smooth.f0_hz is not None:
                times.append(f.time_seconds)
                deltas.append(p_smooth.f0_hz - p_raw.f0_hz)

    plt.figure(figsize=(10, 3))
    plt.plot(times, deltas, ".", markersize=3)
    plt.axhline(0, color="gray", linewidth=1)
    plt.ylabel("Δ f0 (Hz)")
    plt.xlabel("Time (s)")
    plt.title(f"Smoothing Adjustment ({t_start:.2f}s–{t_end:.2f}s)")
    plt.tight_layout()
    plt.show()

def summarize_smoothing_adjustments(raw_pitch_frames, smoothed_pitch_frames):
    """
    Summarize the magnitude and frequency of pitch smoothing adjustments.

    Reports how many frames were modified by smoothing and basic statistics
    of the f0 differences (smoothed - raw), considering only frames where
    both raw and smoothed f0 values are present.
    """
    import numpy as np

    deltas = []

    for raw, smooth in zip(raw_pitch_frames, smoothed_pitch_frames):
        if raw.f0_hz is not None and smooth.f0_hz is not None:
            delta = smooth.f0_hz - raw.f0_hz
            if delta != 0.0:
                deltas.append(delta)

    if not deltas:
        print("Smoothing made no numerical adjustments.")
        return

    deltas = np.array(deltas)

    print("Smoothing adjustment summary:")
    print(f"  Frames adjusted        : {len(deltas)}")
    print(f"  Mean |Δf0| (Hz)         : {np.mean(np.abs(deltas)):.4f}")
    print(f"  Median |Δf0| (Hz)       : {np.median(np.abs(deltas)):.4f}")
    print(f"  Max |Δf0| (Hz)          : {np.max(np.abs(deltas)):.4f}")
    print(f"  Std of Δf0 (Hz)         : {np.std(deltas):.4f}")


def assert_no_pitch_invention(raw_pitch_frames, smoothed_pitch_frames):
    """
    Assert that temporal smoothing does not invent pitch values.

    Raises AssertionError if any frame has f0_hz=None before smoothing
    and a non-None f0_hz after smoothing.
    """
    for i, (raw, smooth) in enumerate(zip(raw_pitch_frames, smoothed_pitch_frames)):
        if raw.f0_hz is None and smooth.f0_hz is not None:
            raise AssertionError(
                f"Smoothing invented pitch at frame {i}: {smooth.f0_hz}"
            )

def run_diagnostics(pitch_frames: list[pf.PitchFrame]):
    diagnostics = summarize_pitch_track(pitch_frames)
    print("Voicing:")
    print(f"  Voiced ratio        : {diagnostics.voicing.voiced_ratio:.3f}")
    print(f"  Failure clustering  : {diagnostics.voicing.failure_clustering:.3f}")

    print("Stability:")
    print(f"  Median |Δf0| (Hz)   : {diagnostics.stability.median_abs_delta_hz:.3f}")
    print(f"  Continuity ratio    : {diagnostics.stability.continuity_ratio:.3f}")

    print("Confidence:")
    print(f"  Mean conf (voiced)        : {diagnostics.confidence.mean_conf_voiced:.3f}")
    print(f"  Mean conf (unvoiced)      : {diagnostics.confidence.mean_conf_unvoiced:.3f}")
    print(f"  Conf–Δf0 corr             : {diagnostics.confidence.conf_stability_corr:.3f}")
    print(f"  Conf autocorr (lag 1)     : {diagnostics.confidence.conf_autocorr_lag1:.3f}")
    print(f"  High-conf median |Δf0| Hz : {diagnostics.confidence.high_conf_median_abs_delta_hz:.3f}")
    print(f"  Conf–Δf0 monotonicity     : {diagnostics.confidence.conf_delta_monotonicity:.3f}")
    
def plot_spectrogram(spec: sp.Spectrogram, db: bool = False):
    """
    Visualize magnitude spectrogram.
    """
    magnitude = spec.magnitude

    if db:
        magnitude = 20 * np.log10(np.maximum(magnitude, 1e-10))

    plt.figure(figsize=(10, 4))
    plt.imshow(
        magnitude.T,
        aspect="auto",
        origin="lower",
        extent=[
            spec.time_axis[0],
            spec.time_axis[-1],
            spec.frequency_axis[0],
            spec.frequency_axis[-1],
        ],
    )
    plt.colorbar(label="Magnitude (dB)" if db else "Magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.show()

def inspect_spectrum_frame(spec: sp.Spectrogram, frame_index: int):
    """
    Plot magnitude and phase of a single STFT frame.
    """
    if frame_index < 0 or frame_index >= spec.complex_spectrum.shape[0]:
        raise ValueError("Invalid frame index.")

    freqs = spec.frequency_axis
    magnitude = spec.magnitude[frame_index]
    phase = spec.phase[frame_index]

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(freqs, magnitude)
    plt.ylabel("Magnitude")
    plt.title(f"Frame {frame_index} Spectrum")

    plt.subplot(2, 1, 2)
    plt.plot(freqs, phase)
    plt.ylabel("Phase (radians)")
    plt.xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()
    
def validate_stft_roundtrip(audio: ab.AudioBuffer, spec: sp.Spectrogram):
    """
    Validate forward → inverse reconstruction error over the covered region.

    The Spectrogram operates only on fully-contained frames and therefore
    represents only the interval [0, covered_length). Samples beyond this
    region are intentionally excluded (no padding policy).
    """

    reconstructed = spec.inverse()

    original = audio.data
    covered_length = spec.covered_length

    if len(reconstructed.data) != covered_length:
        raise ValueError(
            "Reconstructed signal length does not match spectrogram covered length."
        )

    if covered_length > len(original):
        raise ValueError(
            "Covered length exceeds original signal length."
        )

    # Compare only covered region
    error = original[:covered_length] - reconstructed.data

    max_error = np.max(np.abs(error))
    mean_error = np.mean(np.abs(error))

    print("\nSTFT Roundtrip Validation")
    print(f"  Original length      : {len(original)}")
    print(f"  Covered length       : {covered_length}")
    print(f"  Max abs error        : {max_error:.8e}")
    print(f"  Mean abs error       : {mean_error:.8e}")

    # Optional: RMS error
    rms_error = np.sqrt(np.mean(error ** 2))
    print(f"  RMS error            : {rms_error:.8e}")

    # Plot reconstruction error
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 3))
    plt.plot(error, linewidth=0.5)
    plt.title("Reconstruction Error (Covered Region Only)")
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.show()
    
def visualize_window_overlap(spec: sp.Spectrogram):
    """
    Visualize accumulated window overlap energy profile.
    """
    n_frames = spec.complex_spectrum.shape[0]
    hop = spec._hop_size
    window = spec._window_vector
    window_size = spec._window_size

    total_length = hop * (n_frames - 1) + window_size
    energy = np.zeros(total_length)

    for k in range(n_frames):
        start = k * hop
        end = start + window_size
        energy[start:end] += window ** 2

    plt.figure(figsize=(10, 3))
    plt.plot(energy)
    plt.title("Window Overlap Energy Profile")
    plt.tight_layout()
    plt.show()
    
def inspect_note_frames(note_frames, frames, target_time, window=0.05):
    print(f"\nInspecting note hypotheses around {target_time:.2f}s (±{window:.2f}s):\n")

    for f, nf in zip(frames, note_frames):
        if abs(f.time_seconds - target_time) <= window:
            print(f"Frame {f.frame_index:5d} | time={f.time_seconds:7.3f}s")

            if not nf.midi_numbers:
                print("   (no notes)")
                continue

            for midi, freq, mag in zip(
                nf.midi_numbers,
                nf.frequencies_hz,
                nf.magnitudes,
            ):
                print(
                    f"   MIDI {midi:3d} | "
                    f"{freq:8.2f} Hz | "
                    f"mag={mag:.6f}"
                )
                      

def plot_detected_notes(frames, note_frames):
    times = []
    midi_vals = []

    for f, nf in zip(frames, note_frames):
        for midi in nf.midi_numbers:
            times.append(f.time_seconds)
            midi_vals.append(midi)

    if not midi_vals:
        print("No note hypotheses to plot.")
        return

    plt.figure(figsize=(10, 4))
    plt.scatter(times, midi_vals, s=5)
    plt.xlabel("Time (s)")
    plt.ylabel("MIDI Note")
    plt.title("Detected Note Hypotheses")
    plt.tight_layout()
    plt.show()
    
  
def summarize_note_activity(note_frames):
    total_frames = len(note_frames)
    active_frames = sum(1 for nf in note_frames if len(nf.midi_numbers) > 0)

    print("\n--- Note Activity Summary ---")
    print(f"  Total frames     : {total_frames}")
    print(f"  Active frames    : {active_frames}")
    print(f"  Activity ratio   : {active_frames / total_frames:.3f}")
    
    
def main():
    # Example audio inputs used for local diagnostics
    FILE_PATHS = ["bass1.wav", "something.wav", "piano_test.wav"]
    
    audio = ab.load_audio_buffer(f"./bass_files/{FILE_PATHS[2]}")
    num_samples = len(audio.data)
    duration_seconds = num_samples / audio.sample_rate
    print(f"Loaded audio buffer with {len(audio.data)} samples at {audio.sample_rate} Hz for {duration_seconds:.2f} seconds")
    
    frame_size_samples = int(0.1 * audio.sample_rate)  # 100 ms frames
    hop_size_samples = int(0.025 * audio.sample_rate)   # 25 ms hop size
    
    frames = list(fr.build_frames(audio, frame_size_samples, hop_size_samples))
    print(f"Built {len(frames)} frames of size {frame_size_samples} samples with hop size {hop_size_samples}")
    
    pitch_frames = list(
        pf.estimate_pitch_sequence(
            frames,
            audio.sample_rate,
            method="autocorr",
            f_min=30.0,
            f_max=500.0,
        )
    )

    
    print(f"Built {len(pitch_frames)} PitchFrames")
    
    confidence_min = 0.6
    window_size = 5
    
    smoothed_pitch_frames = sm.smooth_pitch_frames(pitch_frames, confidence_min=confidence_min, window_size=window_size)
    print(f"Built {len(smoothed_pitch_frames)} smoothed PitchFrames with confidence_min of {confidence_min} and window size of {window_size}")
    
    debug_frame_function(audio, frames, frame_size_samples, duration_seconds)
    debug_pitch_frame_function(pitch_frames, frames)
    plot_pitch_window(frames, pitch_frames, t_start=35.0, t_end=40.0)
    inspect_frames_around_time(frames, pitch_frames, target_time=2.5, window=0.3)
    assert_no_pitch_invention(pitch_frames, smoothed_pitch_frames)
    summarize_smoothing_adjustments(pitch_frames, smoothed_pitch_frames)
    
    print("Raw (unsmoothed) diagnostics")
    run_diagnostics(pitch_frames)

    print("Post-refinement diagnostics")
    run_diagnostics(smoothed_pitch_frames)
    
    print("\n--- Building Spectrogram ---")

    stft_window = frame_size_samples
    stft_hop = hop_size_samples

    spec = sp.Spectrogram.from_audio_buffer(
        audio,
        window_size=stft_window,
        hop_size=stft_hop,
        fft_size=stft_window,
        window="hann",
    )
    print("Magnitude stats:")
    print("  max:", np.max(spec.magnitude))
    print("  mean:", np.mean(spec.magnitude))
    print("  min:", np.min(spec.magnitude))

    print(f"Spectrogram shape: {spec.complex_spectrum.shape}")

    plot_spectrogram(spec, db=True)
    inspect_spectrum_frame(spec, frame_index=10)
    validate_stft_roundtrip(audio, spec)
    visualize_window_overlap(spec)
    print("\n--- Extracting Spectral Peaks ---")

    peak_frames = list(
        spk.extract_spectral_peaks(
            spectrogram=spec,
            magnitude_threshold=0.02,  # deterministic absolute threshold
            min_bin=1,
        )
    )

    print(f"Built {len(peak_frames)} SpectralPeakFrames")


    print("\n--- Projecting Peaks to Notes ---")

    note_frames = list(
        npj.project_peaks_to_notes(
            peak_frames=peak_frames,
            f_min=30.0,
            f_max=2000.0,
        )
    )

    print(f"Built {len(note_frames)} NoteFrames")

    summarize_note_activity(note_frames)
    # inspect_note_frames(
    #     note_frames=note_frames,
    #     frames=frames,
    #     target_time=2.5,
    #     window=0.3,
    # )

    plot_detected_notes(frames, note_frames)


if __name__ == "__main__":
    main()
