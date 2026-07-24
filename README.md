


# Python Time

> **New here?** See [`SUMMARY.md`](SUMMARY.md) for a concise overview of the project’s goals, architecture, and design philosophy.


**My main goal of this project is to first build some infrastructure, for music/audio analysis, which will later support machine learning and higher-level analysis layers. To start, we are building the framing and pitch identification.**

**Currently, main.py runs some tests/checks of the stuff I've built so far. Once I get a little more fleshed out, I'll write a .md file that explains how to run everything.**

#### I've had chatGPT write most of the docstrings for functions and classes after the fact. None of the other code, unless explicitly referenced, is AI generated.

## This is Stage 0 Of the Project

This project emphasizes deterministic signal processing, explicit uncertainty, and empirical characterization before introducing machine learning.

### Needed info:

In terminal type:
`/opt/homebrew/bin/python3 -m venv .venv`

#### You can replace OPT/HOMEBREW/BIN/PYTHON3 with your preferred/installed python.

Then:
`source .venv/bin/activate`

#### TO close....
`deactivate`


We'll load the audio from the disk, convert to a standardized internal representation.

I've settled on **soundfile** as my audio loader. It's a wrapper around **libsndfile**.  
This is best, since it has a good api and is wisely used. It'll require MP3 fallbacks, potentially.

`pip install soundfile` (if you didn't run pip install soundfile)

The __read__ function has a couple of good parameters to use. We'll use `file, dtype, always_2d` to ensure that the file path is specified, datatype is specified, and that the audio isn't forced to stereo at all times.

##### After we input the file, we get a few variables:
`data` 	--->		 			 the actual audio, represented by a numpy array  
`sample_rate `	--->		 how often during the audio is a sample taken
`num_channels` 	---> 		 how many channels the audio have? (mono/stereo/other)

### Building A Long-Term Package
This is for later reference, but I want to follow this setup:
```
audio_analyzer/
│
├── core/
│   ├── audio_buffer.py
│   ├── framing.py
│   ├── pitch.py
│   └── __init__.py
│
├── io/
│   ├── loaders.py
│   └── __init__.py
│
├── main.py
```

## Various Decisions Made:
### Framing size/hop size
For now, I'm taking a bass-guitar optimized approach to framing.

Hop size is commonly **75% overlap**. Bass frequencies are lower, and thus take more time to cycle. We want at least **2-3 cycles**. 

So, our best solution for this, for bass, is larger frames **_(80-120ms)_**, with overlaps of _75%_. Therefore, hop size will be _(`frame_size`/4)_, or **_25ms_**.

**Important to note- frame size and hop size are expressed in _samples_, not milliseconds, inside the system.**




# ELEMENTS




# AudioBuffer (`AudioBuffer`)

An `AudioBuffer` is a validated, mono, float32 audio signal with exact sample-to-time semantics, serving as the immutable source of truth for all downstream analysis.

#### I want to ensure that audio files don't have to be questioned when they're inputted. So, we make some rules.

#### Overall...
#### "An `AudioBuffer` is a mono, float32, linear PCM signal sampled at a fixed rate (`sample_rate`).  
#### Its shape is `(num_samples, )`.
#### `num_samples / sample_rate` **always** defines time exactly.
#### No resampling, trimming, or normalization is a part of the AudioBuffer"

### The logic that ensures we NEVER break this contract is in `audio_buffer.py`.

We use the `dtype=float32` parameter to ensure our datatype is always the same.

Files can come with more than one channel/dimension, so I want to lock it in to mono, for now. More than just mono, I need the shape locked in. I want it represented by a single 1 dimensional array.

That shape is (num_samples,).

The length of time of an AudioBuffer is `num_samples / sample_rate`, where `num_samples = len(data)`. We use this rather than `frames` (a variable defined by soundfile) simply for consistency's sake.

The stored samples and sample rate are kept private inside the object and accessed through getters, so the immutability contract is explicit in the code rather than just a convention.

### AudioBuffer Object:
```
AudioBuffer {  
_data: float[],  
_sample_rate: int,  
}
```

<TO DO> How should we handle silence? How should we signal errors.


# Framing

### What is a Frame?
A frame is a contiguous slice of the AudioBuffer, defined by:
* a fixed number of samples
* a deterministic start index
* a deterministic time reference
* Frames **may overlap**

Each frame is uniquely identified by an integer index and corresponds to a specific time in the original signal.

### Purpose:
- Framing converts a continuous `AudioBuffer` into a sequence of overlapping, fixed-length time slices ("frames") for local signal analysis.

Audio is continuous, computation is discrete. Framing bridges the gap.

Framing allows us to assume local stationarity, perform time-local analysis, and trade time resolution vs freq resolution explicitly. Without it, pitch estimation is tricky and "time" is a vague concept.

###  Things to think about
- Frame size: window length. Larger frame, better freq resolution, worse time resolution
- Hop size: how granular we get. 
- Overlap is mandatory, so hop size < frame size.
- Time anchoring (is a frame's time its start, center, end, other?
- Edge case: End frames, etc.
- Determinism - this must produce the same frames for the same audiobuffer EVERY time.




------

## Frame 

A `Frame` is a deterministic, fixed-length, time-anchored slice of an AudioBuffer that provides localized, stationary views of the signal for per-frame analysis.

#### I want to ensure there are standardized overlapping frames for digesting. So, we make some rules.
### The logic that ensures we NEVER break this contract is in `framing.py`.

### Input: 
A valid `AudioBuffer`, with all of the guarantees that object comes with. We will **not** re-validate.
```
From AudioBuffer, we can assume:
* data is a 1D array of float32
* sample_rate is fixed and correct
* time is linear and exact
```

### Parameters
Framing determined fully by:
	* `frame_size_samples`, a positive int specifying the number of samples per frame.
	* `hop_size_samples`, a positive int specifying the number of samples between consecutive frame starts.

This must satisfy `hop_size_samples ≤ frame_size_samples`.
The parameters stay constant for the entire framing operation.

### Frame Indexing

Frames are indexed starting from `0`.

For a given frame index `i`:
-   `start_sample = i * hop_size_samples`
-   `end_sample = start_sample + frame_size_samples`

Frames are generated in strictly increasing index order.

### Boundary Policy
Only **fully contained frames** are emitted.

A frame is included **iff:**
`end_sample ≤ num_samples`

No padding, truncation, or signal extension is performed. This guarantees deterministic frame counts and exact duration semantics.

### Time Reference

Each frame has a single canonical time value defined as:

> The time (in seconds) corresponding to the **center of the frame**.

Formally:

```
time_seconds =
  (start_sample + frame_size_samples / 2) / sample_rate
  ``` 

This convention is fixed and must be used consistently throughout the system.

### Frame Contents

Each frame contains:

-   `frame_index`  
    Integer index of the frame.
    
-   `start_sample`  
    Integer index of the first sample in the frame.
    
-   `samples`  
    A 1-D NumPy array of shape `(frame_size_samples,)` containing raw audio samples.
    
-   `time_seconds`  
    Float representing the canonical time of the frame.
    

Frames contain **raw samples only**.

No windowing, normalization, or analysis is applied at this stage

#### Output

Framing produces an ordered sequence of frames such that:

-   Frame indices are contiguous (`0` to `N-1`)
    
-   Frame times are strictly increasing
    
-   Each frame represents a local, stationary approximation of the signal
    

The framing contract does not prescribe a specific container type (e.g., list, array, generator).


#### Guarantees

If framing succeeds:

-   The output sequence is deterministic
    
-   Frame boundaries align exactly with sample indices
    
-   Time mapping from samples to seconds is exact
    
-   No artificial signal content is introduced
    


#### Non-Responsibilities

Framing explicitly does **not**:

-   Apply window functions
    
-   Detect silence or onsets
    
-   Estimate pitch or frequency content
    
-   Normalize amplitude
    
-   Perform any musical interpretation
    

These concerns belong to downstream stages.
### Frame Object
```
Frame {  
frame_index : int
start_sample : int
samples : np.ndarray
time_seconds : float
sample_rate : optional  
}
```


------

# PitchFrame (`PitchFrame`)

A `PitchFrame` is a per-frame, time-aligned hypothesis of fundamental frequency with explicit confidence, designed to be honest, estimator-agnostic, and suitable for downstream temporal reasoning.


#### Purpose
Per-frame pitch estimation answers a specific question. It represents the **best instantaneous pitch hypothesis** for a single `Frame`. It is a local observation.

To start, we will use **autocorrelation-based periodicity detection**

#### Input
* Exactly one `Frame`
* Fixed frame size and time anchoring
* Raw, unwindowed samples.

Pitch estimation **doesn't modify the frame** 

### Rules

 `frame_index`  
    Integer index of the frame.
    
-   `f0_hz`  
    Float, estimated fundamental frequency in Hz. Can be `None`
    
-   `confidence`  
    monotonic reliability score. estimators belief in the validity of `f0_hz`
    
-   `method`  
    String identifier of the estimator used
    
-   `raw_score`  
Algorithm-specific signal strength. Float or may be `None`
    
#### Behavioral Guarantees

-   Exactly one `PitchFrame` is produced per `Frame`
    
-   Deterministic for fixed inputs
    
-   No temporal smoothing
    
-   No octave correction
    
-   No musical interpretation
    

Uncertainty is expressed via:

-   `f0_hz = None`
    
-   low `confidence`

### Object
```
PitchFrame {
frame_index : int
  f0_hz       : float | None
  confidence  : float
  method      : str
  raw_score   : float | None
}
```


## The AutoCorrelation Estimator

The autocorrelation estimator measures periodic similarity within a frame, searches for strong physically plausible delays, infers a fundamental frequency when evidence is sufficient, and explicitly reports uncertainty otherwise.


### Inputs
- `frame.samples`
- `sample_rate`
- `f_min`, `f_max`
### Assumed Invariants
- Samples are finite
- Frame length is fixed
- Time anchoring already handled.

This function **NEVER** looks at neighboring frames.

### Steps:
1. Preprocess
	- Remove DC offset
2. Energy Gate (Optional)
	- If energy is really low, return None with 0.0 confidence.
	- This is silence handling.
3. Autocorrelation Computation
	- Measure similarity of signal with delayed copies of itself
		Where:
			- `max_lag = floor(sample_rate / f_min)`
	- Normalize autocorrelation
4. Lag search window
	- Convert frequency bounds to lag bounds
5. Peak detection and selection
	- Id local min/max within `[lag_min, lag_max]`
	- Select the "best" peak, if below a threshold treat as "no pitch"
6. Fundamental frequency inference
	- If a valid peak at `lag = L` is selected:
		- `f0_hz = sample_rate / L`
7. Confidence Computation
	- Confidence, normalized peak height.
	- Optionally scaled or clipped to `[0, 1]`
8. Failure Cases
	- `f0_hz = None` when:
		* No peaks in lag window
		* Peak strength below threshold
		* Frame energy too low
		* Periodicity ambiguous
	- In all failures, return a valid `PitchFrame` with low or zero confidence.
9. Output Mapping
	- Exactly 1 PitchFrame object per input frame. 



# Temporal Pitch Refinement (`smoothing.py`)

Pitch estimation in this project is intentionally **frame-local**.  
No temporal assumptions are made at the estimation stage.

An optional refinement layer exists that operates **strictly downstream of `PitchFrame` sequences**. Its purpose is to explore conservative, confidence-aware temporal reasoning *without modifying the estimator itself*.

## Design Principles

This layer:

- Consumes an ordered list of `PitchFrame` objects
- Produces a **new list of `PitchFrame` objects**
- Never mutates existing frames
- Never invents pitch where none was estimated
- Preserves abstention (`f0_hz = None`) as a first-class outcome
- Is explicitly allowed to make **no changes**

Temporal refinement is treated as a *pure transformation*, not a correction.

## Scope

Current refinement logic may:
- Suppress isolated low-confidence flicker
- Reinforce locally consistent pitch hypotheses
- Respect estimator confidence thresholds

It does **not**:
- Enforce continuity
- Apply musical priors
- Correct octave errors
- Optimize pitch accuracy

The goal is to explore *what temporal context can reveal*, not to hide estimator failures.

# Diagnostic & Characterization Metrics

In addition to pitch estimation, the project includes a **pure diagnostic layer** designed to characterize estimator behavior over time.

This layer operates strictly on sequences of `PitchFrame` objects (raw or smoothed) and is intentionally **non-corrective**.

## Philosophy

The purpose of diagnostics is **not** to improve pitch estimates.

Instead, these metrics aim to answer questions such as:
- When does the estimator speak vs abstain?
- Are failures isolated or regime-based?
- How stable is pitch when voiced?
- When does estimator confidence align with stability — and when does it not?

No ground truth is assumed.  
No musical structure is imposed.

## Scope & Constraints

This layer:

- Does not modify core abstractions
- Does not smooth or reinterpret pitch
- Does not introduce machine learning
- Treats `f0_hz = None` as signal, not noise
- Is deterministic and side-effect free

All analysis is based solely on:
- frame indices
- presence or absence of pitch
- frame-to-frame pitch deltas (when voiced)
- estimator-reported confidence

## Implemented Diagnostic Axes

**Voicing Diagnostics**
- Voiced ratio
- Failure clustering (distinguishing flicker vs contiguous failure regimes)

**Pitch Stability Diagnostics**
- Frame-to-frame pitch volatility
- Continuity of voiced segments

**Confidence Diagnostics**
- Confidence behavior across voiced and unvoiced frames
- Relationship between confidence and short-term pitch stability
- Identifying cases where confidence no longer reflects stability

## Interpretation

These metrics are descriptive, not evaluative.

They are designed to surface:
- non-obvious estimator behavior
- regime-dependent reliability
- limits of confidence as a stability signal
- cases where high confidence does not imply stable pitch

This layer exists to make estimator behavior **inspectable**, not to make it look better.


# Spectrogram (`Spectrogram`)

A `Spectrogram` is a deterministic, STFT-based time–frequency representation of an `AudioBuffer`, built strictly on top of the existing frame lattice and designed as a reversible analytical substrate.

#### Purpose

Framing gives us time-local slices.

The spectrogram gives us **frequency-domain structure** within those slices.

This layer exists to:

-   Introduce a deterministic time–frequency representation
    
-   Remain fully invertible (within defined coverage)
    
-   Preserve the no-padding policy
    
-   Serve as infrastructure for future frequency-domain analysis
    

It is not an interpretation layer.

----------

### Input

-   A valid `AudioBuffer`
    
-   A fixed `window_size`
    
-   A fixed `hop_size`
    
-   Optional `fft_size` (must satisfy `fft_size ≥ window_size`)
    
-   A specified window type (`"hann"` or `"rectangular"`)
    

We reuse `build_frames(...)`.

No alternate framing logic is introduced.

----------

### Boundary Policy

The spectrogram inherits the framing contract:

Only **fully contained frames** are transformed.

No padding, centering, or signal extension is performed.

If the signal length does not allow a full final frame, the tail is excluded.

The spectrogram spans:

covered_length =  
 hop_size * (n_frames - 1) + window_size

This may be less than the original signal length.

This is intentional.

----------

### Representation

Shape convention:

complex_spectrum.shape = (n_frames, n_bins)  
n_bins = fft_size // 2 + 1   (rFFT convention)

The object exposes:

-   `complex_spectrum`
    
-   `magnitude`
    
-   `phase`
    
-   `time_axis`
    
-   `frequency_axis`
    
-   `covered_length`
    
-   `original_length`
    

All stored arrays are immutable.

No scaling beyond NumPy FFT conventions is applied.

No log compression.

No perceptual warping.

No normalization beyond overlap-add compensation in the inverse.

----------

### Inverse (Overlap-Add Reconstruction)

The inverse transform:

-   Uses `irfft`
    
-   Reapplies the analysis window
    
-   Performs overlap-add
    
-   Normalizes by accumulated window energy
    
-   Returns a new `AudioBuffer`
    
-   Reconstructs exactly the covered region
    

Reconstruction error is on the order of floating-point precision (~1e-7 for float32 input).

The transform is therefore:

AudioBuffer  
 ↓  
Spectrogram  
 ↓  
AudioBuffer (covered region only)

Invertible over its defined domain.

----------

### Window Behavior

For a Hann window with 50% overlap:

-   Interior samples satisfy constant overlap-add behavior.
    
-   Window energy is constant across the interior.
    
-   Boundary samples taper to zero (no padding policy).
    

Window normalization is computed explicitly as:

output /= sum(window^2 overlaps)

No hidden normalization is applied.

----------

### Non-Responsibilities

The spectrogram layer does **not**:

-   Estimate pitch
    
-   Perform masking
    
-   Compute chroma
    
-   Compute MFCC
    
-   Apply mel scaling
    
-   Perform source separation
    
-   Smooth
    
-   Threshold
    
-   Interpret musical structure
    
-   Use machine learning
    

It is infrastructure only.

Future frequency-domain layers will build on top of this abstraction.
