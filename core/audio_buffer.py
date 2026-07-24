# =============================================================================
# Copyright (c) 2026 Jack LaVergne
#
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
# =============================================================================



import soundfile as sf
import numpy as np


class AudioBuffer:
    """
    Immutable container representing a mono, float32, linear PCM audio signal
    sampled at a fixed rate.

    Contract:
    - `data` is a 1D numpy.ndarray of dtype float32 with shape (num_samples,)
    - Audio is mono (single channel)
    - Sample values are finite (no NaN or Inf)
    - `sample_rate` is a positive integer (Hz)
    - Time semantics are exact:
        time_seconds = num_samples / sample_rate

    Non-goals:
    - No resampling
    - No trimming or padding
    - No normalization
    - No interpretation of audio content

    An AudioBuffer represents raw audio exactly as loaded, with no modification
    beyond enforcing the above invariants.
    """
    def __init__(self, data: np.ndarray, sample_rate: int):
        self._data = data
        self._sample_rate = sample_rate

    def get_data(self) -> np.ndarray:
        """
        Return the immutable audio sample array.
        """
        return self._data

    def get_sample_rate(self) -> int:
        """
        Return the fixed sample rate in Hz.
        """
        return self._sample_rate
        
        
        

def load_audio_buffer(file_path: str) -> AudioBuffer:
    """
    Load an audio file from disk and return it as a validated AudioBuffer.

    This function enforces the full AudioBuffer contract by:
    - Loading audio data as float32 PCM
    - Converting multi-channel audio to mono via channel averaging
    - Verifying shape, dtype, finiteness, and duration
    - Verifying sample rate validity

    Guarantees on success:
    - Returns an AudioBuffer whose `data` is a 1D float32 numpy array
    - `data` contains at least one sample
    - `data` contains only finite values
    - `sample_rate` is a positive integer
    - Audio duration is strictly positive

    Failure behavior:
    - Raises or propagates an exception if any contract check fails
    - No partial or invalid AudioBuffer is ever returned

    Non-goals:
    - No resampling
    - No normalization
    - No trimming, padding, or silence removal
    - No semantic interpretation of audio content

    Parameters:
        file_path (str):
            Path to an audio file readable by soundfile.

    Returns:
        AudioBuffer:
            A validated, immutable audio buffer satisfying the AudioBuffer contract.
            
    Raises:
    - TypeError or ValueError if any contract check fails
    - Propagates I/O errors from soundfile

    This function defines the sole entry point by which raw audio enters
    the analysis pipeline.
    """
    
    try:
        with sf.SoundFile(file_path) as f:
            data = f.read(dtype='float32', always_2d=False)
            sample_rate = f.samplerate
            
            # -- Check contract -- #
            
            # numpy array
            if not isinstance(data, np.ndarray):
                raise TypeError("Data must be a numpy.ndarray")
            
            # array of float32
            if data.dtype != np.float32:
                raise TypeError(f"Data must be of type float32, got {data.dtype}")
            
            # convert to mono if stereo
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            
            #must be mono
            if data.ndim!= 1:
                raise ValueError(f"Data must be 1-dimensional (mono), got {data.ndim} dimensions")
            
            #must have at least one sample
            if len(data) == 0:
                raise ValueError("Data must contain at least one sample")
            
            # must not contain NaNs or Infs
            if not np.all(np.isfinite(data)):
                raise ValueError("Data contains non-finite values")
            
            # sample rate must be positive
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError(f"Sample rate must be a positive integer, got {sample_rate}")
            
            num_samples = len(data)
            duration_seconds = num_samples / sample_rate
            
            # positive duration 
            if duration_seconds <= 0:
                raise ValueError("Audio duration must be greater than zero")
            
            # -- End check contract -- #
            
            # Close the file
            f.close()
            return AudioBuffer(data, sample_rate)
            
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
