"""
Advanced Audio Preprocessing for Children's Speech
Includes noise reduction, voice enhancement, and child-specific optimizations.
"""
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Optional imports for advanced features
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa not available, some features disabled")

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available, some features disabled")


class AudioPreprocessor:
    """Advanced audio preprocessing optimized for children's speech"""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.noise_profile = None
        
    def process(self, waveform: torch.Tensor, sample_rate: int, 
                target_length: Optional[float] = None) -> torch.Tensor:
        """Full preprocessing pipeline"""
        # 1. Resample to target rate
        if sample_rate != self.target_sr:
            waveform = self.resample(waveform, sample_rate, self.target_sr)
        
        # 2. Convert to mono if stereo
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure 2D tensor
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # 3. Noise reduction (spectral gating)
        if HAS_LIBROSA and HAS_SCIPY:
            waveform = self.spectral_gating_noise_reduction(waveform)
        
        # 4. Child-specific voice enhancement
        if HAS_LIBROSA:
            waveform = self.enhance_child_voice(waveform)
        
        # 5. Normalize loudness
        waveform = self.loudness_normalization(waveform)
        
        # 6. Trim or pad to target length
        if target_length:
            target_samples = int(target_length * self.target_sr)
            waveform = self.trim_or_pad(waveform, target_samples)
        
        return waveform.squeeze(0)
    
    def resample(self, waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """High-quality resampling"""
        if orig_sr == target_sr:
            return waveform
        
        resampler = T.Resample(orig_sr, target_sr, dtype=waveform.dtype)
        return resampler(waveform)
    
    def spectral_gating_noise_reduction(self, waveform: torch.Tensor, 
                                       noise_threshold: float = 1.5) -> torch.Tensor:
        """Noise reduction using spectral gating"""
        if not HAS_LIBROSA:
            return waveform
            
        audio_np = waveform.numpy().squeeze()
        
        # STFT
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * self.target_sr / hop_length)
        noise_sample = magnitude[:, :max(1, noise_frames)]
        noise_profile = np.mean(noise_sample, axis=1, keepdims=True)
        
        # Spectral gating
        magnitude_clean = np.maximum(magnitude - noise_threshold * noise_profile, 0)
        
        # Keep original phase
        clean_stft = magnitude_clean * np.exp(1j * np.angle(stft))
        
        # Inverse STFT
        clean_audio = librosa.istft(clean_stft, hop_length=hop_length, length=len(audio_np))
        
        return torch.from_numpy(clean_audio).unsqueeze(0).float()
    
    def enhance_child_voice(self, waveform: torch.Tensor) -> torch.Tensor:
        """Enhance child speech characteristics"""
        if not HAS_LIBROSA:
            return waveform
            
        audio_np = waveform.numpy().squeeze()
        
        try:
            # Pitch tracking
            pitches, magnitudes = librosa.piptrack(
                y=audio_np,
                sr=self.target_sr,
                fmin=100,  # Higher min for children
                fmax=400   # Children have higher pitch
            )
            
            # Get dominant pitch
            pitch_indices = magnitudes.argmax(axis=0)
            pitches_at_max = pitches[pitch_indices, np.arange(pitches.shape[1])]
            mean_pitch = np.mean(pitches_at_max[pitches_at_max > 0])
            
            # If pitch is too high, slightly lower it
            if mean_pitch > 350:
                audio_np = librosa.effects.pitch_shift(
                    audio_np, 
                    sr=self.target_sr, 
                    n_steps=-1
                )
            
            # Apply pre-emphasis for clarity
            audio_np = librosa.effects.preemphasis(audio_np)
            
        except Exception as e:
            logger.warning(f"Voice enhancement failed: {e}")
        
        return torch.from_numpy(audio_np).unsqueeze(0).float()
    
    def loudness_normalization(self, waveform: torch.Tensor, 
                              target_peak: float = 0.9) -> torch.Tensor:
        """Normalize loudness to target peak"""
        peak = torch.max(torch.abs(waveform))
        if peak > 0:
            return waveform / peak * target_peak
        return waveform
    
    def trim_or_pad(self, waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
        """Trim or pad audio to target length"""
        current_samples = waveform.shape[-1]
        
        if current_samples > target_samples:
            # Trim from center
            start = (current_samples - target_samples) // 2
            return waveform[..., start:start + target_samples]
        elif current_samples < target_samples:
            # Pad with zeros
            padding = target_samples - current_samples
            return torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform
    
    def detect_child_speech_characteristics(self, waveform: torch.Tensor) -> Dict:
        """Analyze audio for child-specific features"""
        if not HAS_LIBROSA:
            return {}
            
        audio_np = waveform.numpy().squeeze()
        features = {}
        
        try:
            # Spectral centroid
            features['spectral_centroid'] = float(librosa.feature.spectral_centroid(
                y=audio_np, sr=self.target_sr
            ).mean())
            
            # Speech rate estimation
            features['speech_rate'] = self.estimate_speech_rate(audio_np)
            
        except Exception as e:
            logger.warning(f"Feature detection failed: {e}")
        
        return features
    
    def estimate_speech_rate(self, audio: np.ndarray) -> float:
        """Estimate speech rate using energy envelope"""
        if not HAS_LIBROSA or not HAS_SCIPY:
            return 0.0
            
        energy = librosa.feature.rms(y=audio)[0]
        peaks, _ = signal.find_peaks(energy, distance=10)
        return len(peaks) / (len(audio) / self.target_sr)
