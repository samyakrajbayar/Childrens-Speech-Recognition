"""
Child Voice Adaptation Module
Specialized processing for children's speech characteristics.
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ChildVoiceAdapter:
    """Adapt adult-trained models to work better with children's speech"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
        # Child voice characteristics
        self.child_pitch_range = (200, 500)  # Hz
        self.adult_pitch_range = (85, 255)   # Hz
        
        # Formant scaling factor (children have higher formants)
        self.formant_scale = 1.2
        
    def adapt_audio(self, waveform: torch.Tensor, 
                   age_bucket: Optional[str] = None) -> torch.Tensor:
        """Main adaptation pipeline"""
        # Get adaptation parameters based on age
        params = self._get_age_params(age_bucket)
        
        # Apply adaptations
        adapted = waveform.clone()
        
        if HAS_LIBROSA:
            # Pitch normalization
            adapted = self.normalize_pitch(adapted, params['target_pitch'])
            
            # Formant adjustment
            adapted = self.adjust_formants(adapted, params['formant_factor'])
        
        # Speed normalization
        adapted = self.normalize_speed(adapted, params['speed_factor'])
        
        return adapted
    
    def _get_age_params(self, age_bucket: Optional[str]) -> Dict:
        """Get adaptation parameters based on age bucket"""
        default_params = {
            'target_pitch': 150,  # Target adult-like pitch
            'formant_factor': 0.85,  # Scale down formants
            'speed_factor': 1.0
        }
        
        if age_bucket is None:
            return default_params
        
        # Age-specific adjustments
        age_params = {
            '3-4': {'target_pitch': 140, 'formant_factor': 0.8, 'speed_factor': 1.1},
            '5-6': {'target_pitch': 150, 'formant_factor': 0.82, 'speed_factor': 1.05},
            '7-8': {'target_pitch': 160, 'formant_factor': 0.85, 'speed_factor': 1.0},
            '9-10': {'target_pitch': 170, 'formant_factor': 0.88, 'speed_factor': 1.0},
            '11-12': {'target_pitch': 180, 'formant_factor': 0.92, 'speed_factor': 1.0},
        }
        
        return age_params.get(age_bucket, default_params)
    
    def normalize_pitch(self, waveform: torch.Tensor, 
                       target_pitch: float) -> torch.Tensor:
        """Normalize pitch to target value"""
        if not HAS_LIBROSA:
            return waveform
            
        audio_np = waveform.numpy().squeeze()
        
        try:
            # Estimate current pitch
            pitches, magnitudes = librosa.piptrack(
                y=audio_np,
                sr=self.sr,
                fmin=80,
                fmax=500
            )
            
            # Get mean pitch
            pitch_indices = magnitudes.argmax(axis=0)
            pitches_at_max = pitches[pitch_indices, np.arange(pitches.shape[1])]
            valid_pitches = pitches_at_max[pitches_at_max > 0]
            
            if len(valid_pitches) == 0:
                return waveform
            
            current_pitch = np.median(valid_pitches)
            
            if current_pitch > 0:
                # Calculate semitone shift
                semitone_shift = 12 * np.log2(target_pitch / current_pitch)
                
                # Limit shift to reasonable range
                semitone_shift = np.clip(semitone_shift, -6, 6)
                
                # Apply pitch shift
                shifted = librosa.effects.pitch_shift(
                    audio_np,
                    sr=self.sr,
                    n_steps=semitone_shift
                )
                
                return torch.from_numpy(shifted).unsqueeze(0).float()
                
        except Exception as e:
            logger.warning(f"Pitch normalization failed: {e}")
        
        return waveform
    
    def adjust_formants(self, waveform: torch.Tensor, 
                       formant_factor: float) -> torch.Tensor:
        """Adjust formant frequencies to match adult speech"""
        if not HAS_LIBROSA or not HAS_SCIPY:
            return waveform
            
        audio_np = waveform.numpy().squeeze()
        
        try:
            # Use simple spectral envelope modification
            # STFT
            n_fft = 2048
            hop_length = 512
            stft = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length)
            
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Create frequency axis
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
            
            # Create warped frequency mapping
            warped_freqs = freqs * formant_factor
            warped_freqs = np.clip(warped_freqs, 0, self.sr / 2)
            
            # Interpolate magnitude to warped frequencies
            from scipy.interpolate import interp1d
            
            new_magnitude = np.zeros_like(magnitude)
            for t in range(magnitude.shape[1]):
                interp_func = interp1d(
                    warped_freqs, 
                    magnitude[:, t],
                    bounds_error=False,
                    fill_value=0
                )
                new_magnitude[:, t] = interp_func(freqs)
            
            # Reconstruct with original phase
            new_stft = new_magnitude * np.exp(1j * phase)
            
            # Inverse STFT
            output = librosa.istft(new_stft, hop_length=hop_length, length=len(audio_np))
            
            return torch.from_numpy(output).unsqueeze(0).float()
            
        except Exception as e:
            logger.warning(f"Formant adjustment failed: {e}")
        
        return waveform
    
    def normalize_speed(self, waveform: torch.Tensor, 
                       speed_factor: float) -> torch.Tensor:
        """Normalize speech rate"""
        if not HAS_LIBROSA or abs(speed_factor - 1.0) < 0.01:
            return waveform
            
        audio_np = waveform.numpy().squeeze()
        original_len = len(audio_np)
        
        try:
            # Time stretch
            stretched = librosa.effects.time_stretch(audio_np, rate=speed_factor)
            
            # Ensure same length
            if len(stretched) > original_len:
                stretched = stretched[:original_len]
            elif len(stretched) < original_len:
                stretched = np.pad(stretched, (0, original_len - len(stretched)))
            
            return torch.from_numpy(stretched).unsqueeze(0).float()
            
        except Exception as e:
            logger.warning(f"Speed normalization failed: {e}")
        
        return waveform
    
    def detect_child_characteristics(self, waveform: torch.Tensor) -> Dict:
        """Detect child-specific speech characteristics"""
        if not HAS_LIBROSA:
            return {}
            
        audio_np = waveform.numpy().squeeze()
        characteristics = {}
        
        try:
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(
                y=audio_np,
                sr=self.sr,
                fmin=80,
                fmax=500
            )
            
            pitch_indices = magnitudes.argmax(axis=0)
            pitches_at_max = pitches[pitch_indices, np.arange(pitches.shape[1])]
            valid_pitches = pitches_at_max[pitches_at_max > 0]
            
            if len(valid_pitches) > 0:
                characteristics['mean_pitch'] = float(np.mean(valid_pitches))
                characteristics['pitch_std'] = float(np.std(valid_pitches))
                
                # Estimate age bracket based on pitch
                mean_pitch = characteristics['mean_pitch']
                if mean_pitch > 350:
                    characteristics['estimated_age'] = '3-4'
                elif mean_pitch > 300:
                    characteristics['estimated_age'] = '5-6'
                elif mean_pitch > 260:
                    characteristics['estimated_age'] = '7-8'
                elif mean_pitch > 220:
                    characteristics['estimated_age'] = '9-10'
                else:
                    characteristics['estimated_age'] = '11-12'
            
            # Spectral characteristics
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_np, sr=self.sr
            )[0]
            characteristics['spectral_centroid'] = float(np.mean(spectral_centroid))
            
            # Speech rate (syllables per second estimate)
            energy = librosa.feature.rms(y=audio_np)[0]
            if HAS_SCIPY:
                peaks, _ = signal.find_peaks(energy, distance=10)
                duration = len(audio_np) / self.sr
                characteristics['estimated_speech_rate'] = len(peaks) / duration
            
        except Exception as e:
            logger.warning(f"Characteristic detection failed: {e}")
        
        return characteristics
    
    def is_child_voice(self, waveform: torch.Tensor, threshold: float = 0.6) -> Tuple[bool, float]:
        """Detect if the audio contains child speech"""
        characteristics = self.detect_child_characteristics(waveform)
        
        if not characteristics:
            return False, 0.0
        
        # Score based on multiple factors
        score = 0.0
        factors = 0
        
        if 'mean_pitch' in characteristics:
            pitch = characteristics['mean_pitch']
            if pitch > 250:  # Higher pitch indicates child
                score += min(1.0, (pitch - 250) / 150)
            factors += 1
        
        if 'spectral_centroid' in characteristics:
            centroid = characteristics['spectral_centroid']
            if centroid > 2000:  # Higher centroid indicates child
                score += min(1.0, (centroid - 2000) / 2000)
            factors += 1
        
        if factors > 0:
            confidence = score / factors
            return confidence > threshold, confidence
        
        return False, 0.0
