"""
Training Data Augmentation for Children's Speech
Specialized augmentation strategies for robust children's ASR training.
"""
import torch
import numpy as np
import random
from typing import Tuple, List, Optional
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


class ChildSpeechAugmentation:
    """Augmentation strategies specifically for children's speech"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.noise_files: List[np.ndarray] = []
        self.room_impulses: List[np.ndarray] = []
        
    def apply_augmentation_pipeline(self, waveform: torch.Tensor, 
                                   augmentation_prob: float = 0.5) -> torch.Tensor:
        """Apply random augmentations"""
        augmented = waveform.clone()
        
        # Always add slight room reverb (realistic)
        if HAS_SCIPY:
            augmented = self.add_room_reverb(augmented)
        
        # Random augmentations
        if random.random() < augmentation_prob and HAS_LIBROSA:
            augmented = self.random_pitch_shift(augmented, max_steps=2)
        
        if random.random() < augmentation_prob:
            augmented = self.add_background_noise(augmented, snr_range=(15, 25))
        
        if random.random() < augmentation_prob * 0.5 and HAS_LIBROSA:
            augmented = self.time_stretch(augmented, rate_range=(0.9, 1.1))
        
        if random.random() < augmentation_prob * 0.3 and HAS_SCIPY:
            augmented = self.simulate_distance(augmented)
        
        return augmented
    
    def add_room_reverb(self, waveform: torch.Tensor, 
                       wet_dry_ratio: float = 0.2) -> torch.Tensor:
        """Add realistic classroom reverb"""
        if not HAS_SCIPY:
            return waveform
            
        audio_np = waveform.numpy().squeeze()
        
        # Create simple impulse response
        ir_length = int(0.3 * self.sr)  # 300ms reverb
        impulse = np.zeros(ir_length)
        impulse[0] = 1.0
        
        # Add decay
        decay = np.exp(-np.linspace(0, 5, ir_length))
        impulse *= decay
        
        # Convolve
        reverbed = signal.convolve(audio_np, impulse, mode='same')
        
        # Mix with original
        output = (1 - wet_dry_ratio) * audio_np + wet_dry_ratio * reverbed
        
        return torch.from_numpy(output).unsqueeze(0).float()
    
    def random_pitch_shift(self, waveform: torch.Tensor, 
                          max_steps: int = 2) -> torch.Tensor:
        """Pitch shift within reasonable range for children"""
        if not HAS_LIBROSA:
            return waveform
            
        audio_np = waveform.numpy().squeeze()
        
        # Random shift up or down
        n_steps = random.uniform(-max_steps, max_steps)
        
        shifted = librosa.effects.pitch_shift(
            audio_np, 
            sr=self.sr, 
            n_steps=n_steps
        )
        
        return torch.from_numpy(shifted).unsqueeze(0).float()
    
    def add_background_noise(self, waveform: torch.Tensor, 
                            snr_range: Tuple[float, float] = (10, 20)) -> torch.Tensor:
        """Add classroom background noise"""
        audio_np = waveform.numpy().squeeze()
        
        if len(self.noise_files) > 0:
            noise = random.choice(self.noise_files).copy()
            # Resample noise to match length
            if len(noise) < len(audio_np):
                noise = np.tile(noise, int(np.ceil(len(audio_np) / len(noise))))
            noise = noise[:len(audio_np)]
        else:
            # Generate pink noise if no noise files available
            noise = self.generate_pink_noise(len(audio_np))
        
        # Adjust SNR
        snr_db = random.uniform(*snr_range)
        signal_power = np.mean(audio_np ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            scale = np.sqrt(signal_power / noise_power / (10 ** (snr_db / 10)))
            noise = noise * scale
        
        output = audio_np + noise
        
        return torch.from_numpy(output).unsqueeze(0).float()
    
    def generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f noise)"""
        if not HAS_SCIPY:
            # Fallback to white noise
            return np.random.randn(length) * 0.01
            
        white = np.random.randn(length)
        
        # Pink noise filter coefficients
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        
        pink = signal.lfilter(b, a, white)
        return pink
    
    def time_stretch(self, waveform: torch.Tensor, 
                    rate_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """Time stretching without pitch change"""
        if not HAS_LIBROSA:
            return waveform
            
        audio_np = waveform.numpy().squeeze()
        original_len = len(audio_np)
        rate = random.uniform(*rate_range)
        
        stretched = librosa.effects.time_stretch(audio_np, rate=rate)
        
        # Ensure same length
        if len(stretched) > original_len:
            stretched = stretched[:original_len]
        elif len(stretched) < original_len:
            stretched = np.pad(stretched, (0, original_len - len(stretched)))
        
        return torch.from_numpy(stretched).unsqueeze(0).float()
    
    def simulate_distance(self, waveform: torch.Tensor, 
                         distance_range: Tuple[float, float] = (1.0, 5.0)) -> torch.Tensor:
        """Simulate distance by applying low-pass filter"""
        if not HAS_SCIPY:
            return waveform
            
        audio_np = waveform.numpy().squeeze()
        
        # Distance affects high-frequency attenuation
        distance = random.uniform(*distance_range)
        cutoff_freq = min(8000 / distance, self.sr / 2 - 100)  # Nyquist limit
        
        # Apply low-pass filter
        sos = signal.butter(4, cutoff_freq, 'lowpass', fs=self.sr, output='sos')
        filtered = signal.sosfilt(sos, audio_np)
        
        # Reduce amplitude with distance
        amplitude = 1.0 / distance
        filtered = filtered * amplitude
        
        return torch.from_numpy(filtered).unsqueeze(0).float()
    
    def add_noise_files(self, noise_paths: List[str]):
        """Load noise files for augmentation"""
        for path in noise_paths:
            try:
                if HAS_LIBROSA:
                    audio, _ = librosa.load(path, sr=self.sr)
                    self.noise_files.append(audio)
            except Exception as e:
                logger.warning(f"Failed to load noise file {path}: {e}")
    
    def add_room_impulses(self, ir_paths: List[str]):
        """Load room impulse responses"""
        for path in ir_paths:
            try:
                if HAS_LIBROSA:
                    ir, _ = librosa.load(path, sr=self.sr)
                    self.room_impulses.append(ir)
            except Exception as e:
                logger.warning(f"Failed to load IR {path}: {e}")
