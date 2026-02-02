"""
Intelligent Model Ensemble for Children's ASR
Combines Whisper, Wav2Vec2, and child-specialized models.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ensemble models"""
    whisper_model_path: str = "openai/whisper-large-v3"
    wav2vec2_model_path: str = "facebook/wav2vec2-xls-r-2b"
    child_specialist_path: str = "facebook/wav2vec2-base"  # Fallback
    temperature: float = 0.7
    beam_size: int = 5
    repetition_penalty: float = 1.2


class ASREnsemble:
    """Intelligent ensemble of multiple ASR models"""
    
    def __init__(self, device: str = 'cuda', config: ModelConfig = None):
        self.device = device
        self.config = config or ModelConfig()
        
        # Load all models
        self._load_models()
        
        # Confidence calibration parameters
        self.confidence_thresholds = {
            'whisper': 0.85,
            'wav2vec2': 0.80,
            'child_specialist': 0.75
        }
    
    def _load_models(self):
        """Load all ensemble models with memory optimization"""
        
        # Whisper model
        logger.info("Loading Whisper model...")
        try:
            self.whisper_processor = WhisperProcessor.from_pretrained(
                self.config.whisper_model_path
            )
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                self.config.whisper_model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.whisper_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load Whisper: {e}")
            self.whisper_model = None
        
        # Wav2Vec2 model
        logger.info("Loading Wav2Vec2 model...")
        try:
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                self.config.wav2vec2_model_path
            )
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
                self.config.wav2vec2_model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            self.wav2vec2_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load Wav2Vec2: {e}")
            self.wav2vec2_model = None
        
        # Child specialist model
        logger.info("Loading child specialist model...")
        try:
            self.child_processor = Wav2Vec2Processor.from_pretrained(
                self.config.child_specialist_path
            )
            self.child_model = Wav2Vec2ForCTC.from_pretrained(
                self.config.child_specialist_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            self.child_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load child specialist: {e}")
            self.child_model = None
    
    @torch.no_grad()
    def transcribe_with_whisper(self, audio: torch.Tensor, utterance_id: str) -> Tuple[str, float]:
        """Transcribe using Whisper with confidence scoring"""
        if self.whisper_model is None:
            return "", 0.0
            
        try:
            # Prepare input
            audio_np = audio.squeeze().cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np[0]
            
            input_features = self.whisper_processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            if self.device == 'cuda':
                input_features = input_features.half()
            
            # Generate with beam search
            predicted_ids = self.whisper_model.generate(
                input_features,
                temperature=self.config.temperature,
                num_beams=self.config.beam_size,
                repetition_penalty=self.config.repetition_penalty,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Decode
            transcription = self.whisper_processor.batch_decode(
                predicted_ids.sequences,
                skip_special_tokens=True
            )[0]
            
            # Calculate confidence
            confidence = self._calculate_confidence(predicted_ids.scores)
            
            return transcription.strip(), confidence
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return "", 0.0
    
    @torch.no_grad()
    def transcribe_with_wav2vec2(self, audio: torch.Tensor, utterance_id: str) -> Tuple[str, float]:
        """Transcribe using Wav2Vec2 with confidence scoring"""
        if self.wav2vec2_model is None:
            return "", 0.0
            
        try:
            # Process audio
            audio_np = audio.squeeze().cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np[0]
            
            inputs = self.wav2vec2_processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Forward pass
            logits = self.wav2vec2_model(inputs.input_values).logits
            
            # Get predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.wav2vec2_processor.batch_decode(predicted_ids)[0]
            
            # Calculate confidence from logits
            probs = F.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            confidence = torch.mean(max_probs).item()
            
            return transcription.strip(), confidence
            
        except Exception as e:
            logger.error(f"Wav2Vec2 transcription error: {e}")
            return "", 0.0
    
    @torch.no_grad()
    def transcribe_with_child_specialist(self, audio: torch.Tensor, utterance_id: str) -> Tuple[str, float]:
        """Transcribe using child-specialized model"""
        if self.child_model is None:
            return "", 0.0
            
        try:
            # Apply child-specific preprocessing
            processed_audio = self._adapt_for_child_voice(audio)
            
            audio_np = processed_audio.squeeze().cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np[0]
            
            inputs = self.child_processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            logits = self.child_model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.child_processor.batch_decode(predicted_ids)[0]
            
            # Calculate confidence
            probs = F.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            confidence = torch.mean(max_probs).item()
            
            return transcription.strip(), confidence
            
        except Exception as e:
            logger.error(f"Child specialist transcription error: {e}")
            return "", 0.0
    
    def _calculate_confidence(self, scores) -> float:
        """Calculate confidence from model scores"""
        if not scores:
            return 0.5
        
        # Average probability of top token at each step
        total_confidence = 0
        for step_scores in scores:
            step_probs = F.softmax(step_scores[0], dim=-1)
            top_prob = torch.max(step_probs).item()
            total_confidence += top_prob
        
        return total_confidence / len(scores)
    
    def _adapt_for_child_voice(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply child-specific audio adaptation"""
        # Placeholder for pitch normalization, formant adjustment, etc.
        return audio
    
    def predict_batch(self, batch_audio: torch.Tensor, utterance_ids: List[str], 
                     weights: Dict[str, float]) -> List[Dict]:
        """Generate ensemble predictions for a batch"""
        predictions = []
        
        for i, (audio, uid) in enumerate(zip(batch_audio, utterance_ids)):
            # Get predictions from all models
            whisper_text, whisper_conf = self.transcribe_with_whisper(audio.unsqueeze(0), uid)
            wav2vec2_text, wav2vec2_conf = self.transcribe_with_wav2vec2(audio.unsqueeze(0), uid)
            child_text, child_conf = self.transcribe_with_child_specialist(audio.unsqueeze(0), uid)
            
            # Collect valid predictions
            candidates = []
            if whisper_text:
                candidates.append((whisper_text, whisper_conf * weights.get('whisper', 0.45)))
            if wav2vec2_text:
                candidates.append((wav2vec2_text, wav2vec2_conf * weights.get('wav2vec2', 0.35)))
            if child_text:
                candidates.append((child_text, child_conf * weights.get('child_specialist', 0.20)))
            
            # Select best prediction
            if candidates:
                # Sort by weighted confidence
                candidates.sort(key=lambda x: x[1], reverse=True)
                final_text = candidates[0][0]
            else:
                final_text = ""
            
            predictions.append({
                'utterance_id': uid,
                'orthographic_text': final_text
            })
        
        return predictions
    
    def _adjust_weights_by_confidence(self, base_weights: Dict[str, float], 
                                      confidences: List[float]) -> List[float]:
        """Adjust ensemble weights based on per-sample confidence"""
        adjusted = []
        for weight, conf in zip(base_weights.values(), confidences):
            adjusted.append(weight * conf)
        
        # Normalize
        total = sum(adjusted)
        if total > 0:
            return [w/total for w in adjusted]
        return list(base_weights.values())
    
    def _majority_vote(self, transcriptions: List[str]) -> str:
        """Majority voting ensemble"""
        from collections import Counter
        
        # Filter empty transcriptions
        valid = [t for t in transcriptions if t]
        if not valid:
            return ""
        
        # Simple: return the first one if all different
        word_counts = Counter(valid)
        return word_counts.most_common(1)[0][0]
    
    def _weighted_ensemble(self, transcriptions: List[str], weights: List[float]) -> str:
        """Weighted ensemble - select transcription with highest weight"""
        if not transcriptions:
            return ""
        
        valid = [(t, w) for t, w in zip(transcriptions, weights) if t]
        if not valid:
            return ""
        
        valid.sort(key=lambda x: x[1], reverse=True)
        return valid[0][0]
