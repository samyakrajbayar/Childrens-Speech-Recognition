"""
Children's ASR Competition - Main Entry Point
DrivenData: On Top of Pasketti - Word Track

This solution uses OpenAI Whisper for automatic speech recognition.
The model weights should be bundled in the submission ZIP.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    """Main inference function for competition runtime"""
    import torch
    import torchaudio
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    start_time = time.time()
    
    # Competition paths
    base_path = Path('/code_execution')
    data_path = base_path / 'data'
    audio_dir = data_path / 'audio'
    metadata_path = data_path / 'utterance_metadata.jsonl'
    output_path = base_path / 'submission' / 'submission.jsonl'
    src_path = base_path / 'src'
    
    # Try bundled models in order of preference
    model_options = [
        src_path / 'whisper-large-v3',
        src_path / 'whisper-medium',
        src_path / 'whisper-small',
        src_path / 'whisper-base',
        src_path / 'whisper-tiny',
    ]
    
    model_id = None
    for model_path in model_options:
        if model_path.exists():
            model_id = str(model_path)
            logger.info(f"Using bundled model: {model_path.name}")
            break
    
    if model_id is None:
        # Fallback to HuggingFace (won't work without network, but useful for local testing)
        model_id = "openai/whisper-large-v3"
        logger.info(f"Using HuggingFace model: {model_id}")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    logger.info(f"Device: {device}")
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    
    # Load model
    logger.info("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    # Set English transcribe mode
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )
    logger.info("Model loaded")
    
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = [json.loads(line) for line in f]
    
    total = len(metadata)
    logger.info(f"Processing {total} utterances")
    
    # Transcription function
    @torch.no_grad()
    def transcribe(audio_path):
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Resample to 16kHz
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Prepare input
            audio_np = waveform.squeeze().numpy()
            inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device, dtype=dtype)
            
            # Generate
            generated = model.generate(
                input_features,
                max_new_tokens=448,
                num_beams=5,
                temperature=0.0
            )
            
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            return text.strip().lower()
            
        except Exception as e:
            logger.error(f"Error: {audio_path.name}: {str(e)[:50]}")
            return ""
    
    # Process all utterances
    predictions = []
    for i, item in enumerate(metadata):
        utterance_id = item['utterance_id']
        audio_path = audio_dir / item['audio_path']
        
        text = transcribe(audio_path)
        predictions.append({
            'utterance_id': utterance_id,
            'orthographic_text': text
        })
        
        # Progress logging (sparse)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info(f"[{i+1}/{total}] {(i+1)/total*100:.0f}% - ETA: {eta/60:.1f}min")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    elapsed = time.time() - start_time
    logger.info(f"Complete! {total} utterances in {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
