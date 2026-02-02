"""
Children's ASR Competition - Main Entry Point (Lightweight Version)
Uses pre-cached Whisper models from HuggingFace Hub.
For DrivenData: On Top of Pasketti - Word Track
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path
import logging

warnings.filterwarnings('ignore')

# Configure minimal logging (500 line limit)
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
    
    # Check for bundled model first, then fall back to HuggingFace
    src_path = base_path / 'src'
    local_model = src_path / 'whisper-large-v3'
    
    if local_model.exists():
        model_id = str(local_model)
        logger.info(f"Using bundled model: {model_id}")
    else:
        # Use HuggingFace model - runtime may have it cached
        model_id = "openai/whisper-large-v3"
        logger.info(f"Using HuggingFace model: {model_id}")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    logger.info("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    # Set English transcribe mode
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )
    logger.info("Model loaded successfully")
    
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
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            audio_np = waveform.squeeze().numpy()
            inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            if device == 'cuda':
                input_features = input_features.half()
            
            generated = model.generate(input_features, max_new_tokens=448)
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            return text.strip().lower()
        except Exception as e:
            logger.error(f"Error: {audio_path}: {e}")
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
        
        # Log progress every 100
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info(f"Progress: {i+1}/{total} - ETA: {eta/60:.1f}min")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    logger.info(f"Done! {total} utterances in {(time.time()-start_time)/60:.1f}min")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
