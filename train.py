"""
Fine-tuning Script for Children's ASR Models
Trains Whisper and Wav2Vec2 models on children's speech data.
"""
import json
import os
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, Audio
import pandas as pd
from sklearn.model_selection import train_test_split
import jiwer
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_type: str = "whisper"  # whisper or wav2vec2
    base_model: str = "openai/whisper-large-v3"
    output_dir: str = "./models/whisper_finetuned"
    
    # Training params
    batch_size: int = 8
    learning_rate: float = 1e-5
    max_steps: int = 10000
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 2
    
    # Evaluation
    eval_steps: int = 1000
    save_steps: int = 1000
    
    # Hardware
    fp16: bool = True
    gradient_checkpointing: bool = True


class WhisperDataCollator:
    """Data collator for Whisper training"""
    
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
    
    def __call__(self, features: List[Dict]) -> Dict:
        # Extract audio and transcripts
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        # Pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features, 
            return_tensors="pt"
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features, 
            return_tensors="pt"
        )
        
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].eq(0), -100
        )
        
        batch["labels"] = labels
        
        return batch


class Wav2Vec2DataCollator:
    """Data collator for Wav2Vec2 training"""
    
    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: List[Dict]) -> Dict:
        # Separate inputs and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad labels
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Replace padding with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].eq(0), -100
        )
        
        batch["labels"] = labels
        
        return batch


class ChildSpeechFineTuner:
    """Fine-tune models on children's speech data"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        
    def prepare_dataset(self, manifest_path: str, audio_dir: str):
        """Prepare dataset for training"""
        logger.info(f"Loading data from {manifest_path}")
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        df = pd.DataFrame(data)
        
        # Add full audio paths
        df['audio_path'] = df['audio_path'].apply(
            lambda x: os.path.join(audio_dir, x)
        )
        
        # Split train/val
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=42
        )
        
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Cast audio column
        train_dataset = train_dataset.cast_column(
            "audio_path", 
            Audio(sampling_rate=16000)
        )
        val_dataset = val_dataset.cast_column(
            "audio_path", 
            Audio(sampling_rate=16000)
        )
        
        return train_dataset, val_dataset
    
    def prepare_whisper_features(self, examples):
        """Prepare features for Whisper training"""
        audio = examples["audio_path"]
        
        # Process audio
        input_features = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="np"
        ).input_features[0]
        
        # Process labels
        labels = self.processor.tokenizer(
            examples["orthographic_text"],
            return_tensors="np"
        ).input_ids[0]
        
        return {
            "input_features": input_features,
            "labels": labels,
        }
    
    def prepare_wav2vec2_features(self, examples):
        """Prepare features for Wav2Vec2 training"""
        audio = examples["audio_path"]
        
        # Process audio
        input_values = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="np"
        ).input_values[0]
        
        # Process labels
        with self.processor.as_target_processor():
            labels = self.processor(
                examples["orthographic_text"],
                return_tensors="np"
            ).input_ids[0]
        
        return {
            "input_values": input_values,
            "labels": labels,
        }
    
    def compute_wer(self, pred):
        """Compute WER metric"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Handle different model outputs
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        
        # Replace -100 with pad token id
        label_ids = np.where(
            label_ids != -100, 
            label_ids, 
            self.processor.tokenizer.pad_token_id
        )
        
        # Decode
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = jiwer.wer(label_str, pred_str)
        
        return {"wer": wer}
    
    def train_whisper(self, train_dataset, val_dataset):
        """Fine-tune Whisper model"""
        logger.info("Loading Whisper model and processor...")
        
        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(
            self.config.base_model, 
            language="english", 
            task="transcribe"
        )
        
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.base_model
        )
        
        # Set forced decoder IDs
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="english", 
            task="transcribe"
        )
        self.model.config.suppress_tokens = []
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Prepare features
        logger.info("Preparing training features...")
        train_dataset = train_dataset.map(
            self.prepare_whisper_features,
            remove_columns=train_dataset.column_names,
            num_proc=4,
        )
        val_dataset = val_dataset.map(
            self.prepare_whisper_features,
            remove_columns=val_dataset.column_names,
            num_proc=4,
        )
        
        # Data collator
        data_collator = WhisperDataCollator(self.processor)
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=self.config.fp16,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=225,
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_wer,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save
        logger.info(f"Saving model to {self.config.output_dir}")
        trainer.save_model(self.config.output_dir)
        self.processor.save_pretrained(self.config.output_dir)
    
    def train_wav2vec2(self, train_dataset, val_dataset):
        """Fine-tune Wav2Vec2 model"""
        logger.info("Loading Wav2Vec2 model and processor...")
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(
            self.config.base_model
        )
        
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.config.base_model,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer),
        )
        
        # Freeze feature extractor
        self.model.freeze_feature_encoder()
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Prepare features
        logger.info("Preparing training features...")
        train_dataset = train_dataset.map(
            self.prepare_wav2vec2_features,
            remove_columns=train_dataset.column_names,
            num_proc=4,
        )
        val_dataset = val_dataset.map(
            self.prepare_wav2vec2_features,
            remove_columns=val_dataset.column_names,
            num_proc=4,
        )
        
        # Data collator
        data_collator = Wav2Vec2DataCollator(self.processor)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=self.config.fp16,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_wer,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save
        logger.info(f"Saving model to {self.config.output_dir}")
        trainer.save_model(self.config.output_dir)
        self.processor.save_pretrained(self.config.output_dir)
    
    def train(self, manifest_path: str, audio_dir: str):
        """Main training entry point"""
        # Prepare data
        train_dataset, val_dataset = self.prepare_dataset(manifest_path, audio_dir)
        
        # Train appropriate model
        if self.config.model_type == "whisper":
            self.train_whisper(train_dataset, val_dataset)
        elif self.config.model_type == "wav2vec2":
            self.train_wav2vec2(train_dataset, val_dataset)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune ASR models on children's speech")
    parser.add_argument("--model_type", type=str, default="whisper", choices=["whisper", "wav2vec2"])
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--manifest", type=str, required=True, help="Path to training manifest")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio directory")
    parser.add_argument("--output_dir", type=str, default="./models/finetuned")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_type=args.model_type,
        base_model=args.base_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )
    
    trainer = ChildSpeechFineTuner(config)
    trainer.train(args.manifest, args.audio_dir)


if __name__ == "__main__":
    main()
