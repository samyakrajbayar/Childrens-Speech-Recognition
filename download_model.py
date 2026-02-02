"""
Create Competition Submission - Model Download Script
Run this on a system with compatible torch/transformers to download Whisper model.

Usage:
  pip install torch transformers
  python download_model.py
  python package_submission.py
"""
import os
import sys
import shutil
import zipfile
from pathlib import Path

# Choose model size based on your requirements
# Options: openai/whisper-tiny (~150MB), openai/whisper-base (~290MB), 
#          openai/whisper-small (~960MB), openai/whisper-medium (~3GB),
#          openai/whisper-large-v3 (~6GB)
MODEL_ID = "openai/whisper-medium"  # Good balance of quality and size
OUTPUT_DIR = "whisper-medium"


def download_model():
    """Download Whisper model for offline use"""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    print(f"Downloading {MODEL_ID}...")
    print("This may take several minutes depending on your connection...")
    
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    
    # Calculate size
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    
    print(f"✓ Model saved to {OUTPUT_DIR}")
    print(f"  Total size: {total_size / 1e9:.2f} GB")
    return Path(OUTPUT_DIR)


def create_zip():
    """Create submission.zip with main.py and model"""
    output_zip = Path("submission.zip")
    
    print(f"\nCreating {output_zip}...")
    
    if output_zip.exists():
        output_zip.unlink()
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add main.py
        main_content = open("submission_main.py", "r").read()
        
        # Modify to use the bundled model path
        main_content = main_content.replace(
            'model_id = "openai/whisper-large-v3"',
            f'model_id = str(src_path / "{OUTPUT_DIR}")'
        )
        
        zf.writestr("main.py", main_content)
        print("  + main.py")
        
        # Add model directory
        model_path = Path(OUTPUT_DIR)
        if model_path.exists():
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(Path.cwd())
                    zf.write(file_path, arcname)
            print(f"  + {OUTPUT_DIR}/ (model weights)")
        else:
            print(f"  ⚠ Model directory not found: {OUTPUT_DIR}")
            print("    Run download first!")
            return False
    
    size_mb = output_zip.stat().st_size / 1e6
    print(f"\n✓ Created {output_zip} ({size_mb:.1f} MB)")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--package":
        create_zip()
    else:
        print("=" * 60)
        print("Step 1: Download Model")
        print("=" * 60)
        try:
            download_model()
        except Exception as e:
            print(f"✗ Error downloading model: {e}")
            print("\nMake sure you have torch and transformers installed:")
            print("  pip install torch transformers")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("Step 2: Create Submission ZIP")
        print("=" * 60)
        if create_zip():
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print("\nUpload submission.zip to DrivenData:")
            print("https://www.drivendata.org/competitions/308/childrens-word-asr/submissions/")
