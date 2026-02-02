"""
Create Competition Submission ZIP
Downloads Whisper model and packages everything for DrivenData submission.
"""
import os
import sys
import shutil
import zipfile
from pathlib import Path
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import torch
        import transformers
        import torchaudio
        print("✓ All dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: pip install torch torchaudio transformers")
        return False

def download_whisper_model(output_dir: Path, model_id: str = "openai/whisper-large-v3"):
    """Download Whisper model for offline use"""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    model_path = output_dir / "whisper-large-v3"
    
    if model_path.exists():
        print(f"✓ Model already exists at {model_path}")
        return model_path
    
    print(f"Downloading {model_id}...")
    print("This may take 10-20 minutes depending on your connection...")
    
    # Download processor
    print("  - Downloading processor...")
    processor = WhisperProcessor.from_pretrained(model_id)
    
    # Download model
    print("  - Downloading model weights (~6GB)...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    # Save locally
    print(f"  - Saving to {model_path}...")
    model_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(model_path)
    model.save_pretrained(model_path)
    
    print(f"✓ Model saved to {model_path}")
    return model_path

def create_submission_zip(project_dir: Path, output_zip: Path):
    """Create the submission ZIP file"""
    
    print(f"\nCreating submission ZIP...")
    
    # Files to include
    include_files = [
        "submission_main.py",  # Will be renamed to main.py
    ]
    
    include_dirs = [
        "whisper-large-v3",  # Model weights
    ]
    
    # Create ZIP
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add main.py (renamed from submission_main.py)
        main_py = project_dir / "submission_main.py"
        if main_py.exists():
            zf.write(main_py, "main.py")
            print(f"  + main.py (from submission_main.py)")
        else:
            print(f"  ✗ submission_main.py not found!")
            return False
        
        # Add model directory
        model_dir = project_dir / "whisper-large-v3"
        if model_dir.exists():
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(project_dir)
                    zf.write(file_path, arcname)
            print(f"  + whisper-large-v3/ (model weights)")
        else:
            print(f"  ⚠ Model directory not found - using HuggingFace cache fallback")
    
    # Check size
    size_gb = output_zip.stat().st_size / (1024**3)
    print(f"\n✓ Created {output_zip}")
    print(f"  Size: {size_gb:.2f} GB")
    
    if size_gb > 20:
        print("  ⚠ Warning: ZIP exceeds 20GB limit!")
    
    return True

def main():
    """Main function to create submission"""
    print("=" * 60)
    print("DrivenData Children's ASR - Submission Creator")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Paths
    project_dir = Path(__file__).parent
    output_zip = project_dir / "submission.zip"
    
    print(f"\nProject directory: {project_dir}")
    
    # Step 1: Download model
    print("\n" + "-" * 40)
    print("Step 1: Download Whisper Model")
    print("-" * 40)
    
    try:
        model_path = download_whisper_model(project_dir)
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        print("\nAlternative: The submission will use HuggingFace cache")
        print("This works if the competition runtime has network access during model loading")
    
    # Step 2: Create ZIP
    print("\n" + "-" * 40)
    print("Step 2: Create Submission ZIP")
    print("-" * 40)
    
    success = create_submission_zip(project_dir, output_zip)
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"\nSubmission file: {output_zip}")
        print("\nNext steps:")
        print("1. Go to https://www.drivendata.org/competitions/308/childrens-word-asr/")
        print("2. Login and enroll in the competition")
        print("3. Upload submission.zip for a SMOKE TEST first")
        print("4. If smoke test passes, submit for full evaluation")
    else:
        print("\n✗ Failed to create submission")
        sys.exit(1)


if __name__ == "__main__":
    main()
