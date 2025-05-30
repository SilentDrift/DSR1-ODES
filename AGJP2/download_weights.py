"""Fetch the 1.5 B DeepSeek‑R1 distilled checkpoint (~3.6 GB)."""
from huggingface_hub import snapshot_download
import pathlib
import sys

MODEL_ID   = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TARGET_DIR = pathlib.Path("models/deepseek_r1_1_5b")

def main():
    print(f"🔄 Downloading {MODEL_ID}...")
    print(f"📁 Target directory: {TARGET_DIR.resolve()}")
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ Model downloaded successfully to {TARGET_DIR.resolve()}")
        
        # Verify download
        if (TARGET_DIR / "config.json").exists():
            print("✅ Download verified - config.json found")
        else:
            print("⚠️  Warning: config.json not found, download may be incomplete")
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("💡 Make sure you have internet connection and sufficient disk space (~4 GB)")
        sys.exit(1)

if __name__ == "__main__":
    main() 