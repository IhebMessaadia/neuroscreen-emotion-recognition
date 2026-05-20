"""Download the pretrained backbones (DINOv2 and CLIP) used by the video branch.

They are about 2 GB, so they are not stored in the repository. Run this once:

    python deployment_video_branch_v8d/download_models.py
"""
import os
from pathlib import Path

HF_CACHE = Path(__file__).resolve().parent / "assets" / "hf_cache"
os.environ["HF_HOME"] = str(HF_CACHE)

from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPModel  # noqa: E402

AutoImageProcessor.from_pretrained("facebook/dinov2-base")
AutoModel.from_pretrained("facebook/dinov2-base")
CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=str(HF_CACHE))
CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=str(HF_CACHE))
print(f"Models downloaded to {HF_CACHE}")
