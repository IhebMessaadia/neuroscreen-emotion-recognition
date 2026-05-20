Final video branch deployment package

This folder contains the final video branch deployment package for the multimodal emotion recognition project.

Final model: V8D
Model type: decision-level ensemble
Formula: V8D = 0.68 * V5 + 0.32 * V8C

Classes:
happy, sad, stressed, neutral

Confirmed CREMA-D actor-disjoint test metrics:
Accuracy: 0.7357
Macro-F1: 0.7302
Balanced Accuracy: 0.7350
Smoke test: PASS

The V8D video branch combines:
V5: DINOv2-based ensemble using selected V3 and V4 temporal heads.
V8C: face-cropped CLIP ViT-L/14 temporal branch.

Use:

from video_branch_v8d_inference import predict_video_emotion

result = predict_video_emotion("path/to/video.mp4")

The function returns:
{
  "pred_label": "...",
  "probabilities": {
    "happy": float,
    "sad": float,
    "stressed": float,
    "neutral": float
  },
  "confidence": float,
  "model_name": "V8D",
  "alpha": 0.68
}

The main multimodal project should use result["probabilities"] for late fusion with the other modality branches. Keep the class order fixed as happy, sad, stressed, neutral, and combine the probability vectors at the decision level using the multimodal fusion weights chosen in the main project. Do not fuse using only pred_label unless a hard-voting baseline is explicitly being evaluated.

The package uses paths relative to this deployment folder. It can be moved into another project as long as the assets folder and video_branch_v8d_bundle.pth stay next to video_branch_v8d_inference.py.

RAVDESS external evaluation was not completed and should not be reported as completed. The final confirmed results for this branch are CREMA-D test results only.
