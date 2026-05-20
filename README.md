# NeuroScreen: Multimodal Emotion & Stress Recognition

Senior project at Karabük University, Department of Computer Engineering. NeuroScreen detects emotion and stress by combining **four signals**: facial video, EEG (brain activity), ECG (heart activity) and GSR (skin conductance). Each signal has its own model, and their probability outputs are fused at the decision level. A live **Tkinter dashboard** records from the webcam and shows the prediction.

| Signal | What it captures | Model |
|---|---|---|
| **Face / video** | Visual expression | ResNet-18 (images); DINOv2 + CLIP ViT-L/14 temporal ensemble (video) |
| **EEG** | Brain activity → emotional state | Random Forest |
| **ECG** | Heart rate → stress and excitement | Random Forest |
| **GSR** | Skin sweat → arousal and stress | Random Forest |

<!-- Add a screenshot of the dashboard: put an image in a docs/ folder and uncomment the line below -->
<!-- ![Dashboard](docs/dashboard.png) -->

## Highlights

- **Video branch (V8D)**: a decision-level ensemble `0.68 × V5 + 0.32 × V8C`
  - **V5**: DINOv2-base frame features with BiGRU-attention and light Transformer temporal heads, ensembled across seeds
  - **V8C**: face-cropped (MediaPipe) CLIP ViT-L/14 temporal model
  - **CREMA-D actor-disjoint test set:** 73.6% accuracy, 0.730 macro-F1, 0.735 balanced accuracy on 4 classes (happy, sad, stressed, neutral)
- **Late fusion** (`Fusion.py`): weighted average of class probabilities (face 0.35, GSR 0.25, EEG 0.20, ECG 0.20), with a physiological-only fallback when no video is available
- **Live dashboard** (`dashboard.py`): records 3 seconds from the webcam, runs every branch and displays the fused emotion with per-signal probabilities

## Repository structure

```
├── dashboard.py                  # Live Tkinter dashboard
├── Fusion.py                     # Decision-level fusion of all branches
├── train.py                      # Face image model (ResNet-18)
├── train_eeg.py / train_ecg.py / train_gsr.py   # Physiological models
├── *_model.pkl, face_model.pth   # Trained models (Git LFS)
├── eeg_feature_importance.png
└── deployment_video_branch_v8d/  # Video emotion branch
    ├── video_branch_v8d_inference.py   # predict_video_emotion(path)
    ├── test_video_branch.py            # Smoke test
    ├── download_models.py              # Fetches DINOv2 + CLIP (~2 GB)
    └── assets/                         # Trained heads, face detector, training scripts
```

## Getting started

Requires Python 3.10+ and [Git LFS](https://git-lfs.com/) (for the trained models).

```bash
git lfs install
git clone https://github.com/IhebMessaadia/NeuroScreenEmotionRecognition.git
cd NeuroScreenEmotionRecognition
pip install -r requirements.txt

# One-time download of the pretrained video backbones (~2 GB)
python deployment_video_branch_v8d/download_models.py

# Test the video branch on any short face video
python deployment_video_branch_v8d/test_video_branch.py --video path/to/video.mp4
```

Use the video branch from your own code:

```python
from video_branch_v8d_inference import predict_video_emotion

result = predict_video_emotion("clip.mp4")
print(result["pred_label"], result["probabilities"])
```

## Datasets

The datasets aren't included in this repository because of their size and licenses. To retrain the models or run the dashboard, put them in these folders:

| Folder | Used by |
|---|---|
| `Emotions Pics/train`, `Emotions Pics/test` | `train.py` (one sub-folder per class) |
| `EEG Emotion & Stress Recognition/EEG_Dataset_Labeled.csv` | `train_eeg.py`, `dashboard.py` |
| `ECG + GSR/ECG_Dataset_Labeled.csv`, `ECG + GSR/GSR_Dataset_Labeled.csv` | `train_ecg.py`, `train_gsr.py`, `dashboard.py` |

The video branch was trained and evaluated on [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D).

## Author

**Iheb Messaadia**: [GitHub](https://github.com/IhebMessaadia)
