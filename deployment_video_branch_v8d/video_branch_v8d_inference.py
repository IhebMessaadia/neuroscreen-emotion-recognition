from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn


DEPLOYMENT_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = DEPLOYMENT_DIR / "video_branch_v8d_bundle.pth"
ASSETS_DIR = DEPLOYMENT_DIR / "assets"
HF_CACHE = ASSETS_DIR / "hf_cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))

CLASS_NAMES = ["happy", "sad", "stressed", "neutral"]
CLASS5 = ["happy", "sad", "anger", "fear", "neutral"]
ALPHA = 0.68
V8C_WEIGHT = 0.32
DINOV2_MODEL_ID = "facebook/dinov2-base"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
FRAMES_PER_VIDEO = 16
FRAME_BATCH = 4
V8C_FEATURE_DIM = 2304


@dataclass
class V3Config:
    hidden_dim: int = 384
    transformer_layers: int = 1
    transformer_heads: int = 4

    def __post_init__(self) -> None:
        pass


@dataclass
class V4Config:
    hidden_dim: int = 384

    def __post_init__(self) -> None:
        pass


class AttentionPool(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(64, dim // 2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(64, dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.net(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class V3TransformerLight(nn.Module):
    def __init__(self, input_dim: int, cfg: V3Config, dropout: float):
        super().__init__()
        h = int(cfg.hidden_dim)
        self.in_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, h)
        self.pos = nn.Parameter(torch.zeros(1, 16, h))
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=int(cfg.transformer_heads),
            dim_feedforward=h * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(cfg.transformer_layers))
        self.pool = AttentionPool(h, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, len(CLASS_NAMES)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.in_norm(x))
        x = x + self.pos[:, : x.size(1), :]
        x = self.encoder(x)
        return self.classifier(self.pool(x))


class V3BiGruAttnHead(nn.Module):
    def __init__(self, input_dim: int, cfg: V3Config, dropout: float):
        super().__init__()
        h = int(cfg.hidden_dim)
        self.in_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, h)
        self.gru = nn.GRU(h, h // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttentionPool(h, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, len(CLASS_NAMES)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.in_norm(x))
        x, _ = self.gru(x)
        return self.classifier(self.pool(x))


class V4BiGruAttn(nn.Module):
    def __init__(self, input_dim: int, cfg: V4Config, dropout: float):
        super().__init__()
        h = int(cfg.hidden_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, h)
        self.gru = nn.GRU(h, h // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttentionPool(h, dropout)
        self.cls = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, len(CLASS5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm(x))
        x, _ = self.gru(x)
        return self.cls(self.pool(x))


def _rel_path(path: str) -> Path:
    return DEPLOYMENT_DIR / path


def _load_bundle() -> dict[str, Any]:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Missing deployment bundle: {BUNDLE_PATH}")
    return torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hf_output_to_tensor(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0]
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if isinstance(outputs, (tuple, list)) and outputs:
        return _hf_output_to_tensor(outputs[0])
    raise TypeError(f"Unsupported HuggingFace output type: {type(outputs)}")


def _sample_frames(video_path: Path) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count for video: {video_path}")
    indices = np.linspace(0, max(count - 1, 0), FRAMES_PER_VIDEO).round().astype(int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def _extract_dinov2_features(video_path: Path, device: torch.device) -> torch.Tensor:
    from transformers import AutoImageProcessor, AutoModel

    frames = _sample_frames(video_path)
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID, local_files_only=True).to(device)
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(frames), FRAME_BATCH):
            inputs = processor(images=frames[i : i + FRAME_BATCH], return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(**inputs)
            else:
                out = model(**inputs)
            emb = _hf_output_to_tensor(out)
            chunks.append(emb.float().detach().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0).unsqueeze(0)


def _build_v3_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = V3Config()
    for key, value in ckpt["run_cfg"].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.__post_init__()
    head = str(ckpt["run_cfg"]["head"])
    dropout = float(ckpt["run_cfg"]["dropout"])
    input_dim = int(ckpt.get("input_dim", 768))
    if head == "transformer_light":
        model = V3TransformerLight(input_dim, cfg, dropout)
    elif head == "bigru_attn":
        model = V3BiGruAttnHead(input_dim, cfg, dropout)
    else:
        raise ValueError(f"Unsupported V3 head in deployment: {head}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _build_v4_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = V4Config()
    for key, value in ckpt["run_cfg"].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.__post_init__()
    head = str(ckpt["run_cfg"]["head"])
    if head != "bigru_attn":
        raise ValueError(f"Unsupported V4 head in deployment: {head}")
    model = V4BiGruAttn(int(ckpt.get("input_dim", 768)), cfg, float(ckpt["run_cfg"]["dropout"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _softmax_with_bias(probs: np.ndarray, bias_by_label: dict[str, float]) -> np.ndarray:
    bias = np.array([float(bias_by_label[label]) for label in CLASS_NAMES], dtype=np.float64)
    logits = np.log(np.clip(probs.reshape(1, -1), 1e-8, 1.0)) + bias.reshape(1, -1)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True))[0]


def _predict_v3(features: torch.Tensor, asset_paths: dict[str, Any], device: torch.device) -> np.ndarray:
    v3_dir = _rel_path(asset_paths["v3_dir"])
    selected_runs = _load_json(v3_dir / "ensemble_metrics.json")["selected_runs"]
    calibration = _load_json(v3_dir / "calibrated_ensemble_metrics.json")
    probs = []
    with torch.no_grad():
        for run_name in selected_runs:
            model = _build_v3_model(v3_dir / run_name / "best_model.pt", device)
            logits = model(features.to(device))
            probs.append(torch.softmax(logits.float(), dim=1).detach().cpu().numpy()[0])
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    avg = np.mean(np.stack(probs), axis=0)
    return _softmax_with_bias(avg, calibration["bias"])


def _predict_v4(features: torch.Tensor, asset_paths: dict[str, Any], device: torch.device) -> np.ndarray:
    v4_dir = _rel_path(asset_paths["v4_dir"])
    selected_runs = _load_json(v4_dir / "ensemble_metrics.json")["selected_runs"]
    calibration = _load_json(v4_dir / "calibrated_ensemble_metrics.json")
    probs = []
    with torch.no_grad():
        for run_name in selected_runs:
            model = _build_v4_model(v4_dir / run_name / "best_model.pt", device)
            logits = model(features.to(device))
            p5 = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
            p4 = np.stack([p5[:, 0], p5[:, 1], p5[:, 2] + p5[:, 3], p5[:, 4]], axis=1)
            probs.append((p4 / np.maximum(p4.sum(axis=1, keepdims=True), 1e-12))[0])
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    avg = np.mean(np.stack(probs), axis=0)
    return _softmax_with_bias(avg, calibration["bias"])


def _predict_v5(features: torch.Tensor, asset_paths: dict[str, Any], device: torch.device) -> np.ndarray:
    v3_prob = _predict_v3(features, asset_paths, device)
    v4_prob = _predict_v4(features, asset_paths, device)
    recipe = _load_json(_rel_path(asset_paths["v5_recipe"]))
    weights = np.array(recipe["recipe"]["weights"], dtype=np.float64)
    v5_prob = weights * v3_prob + (1.0 - weights) * v4_prob
    return v5_prob / np.maximum(v5_prob.sum(), 1e-12)


def _largest_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    return max(boxes, key=lambda box: box[2] * box[3]) if boxes else None


def _expand_box(box: tuple[int, int, int, int], width: int, height: int, scale: float = 0.20) -> tuple[int, int, int, int]:
    x, y, w, h = box
    px, py = int(w * scale), int(h * scale)
    return max(0, x - px), max(0, y - py), min(width, x + w + px), min(height, y + h + py)


class FaceCropper:
    def __init__(self, face_model_path: Path) -> None:
        if not face_model_path.exists() or face_model_path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing MediaPipe face detector model: {face_model_path}")
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(face_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.45,
        )
        self.mp_detector = vision.FaceDetector.create_from_options(options)
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _mediapipe_box(self, rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.mp_detector.detect(image)
        boxes = []
        for det in result.detections:
            bb = det.bounding_box
            boxes.append((int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)))
        return _largest_box(boxes)

    def _haar_box(self, rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(35, 35))
        return _largest_box([(int(x), int(y), int(w), int(h)) for x, y, w, h in faces])

    def crop_frames(self, frames: list[Image.Image]) -> list[Image.Image]:
        crops: list[Image.Image] = []
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"))
            h, w = rgb.shape[:2]
            box = self._mediapipe_box(rgb) or self._haar_box(rgb)
            if box is None:
                crops.append(frame)
                continue
            x1, y1, x2, y2 = _expand_box(box, w, h)
            crops.append(Image.fromarray(rgb[y1:y2, x1:x2]))
        return crops


def _encode_clip_temporal(crops: list[Image.Image], device: torch.device) -> np.ndarray:
    from transformers import CLIPImageProcessor, CLIPModel

    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=str(HF_CACHE), local_files_only=True)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, cache_dir=str(HF_CACHE), local_files_only=True).to(device)
    model.eval()
    frame_embs = []
    with torch.no_grad():
        for start in range(0, len(crops), FRAME_BATCH):
            inputs = processor(images=crops[start : start + FRAME_BATCH], return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            pixel_values = inputs["pixel_values"]
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb = model.get_image_features(pixel_values=pixel_values)
            else:
                emb = model.get_image_features(pixel_values=pixel_values)
            emb = _hf_output_to_tensor(emb)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            frame_embs.append(emb.detach().cpu().numpy().astype(np.float32))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    arr = np.concatenate(frame_embs, axis=0)
    feature = np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr.max(axis=0)], axis=0).astype(np.float32)
    if feature.shape[0] != V8C_FEATURE_DIM:
        raise RuntimeError(f"Unexpected V8C feature dimension: {feature.shape[0]}")
    return feature.reshape(1, -1)


def _predict_v8c(video_path: Path, asset_paths: dict[str, Any], device: torch.device) -> np.ndarray:
    frames = _sample_frames(video_path)
    cropper = FaceCropper(_rel_path(asset_paths["face_detector"]))
    crops = cropper.crop_frames(frames)
    feature = _encode_clip_temporal(crops, device)
    with _rel_path(asset_paths["v8c_model"]).open("rb") as handle:
        clf = pickle.load(handle)
    probs = np.asarray(clf.predict_proba(feature), dtype=np.float64)
    classes = getattr(clf, "classes_", np.arange(len(CLASS_NAMES)))
    aligned = np.zeros((1, len(CLASS_NAMES)), dtype=np.float64)
    for src_idx, cls in enumerate(classes):
        aligned[:, int(cls)] = probs[:, src_idx]
    return aligned[0] / np.maximum(aligned[0].sum(), 1e-12)


def predict_video_emotion(video_path: str) -> dict[str, Any]:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video path does not exist: {path}")

    bundle = _load_bundle()
    asset_paths = bundle["asset_paths"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = _extract_dinov2_features(path, device)
    v5_prob = _predict_v5(features, asset_paths, device)
    v8c_prob = _predict_v8c(path, asset_paths, device)
    final_prob = ALPHA * v5_prob + V8C_WEIGHT * v8c_prob
    final_prob = final_prob / np.maximum(final_prob.sum(), 1e-12)

    pred_idx = int(final_prob.argmax())
    probabilities = {label: float(final_prob[i]) for i, label in enumerate(CLASS_NAMES)}
    confidence = float(final_prob[pred_idx])
    return {
        "pred_label": CLASS_NAMES[pred_idx],
        "probabilities": probabilities,
        "confidence": confidence,
        "model_name": "V8D",
        "alpha": ALPHA,
    }


def _main() -> int:
    import argparse
    import json as json_module

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    args = parser.parse_args()
    print(json_module.dumps(predict_video_emotion(args.video_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
