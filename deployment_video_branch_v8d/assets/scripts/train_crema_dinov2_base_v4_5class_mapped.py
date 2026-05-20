import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


CLASS5 = ["happy", "sad", "anger", "fear", "neutral"]
CLASS4 = ["happy", "sad", "stressed", "neutral"]
CLASS5_TO_ID = {name: i for i, name in enumerate(CLASS5)}
CLASS4_TO_ID = {name: i for i, name in enumerate(CLASS4)}
EMOTION_TO_5 = {"HAP": "happy", "SAD": "sad", "ANG": "anger", "FEA": "fear", "NEU": "neutral"}


@dataclass
class V4Config:
    cache_dir: str = "cache/dinov2_base_16f_features"
    v2_output_dir: str = "runs/dinov2_base_cached_temporal"
    v3_output_dir: str = "runs/dinov2_base_v3_ensemble"
    output_dir: str = "runs/dinov2_base_v4_5class_mapped"
    batch_size: int = 64
    epochs: int = 80
    early_stopping_patience: int = 12
    seeds: List[int] = None
    heads: List[str] = None
    learning_rates: List[float] = None
    dropouts: List[float] = None
    weight_decays: List[float] = None
    label_smoothing: float = 0.05
    feature_noise_std: float = 0.01
    feature_dropout_p: float = 0.05
    sad_weight_multiplier: float = 1.2
    top_k_ensemble: int = 5
    hidden_dim: int = 384
    num_workers: int = 0
    mixed_precision: bool = True
    debug_max_videos: int = 0
    debug_epochs: int = 2
    calibration_objective: str = "macro_f1_plus_sad_bonus"
    sad_bonus_weight: float = 0.15

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [22, 33, 44, 55, 66]
        if self.heads is None:
            self.heads = ["bigru_attn", "transformer_light"]
        if self.learning_rates is None:
            self.learning_rates = [0.0001, 0.0002]
        if self.dropouts is None:
            self.dropouts = [0.35, 0.45]
        if self.weight_decays is None:
            self.weight_decays = [0.001, 0.003]


def load_config(path: Path) -> V4Config:
    cfg = V4Config()
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = set(asdict(cfg))
    for key, value in data.items():
        if key in valid:
            setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def emotion_from_name(value: str) -> Optional[str]:
    stem = Path(value).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    return parts[2].upper()


def map5_to_4_probs(probs5: np.ndarray) -> np.ndarray:
    out = np.zeros((probs5.shape[0], 4), dtype=np.float64)
    out[:, 0] = probs5[:, CLASS5_TO_ID["happy"]]
    out[:, 1] = probs5[:, CLASS5_TO_ID["sad"]]
    out[:, 2] = probs5[:, CLASS5_TO_ID["anger"]] + probs5[:, CLASS5_TO_ID["fear"]]
    out[:, 3] = probs5[:, CLASS5_TO_ID["neutral"]]
    return out


def label5_to_label4(label5: int) -> int:
    name = CLASS5[int(label5)]
    if name in {"anger", "fear"}:
        return CLASS4_TO_ID["stressed"]
    return CLASS4_TO_ID[name]


def load_v2_samples(cfg: V4Config) -> Dict[str, List[Dict[str, object]]]:
    split_path = Path(cfg.v2_output_dir) / "actor_group_split.json"
    manifest_path = Path(cfg.v2_output_dir) / "cache_manifest.csv"
    if not split_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Missing V2 actor_group_split.json or cache_manifest.csv")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    manifest = {}
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            manifest[row["video_id"]] = row
    result = {}
    for split_name in ["train", "val", "test"]:
        rows = []
        for item in split[split_name]:
            video_id = str(item["video_id"])
            row = manifest.get(video_id, {})
            video_path = str(item.get("path", row.get("video_path", "")))
            emotion = str(item.get("emotion_code") or emotion_from_name(video_id) or emotion_from_name(video_path))
            class5 = EMOTION_TO_5.get(emotion)
            if class5 is None:
                continue
            cache_path = Path(row.get("cache_path", item.get("cache_path", "")))
            if not cache_path.exists():
                raise FileNotFoundError(f"Missing cache for {video_id}: {cache_path}")
            label5 = CLASS5_TO_ID[class5]
            rows.append(
                {
                    "split": split_name,
                    "video_id": video_id,
                    "actor_id": str(item["actor_id"]),
                    "emotion_code": emotion,
                    "class5": class5,
                    "label5": label5,
                    "class4": CLASS4[label5_to_label4(label5)],
                    "label4": label5_to_label4(label5),
                    "video_path": video_path,
                    "cache_path": str(cache_path),
                }
            )
        result[split_name] = rows
    actor_overlap = (
        set(x["actor_id"] for x in result["train"]) & set(x["actor_id"] for x in result["val"])
        or set(x["actor_id"] for x in result["train"]) & set(x["actor_id"] for x in result["test"])
        or set(x["actor_id"] for x in result["val"]) & set(x["actor_id"] for x in result["test"])
    )
    if actor_overlap:
        raise RuntimeError("Actor leakage detected.")
    return result


def counts(samples: Sequence[Dict[str, object]], key: str, names: Sequence[str]) -> Dict[str, int]:
    c = Counter(str(x[key]) for x in samples)
    return {name: int(c.get(name, 0)) for name in names}


def print_split_report(samples_by_split):
    for split_name in ["train", "val", "test"]:
        samples = samples_by_split[split_name]
        actors = sorted(set(x["actor_id"] for x in samples))
        print(f"{split_name:5s}: videos={len(samples):4d} actors={len(actors):2d} 5class={counts(samples, 'class5', CLASS5)} 4class={counts(samples, 'class4', CLASS4)}")


def limit_debug(samples_by_split, max_videos: int):
    if max_videos <= 0:
        return samples_by_split
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    out = {}
    for split_name, samples in samples_by_split.items():
        target = max(5, int(round(max_videos * ratios[split_name])))
        by_class = defaultdict(list)
        for sample in samples:
            by_class[sample["class5"]].append(sample)
        selected = []
        pos = 0
        while len(selected) < target:
            added = False
            for name in CLASS5:
                items = by_class[name]
                if pos < len(items):
                    selected.append(items[pos])
                    added = True
                    if len(selected) >= target:
                        break
            if not added:
                break
            pos += 1
        out[split_name] = selected
    return out


class FeatureDataset(Dataset):
    def __init__(self, samples, train: bool, noise_std: float, dropout_p: float):
        self.samples = list(samples)
        self.train = train
        self.noise_std = float(noise_std)
        self.dropout_p = float(dropout_p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        data = torch.load(s["cache_path"], map_location="cpu")
        x = data["features"].float()
        if self.train:
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            if self.dropout_p > 0:
                mask = (torch.rand(x.shape[-1]) >= self.dropout_p).float()
                x = x * mask.view(1, -1)
        return {"features": x, "label5": torch.tensor(int(s["label5"]), dtype=torch.long), "path": s["video_path"]}


def collate(batch):
    return {
        "features": torch.stack([x["features"] for x in batch]),
        "labels5": torch.stack([x["label5"] for x in batch]),
        "paths": [x["path"] for x in batch],
    }


class AttentionPool(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, max(64, dim // 2)), nn.Tanh(), nn.Dropout(dropout), nn.Linear(max(64, dim // 2), 1))

    def forward(self, x):
        w = torch.softmax(self.net(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)


class TransformerLight(nn.Module):
    def __init__(self, input_dim, cfg, dropout):
        super().__init__()
        h = int(cfg.hidden_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, h)
        self.pos = nn.Parameter(torch.zeros(1, 16, h))
        layer = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=h * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.pool = AttentionPool(h, dropout)
        self.cls = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, len(CLASS5)))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        x = self.proj(self.norm(x))
        x = x + self.pos[:, : x.size(1), :]
        x = self.encoder(x)
        return self.cls(self.pool(x))


class BiGruAttn(nn.Module):
    def __init__(self, input_dim, cfg, dropout):
        super().__init__()
        h = int(cfg.hidden_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, h)
        self.gru = nn.GRU(h, h // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttentionPool(h, dropout)
        self.cls = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, len(CLASS5)))

    def forward(self, x):
        x = self.proj(self.norm(x))
        x, _ = self.gru(x)
        return self.cls(self.pool(x))


def build_model(head, input_dim, cfg, dropout):
    if head == "bigru_attn":
        return BiGruAttn(input_dim, cfg, dropout)
    if head == "transformer_light":
        return TransformerLight(input_dim, cfg, dropout)
    raise ValueError(head)


def first_dim(samples):
    return int(torch.load(samples[0]["cache_path"], map_location="cpu")["features"].shape[-1])


def plan(cfg):
    rows = []
    for si, seed in enumerate(cfg.seeds):
        for hi, head in enumerate(cfg.heads):
            rows.append(
                {
                    "seed": int(seed),
                    "head": head,
                    "learning_rate": float(cfg.learning_rates[(si + hi) % len(cfg.learning_rates)]),
                    "dropout": float(cfg.dropouts[(si + hi) % len(cfg.dropouts)]),
                    "weight_decay": float(cfg.weight_decays[(si + 2 * hi) % len(cfg.weight_decays)]),
                }
            )
    return rows


def sample_weights(samples):
    c = Counter(int(s["label5"]) for s in samples)
    return WeightedRandomSampler(torch.DoubleTensor([1.0 / max(1, c[int(s["label5"])]) for s in samples]), len(samples), replacement=True)


def class_weights(samples, cfg, device):
    y = np.array([int(s["label5"]) for s in samples])
    w = compute_class_weight("balanced", classes=np.arange(len(CLASS5)), y=y).astype(np.float32)
    w[CLASS5_TO_ID["sad"]] *= float(cfg.sad_weight_multiplier)
    return torch.tensor(w, dtype=torch.float32, device=device)


def metrics_from_probs(labels5, probs5):
    labels5 = np.asarray(labels5, dtype=np.int64)
    preds5 = probs5.argmax(axis=1)
    labels4 = np.asarray([label5_to_label4(x) for x in labels5], dtype=np.int64)
    probs4 = map5_to_4_probs(probs5)
    preds4 = probs4.argmax(axis=1)
    m5 = {
        "accuracy": float(accuracy_score(labels5, preds5)),
        "balanced_accuracy": float(balanced_accuracy_score(labels5, preds5)),
        "macro_f1": float(f1_score(labels5, preds5, average="macro", zero_division=0)),
    }
    m4 = {
        "accuracy": float(accuracy_score(labels4, preds4)),
        "balanced_accuracy": float(balanced_accuracy_score(labels4, preds4)),
        "macro_f1": float(f1_score(labels4, preds4, average="macro", zero_division=0)),
        "sad_f1": float(classification_report(labels4, preds4, labels=list(range(len(CLASS4))), target_names=CLASS4, zero_division=0, output_dict=True)["sad"]["f1-score"]),
    }
    return m5, m4, preds5, labels4, probs4, preds4


def write_pred(path, paths, labels, preds, probs, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "pred_label"] + [f"prob_{n}" for n in names])
        for p, y, pred, prob in zip(paths, labels, preds, probs):
            writer.writerow([p, names[int(y)], names[int(pred)]] + [f"{float(x):.8f}" for x in prob])


def save_cm(labels, preds, names, path, title):
    cm = confusion_matrix(labels, preds, labels=list(range(len(names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_loaders(samples, cfg):
    train_ds = FeatureDataset(samples["train"], True, cfg.feature_noise_std, cfg.feature_dropout_p)
    val_ds = FeatureDataset(samples["val"], False, 0, 0)
    test_ds = FeatureDataset(samples["test"], False, 0, 0)
    kwargs = {"batch_size": int(cfg.batch_size), "num_workers": int(cfg.num_workers), "pin_memory": torch.cuda.is_available(), "collate_fn": collate}
    return (
        DataLoader(train_ds, sampler=sample_weights(samples["train"]), **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )


def train_epoch(model, loader, criterion, optimizer, device, amp):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    labels, probs = [], []
    for batch in loader:
        x = batch["features"].to(device)
        y = batch["labels5"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        labels.extend(y.detach().cpu().tolist())
        probs.extend(torch.softmax(logits.detach().float(), dim=1).cpu().numpy().tolist())
    return metrics_from_probs(labels, np.asarray(probs))[1]


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    labels, probs, paths = [], [], []
    for batch in loader:
        x = batch["features"].to(device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(x)
        labels.extend(batch["labels5"].cpu().tolist())
        probs.extend(torch.softmax(logits.float(), dim=1).cpu().numpy().tolist())
        paths.extend(batch["paths"])
    probs = np.asarray(probs, dtype=np.float64)
    m5, m4, preds5, labels4, probs4, preds4 = metrics_from_probs(labels, probs)
    return {"labels5": np.asarray(labels), "probs5": probs, "preds5": preds5, "labels4": labels4, "probs4": probs4, "preds4": preds4, "m5": m5, "m4": m4, "paths": paths}


def train_one(run_cfg, idx, cfg, samples, device, epochs):
    seed_everything(int(run_cfg["seed"]))
    run_name = f"run_{idx:02d}_{run_cfg['head']}_seed{run_cfg['seed']}"
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_dim = first_dim(samples["train"])
    train_loader, val_loader, test_loader = make_loaders(samples, cfg)
    model = build_model(run_cfg["head"], input_dim, cfg, run_cfg["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(samples["train"], cfg, device), label_smoothing=float(cfg.label_smoothing))
    optimizer = AdamW(model.parameters(), lr=run_cfg["learning_rate"], weight_decay=run_cfg["weight_decay"])
    best, best_epoch, bad = -1.0, 0, 0
    start = time.time()
    history = []
    for epoch in range(1, int(epochs) + 1):
        tr = train_epoch(model, train_loader, criterion, optimizer, device, bool(cfg.mixed_precision))
        val = evaluate(model, val_loader, device, bool(cfg.mixed_precision))
        score = float(val["m4"]["macro_f1"])
        history.append({"epoch": epoch, "train_mapped_4class": tr, "val_internal_5class": val["m5"], "val_mapped_4class": val["m4"]})
        print(f"{run_name} epoch {epoch}/{epochs}: train_4mF1={tr['macro_f1']:.4f} val_4mF1={score:.4f} val_sadF1={val['m4']['sad_f1']:.4f}")
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "run_cfg": run_cfg, "input_dim": input_dim, "best_epoch": best_epoch, "best_val_mapped_macro_f1": best}, run_dir / "best_model.pt")
        else:
            bad += 1
        if bad >= int(cfg.early_stopping_patience):
            break
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val = evaluate(model, val_loader, device, bool(cfg.mixed_precision))
    test = evaluate(model, test_loader, device, bool(cfg.mixed_precision))
    write_pred(run_dir / "val_predictions.csv", val["paths"], val["labels4"], val["preds4"], val["probs4"], CLASS4)
    write_pred(run_dir / "test_predictions.csv", test["paths"], test["labels4"], test["preds4"], test["probs4"], CLASS4)
    write_pred(run_dir / "val_predictions_5class.csv", val["paths"], val["labels5"], val["preds5"], val["probs5"], CLASS5)
    write_pred(run_dir / "test_predictions_5class.csv", test["paths"], test["labels5"], test["preds5"], test["probs5"], CLASS5)
    payload = {
        "run_name": run_name,
        "run_cfg": run_cfg,
        "best_epoch": int(ckpt["best_epoch"]),
        "best_val_mapped_macro_f1": float(ckpt["best_val_mapped_macro_f1"]),
        "elapsed_seconds": time.time() - start,
        "val_internal_5class": val["m5"],
        "test_internal_5class": test["m5"],
        "val_mapped_4class": val["m4"],
        "test_mapped_4class": test["m4"],
        "val_mapped_4class_report": classification_report(val["labels4"], val["preds4"], labels=list(range(len(CLASS4))), target_names=CLASS4, zero_division=0, output_dict=True),
        "test_mapped_4class_report": classification_report(test["labels4"], test["preds4"], labels=list(range(len(CLASS4))), target_names=CLASS4, zero_division=0, output_dict=True),
        "history": history,
    }
    save_json(payload, run_dir / "metrics.json")
    return payload


def read_probs(path, names):
    paths, labels, probs = [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
            labels.append(names.index(row["true_label"]))
            probs.append([float(row[f"prob_{n}"]) for n in names])
    return paths, np.asarray(labels), np.asarray(probs)


def metric_payload(labels4, probs4):
    preds4 = probs4.argmax(axis=1)
    m4 = {
        "accuracy": float(accuracy_score(labels4, preds4)),
        "balanced_accuracy": float(balanced_accuracy_score(labels4, preds4)),
        "macro_f1": float(f1_score(labels4, preds4, average="macro", zero_division=0)),
        "sad_f1": float(classification_report(labels4, preds4, labels=list(range(len(CLASS4))), target_names=CLASS4, zero_division=0, output_dict=True)["sad"]["f1-score"]),
    }
    return m4, preds4


def calibrate(out, val_paths, val_labels, val_probs, test_paths, test_labels, test_probs, cfg):
    logits_val = np.log(np.clip(val_probs, 1e-8, 1.0))
    logits_test = np.log(np.clip(test_probs, 1e-8, 1.0))
    candidates = [-0.5, -0.25, 0.0, 0.25, 0.5]
    best_score, best_bias = -1.0, np.zeros(4)
    for b0 in candidates:
        for b1 in candidates:
            for b2 in candidates:
                for b3 in candidates:
                    bias = np.array([b0, b1, b2, b3])
                    p = softmax(logits_val + bias.reshape(1, -1))
                    m, _ = metric_payload(val_labels, p)
                    score = m["macro_f1"] + (float(cfg.sad_bonus_weight) * m["sad_f1"] if cfg.calibration_objective == "macro_f1_plus_sad_bonus" else 0.0)
                    if score > best_score:
                        best_score, best_bias = score, bias
    val_cal = softmax(logits_val + best_bias.reshape(1, -1))
    test_cal = softmax(logits_test + best_bias.reshape(1, -1))
    val_m, val_preds = metric_payload(val_labels, val_cal)
    test_m, test_preds = metric_payload(test_labels, test_cal)
    write_pred(out / "calibrated_ensemble_val_predictions.csv", val_paths, val_labels, val_preds, val_cal, CLASS4)
    write_pred(out / "calibrated_ensemble_test_predictions.csv", test_paths, test_labels, test_preds, test_cal, CLASS4)
    save_cm(val_labels, val_preds, CLASS4, out / "calibrated_ensemble_confusion_matrix_val_4class.png", "V4 calibrated val 4-class")
    save_cm(test_labels, test_preds, CLASS4, out / "calibrated_ensemble_confusion_matrix_test_4class.png", "V4 calibrated test 4-class")
    payload = {
        "bias": {n: float(v) for n, v in zip(CLASS4, best_bias)},
        "objective": cfg.calibration_objective,
        "val_mapped_4class": val_m,
        "test_mapped_4class": test_m,
        "test_mapped_4class_report": classification_report(test_labels, test_preds, labels=list(range(len(CLASS4))), target_names=CLASS4, zero_division=0, output_dict=True),
    }
    save_json(payload, out / "calibrated_ensemble_metrics.json")
    return payload


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ensemble(cfg):
    out = Path(cfg.output_dir)
    runs = []
    for mp in sorted(out.glob("run_*/metrics.json")):
        data = json.loads(mp.read_text(encoding="utf-8"))
        runs.append((float(data["best_val_mapped_macro_f1"]), mp.parent, data))
    if not runs:
        print("No V4 runs available for ensemble.")
        return None
    runs.sort(key=lambda x: x[0], reverse=True)
    selected = runs[: max(1, int(cfg.top_k_ensemble))]
    print("Selected V4 ensemble members:")
    for score, path, _ in selected:
        print(f"  {path.name}: val_mapped_macroF1={score:.4f}")
    val_paths, val_labels, val_probs = read_probs(selected[0][1] / "val_predictions.csv", CLASS4)
    test_paths, test_labels, test_probs = read_probs(selected[0][1] / "test_predictions.csv", CLASS4)
    val_stack, test_stack = [val_probs], [test_probs]
    for _, path, _ in selected[1:]:
        _, labels, probs = read_probs(path / "val_predictions.csv", CLASS4)
        if not np.array_equal(labels, val_labels):
            raise RuntimeError("Validation order mismatch")
        val_stack.append(probs)
        _, labels, probs = read_probs(path / "test_predictions.csv", CLASS4)
        if not np.array_equal(labels, test_labels):
            raise RuntimeError("Test order mismatch")
        test_stack.append(probs)
    val_avg = np.mean(np.stack(val_stack), axis=0)
    test_avg = np.mean(np.stack(test_stack), axis=0)
    val_m, val_preds = metric_payload(val_labels, val_avg)
    test_m, test_preds = metric_payload(test_labels, test_avg)
    write_pred(out / "ensemble_val_predictions.csv", val_paths, val_labels, val_preds, val_avg, CLASS4)
    write_pred(out / "ensemble_test_predictions.csv", test_paths, test_labels, test_preds, test_avg, CLASS4)
    save_cm(val_labels, val_preds, CLASS4, out / "ensemble_confusion_matrix_val_4class.png", "V4 ensemble val 4-class")
    save_cm(test_labels, test_preds, CLASS4, out / "ensemble_confusion_matrix_test_4class.png", "V4 ensemble test 4-class")
    payload = {
        "selected_runs": [p.name for _, p, _ in selected],
        "val_mapped_4class": val_m,
        "test_mapped_4class": test_m,
        "test_mapped_4class_report": classification_report(test_labels, test_preds, labels=list(range(len(CLASS4))), target_names=CLASS4, zero_division=0, output_dict=True),
    }
    save_json(payload, out / "ensemble_metrics.json")
    best = selected[0][2]
    save_json(best, out / "best_single_metrics.json")
    save_json({"ensemble": payload}, out / "4class_mapped_metrics.json")
    calibrate(out, val_paths, val_labels, val_avg, test_paths, test_labels, test_avg, cfg)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_v4_5class_mapped.json"))
    parser.add_argument("--debug-max-videos", type=int, default=None)
    parser.add_argument("--debug-epochs", type=int, default=None)
    parser.add_argument("--run-debug-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--ensemble-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.debug_max_videos is not None:
        cfg.debug_max_videos = int(args.debug_max_videos)
    if args.debug_epochs is not None:
        cfg.debug_epochs = int(args.debug_epochs)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(asdict(cfg), out / "used_config.json")
    if args.analyze_only:
        import analyze_v4_results

        analyze_v4_results.main()
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_debug_only:
        device = torch.device("cpu")
    print("=" * 90)
    print("CREMA-D V4 5-class internal, 4-class mapped")
    print("=" * 90)
    print(f"Python executable: {sys.executable}")
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'}")
    print(f"Training device for this command: {device}")
    if device.type != "cuda" and not args.run_debug_only:
        raise RuntimeError("CUDA is required for full V4 training.")
    samples = load_v2_samples(cfg)
    if args.run_debug_only or int(cfg.debug_max_videos) > 0:
        samples = limit_debug(samples, int(cfg.debug_max_videos))
    print_split_report(samples)
    save_json(
        {
            split: {"internal_5class": counts(rows, "class5", CLASS5), "mapped_4class": counts(rows, "class4", CLASS4), "videos": len(rows)}
            for split, rows in samples.items()
        },
        out / "label_counts.json",
    )
    if args.ensemble_only:
        ensemble(cfg)
        return
    if not (args.train or args.run_debug_only):
        print("No training started. Use --train for full V4 or --run-debug-only for smoke test.")
        return
    epochs = int(cfg.debug_epochs) if args.run_debug_only else int(cfg.epochs)
    rows = plan(cfg)
    print(f"Run plan: {len(rows)} runs | epochs per run: {epochs}")
    done = []
    for idx, run_cfg in enumerate(rows, start=1):
        print("=" * 90)
        print(f"Starting run {idx}/{len(rows)}: {run_cfg}")
        done.append(train_one(run_cfg, idx, cfg, samples, device, epochs))
    save_json({"completed_runs": len(done), "runs": done}, out / ("debug_summary.json" if args.run_debug_only else "training_summary.json"))
    save_json({"runs": [{"run": d["run_name"], "val_internal_5class": d["val_internal_5class"], "test_internal_5class": d["test_internal_5class"]} for d in done]}, out / "5class_internal_metrics.json")
    ensemble(cfg)
    print(f"Completed runs: {len(done)}")
    if args.run_debug_only:
        print("Debug-only run complete. Full V4 training was not started.")


if __name__ == "__main__":
    main()
